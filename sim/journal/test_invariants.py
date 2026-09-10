"""
Journal edition -- executable checks of the paper's claims.

Every proposition the paper states about the implementation is checked here by
brute force on randomly generated observations and adversarial candidate sets.
These are the tests the manuscript refers to when it says the deployment
invariants are unit-tested; they run in a few seconds.

    python3 test_invariants.py

Author: Liang Dong.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import replace

import numpy as np

from core import (
    Action, Observation, LatentState, SCENARIOS, SAFE_FALLBACK,
    PHI_BASE, PHI_AUG, PREDICATE_SETS, P_MAX_POLICY,
    WAVEFORMS, MOD_CODES, BEAMS, ROUTES, SLICES, WORKLOADS, POWERS,
    step_environment, observe, admissible, violated_predicates, hazards,
    Twin, TwinParams,
)
from shield import (
    repair, split_predicates, COMPONENT_DOMAIN, GLOBAL_SCOPE,
    ShieldFilterController, ShieldRepairController,
)
from adversary import PLANNERS
from multicell import (
    admit_sequential, admit_per_cell, admit_static_share, g_power,
    cluster_budget, psi_violated, _best_admissible,
)
from experiments import load_confusion

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------

def random_observations(n: int, seed: int = 0) -> list[Observation]:
    """Observations drawn across all scenarios and phases, including the
    adversarial trust states and both perception conditions."""
    conf = load_confusion("radioml")
    out = []
    rng = np.random.default_rng(seed)
    for i, sc in enumerate(itertools.islice(itertools.cycle(SCENARIOS), n)):
        st = LatentState(rng=np.random.default_rng(seed * 1000 + i))
        st.cycle = int(rng.integers(0, 300))
        step_environment(st, SCENARIOS[sc])
        st.pa_temp_c = float(rng.uniform(35.0, 90.0))
        st.prio_deliv_ewma = float(rng.uniform(0.0, 1.0))
        st.incumbent_active = bool(rng.random() < 0.4)
        o = observe(st, Action(), SCENARIOS[sc],
                    confusion=conf if rng.random() < 0.5 else None)
        # Exercise the global-scope predicate too, which the scenarios alone
        # never trigger.
        if rng.random() < 0.15:
            o.twin_uncertainty = 0.9
        out.append(o)
    return out


def random_actions(n: int, seed: int = 0) -> list[Action]:
    """Uniform over the whole action space, so most draws are inadmissible."""
    rng = np.random.default_rng(seed)
    return [Action(waveform=str(rng.choice(WAVEFORMS)),
                   mod_code=str(rng.choice(MOD_CODES)),
                   beam=str(rng.choice(BEAMS)),
                   route=str(rng.choice(ROUTES)),
                   slice_=str(rng.choice(SLICES)),
                   workload=str(rng.choice(WORKLOADS)),
                   power_dbm=float(rng.choice(POWERS))) for _ in range(n)]


# ---------------------------------------------------------------------------
# Lemma 1: shield soundness
# ---------------------------------------------------------------------------

def test_lemma1_soundness():
    print("\nLemma 1 (shield soundness)")
    conf = load_confusion("radioml")
    twin = Twin(TwinParams(), seed=0, confusion=conf)
    bad_filter = bad_repair = 0
    total = 0
    for ps in ("phi", "phi_plus"):
        phi = PREDICATE_SETS[ps]
        for planner_kind in ("benign", "naive", "adaptive"):
            rng = np.random.default_rng(7)
            planner = PLANNERS[planner_kind](rng, k=5)
            for o in random_observations(150, seed=hash(ps) % 97):
                f = ShieldFilterController(planner=planner, phi=phi)
                r = ShieldRepairController(planner=planner, phi=phi)
                for ctrl, tag in ((f, "filter"), (r, "repair")):
                    a = ctrl.step(o, twin).action
                    total += 1
                    if not admissible(o, a, phi):
                        if tag == "filter":
                            bad_filter += 1
                        else:
                            bad_repair += 1
    check("actuated action is Phi-admissible for every (o, planner, Phi)",
          bad_filter == 0 and bad_repair == 0,
          f"{bad_filter} filter and {bad_repair} repair violations of {total}")


# ---------------------------------------------------------------------------
# Lemma 3: repair soundness on arbitrary actions
# ---------------------------------------------------------------------------

def test_lemma3_repair_sound():
    print("\nLemma 3 (repair soundness)")
    bad = 0
    n = 0
    for ps in ("phi", "phi_plus"):
        phi = PREDICATE_SETS[ps]
        for o in random_observations(120, seed=11):
            for a in random_actions(25, seed=13):
                out, _ = repair(o, a, phi)
                n += 1
                if not admissible(o, out, phi):
                    bad += 1
    check("repair(o, a) is admissible for arbitrary a", bad == 0,
          f"{bad} of {n}")


# ---------------------------------------------------------------------------
# Proposition 1: one-pass exactness and nearest-action optimality
# ---------------------------------------------------------------------------

def _cost(a: Action, b: Action) -> int:
    """Component-wise displacement along each ordered domain."""
    c = 0
    for comp, dom in COMPONENT_DOMAIN.items():
        try:
            c += abs(dom.index(getattr(a, comp)) - dom.index(getattr(b, comp)))
        except ValueError:
            c += len(dom)
    return c


def _brute_force_nearest(o: Observation, a: Action, phi) -> int:
    """Minimum displacement cost to any admissible action, by exhaustive
    search over the product of per-component domains."""
    best = None
    sep, _ = split_predicates(phi)
    # The objective and the constraints are both separable, so the exhaustive
    # minimum is the sum of the per-component minima; computing it that way
    # keeps the test linear instead of 20k-way.
    total = 0
    for comp, dom in COMPONENT_DOMAIN.items():
        preds = sep.get(comp, [])
        cur = getattr(a, comp)
        best_c = None
        for v in dom:
            trial = replace(a, **{comp: v})
            if any(p.fn(o, trial) for p in preds):
                continue
            c = abs(dom.index(v) - dom.index(cur))
            best_c = c if best_c is None else min(best_c, c)
        if best_c is None:
            return -1                     # component infeasible: not expected
        total += best_c
    return total


def test_prop1_nearest_action():
    print("\nProposition 1 (one-pass exactness, nearest action)")
    phi = PHI_BASE
    worse = 0
    n = 0
    for o in random_observations(80, seed=17):
        if any(p.fn(o, SAFE_FALLBACK) for p in phi):
            continue
        # skip observations where the global-scope predicate fires: there the
        # operator is defined to return the fallback, not a projection
        sep, glob = split_predicates(phi)
        for a in random_actions(20, seed=19):
            if any(p.fn(o, a) for p in glob):
                continue
            out, _ = repair(o, a, phi)
            opt = _brute_force_nearest(o, a, phi)
            if opt < 0:
                continue
            n += 1
            if _cost(a, out) > opt:
                worse += 1
    check("repair attains the minimum component-wise displacement",
          worse == 0, f"{worse} of {n} sub-optimal")


def test_prop1_single_pass():
    """The paper's O(KdL) repair cost is the SINGLE-pass cost, and the pass cap
    p_max only bounds the non-separable case.  Both deployed sets must therefore
    land in A_Phi(o) after one coordinate pass, not just the base set."""
    print("\nProposition 1 (single pass suffices)")
    for name, phi in (("Phi", PHI_BASE), ("Phi+", PHI_AUG)):
        _single_pass_for(name, phi)


def _single_pass_for(name, phi):
    multi = 0
    n = 0
    for o in random_observations(80, seed=23):
        sep, glob = split_predicates(phi)
        for a in random_actions(20, seed=29):
            if any(p.fn(o, a) for p in glob):
                continue
            # one manual pass over components
            out = a.copy()
            for comp, preds in sep.items():
                if any(p.fn(o, out) for p in preds):
                    dom = COMPONENT_DOMAIN[comp]
                    i0 = dom.index(getattr(out, comp))
                    for i in sorted(range(len(dom)), key=lambda i: (abs(i - i0), i)):
                        trial = replace(out, **{comp: dom[i]})
                        if not any(p.fn(o, trial) for p in preds):
                            out = trial
                            break
            n += 1
            if not admissible(o, out, phi):
                multi += 1
    check(f"{name}: a single coordinate pass already lands in A_Phi(o)",
          multi == 0, f"{multi} of {n} needed more than one pass")


def test_component_feasible_nonempty():
    """Algorithm 2 projects onto the nearest value in A^Phi_j(o).  That argmin
    is undefined if the set is empty, so the pseudocode falls back to the
    signed action's j-th component.  For the deployed sets the guard should
    never fire -- if this check ever fails, the guard has become load-bearing
    and the prose describing it must change."""
    print("\nAlgorithm 2 (per-component feasible sets are non-empty)")
    for name, phi in (("Phi", PHI_BASE), ("Phi+", PHI_AUG)):
        sep, glob = split_predicates(phi)
        empty = cells = 0
        for o in random_observations(120, seed=131):
            for a in random_actions(20, seed=137):
                if any(p.fn(o, a) for p in glob):
                    continue
                for comp, preds in sep.items():
                    cells += 1
                    dom = COMPONENT_DOMAIN[comp]
                    if not any(not any(p.fn(o, replace(a, **{comp: v})) for p in preds)
                               for v in dom):
                        empty += 1
        check(f"{name}: A^Phi_j(o) is never empty", empty == 0,
              f"{empty} empty of {cells} component checks")


# ---------------------------------------------------------------------------
# Proposition 2: coverage monotonicity
# ---------------------------------------------------------------------------

def test_prop2_coverage_monotone():
    print("\nProposition 2 (repair enlarges the admitted pool)")
    bad = 0
    empty_rep = 0
    n = 0
    for ps in ("phi", "phi_plus"):
        phi = PREDICATE_SETS[ps]
        rng = np.random.default_rng(31)
        planner = PLANNERS["naive"](rng, k=5)
        for o in random_observations(120, seed=37):
            cands = planner.propose(o)
            filt = {a.sig() for a in cands if admissible(o, a, phi)}
            rep = {(a if admissible(o, a, phi) else repair(o, a, phi)[0]).sig()
                   for a in cands}
            rep.add(SAFE_FALLBACK.sig())
            n += 1
            if not filt <= rep:
                bad += 1
            if not rep:
                empty_rep += 1
    check("C_filt is a subset of C_rep", bad == 0, f"{bad} of {n}")
    check("C_rep is never empty", empty_rep == 0, f"{empty_rep} of {n}")


# ---------------------------------------------------------------------------
# Theorem 2: cluster soundness
# ---------------------------------------------------------------------------

def test_thm2_cluster():
    print("\nTheorem 2 (cluster soundness under a coupled predicate)")
    conf = load_confusion("radioml")
    phi = PHI_BASE
    twin = Twin(TwinParams(), seed=3, confusion=conf)
    for scheme, fn, expect_sound in (("sequential", admit_sequential, True),
                                     ("static_share", admit_static_share, True),
                                     ("per_cell", admit_per_cell, False)):
        psi_bad = phi_bad = n = 0
        for C in (3, 7, 12):
            B = cluster_budget(C)
            rng = np.random.default_rng(41)
            planners = [PLANNERS["naive"](np.random.default_rng(41 + i), k=5)
                        for i in range(C)]
            obs_pool = random_observations(30, seed=43)
            for t in range(0, len(obs_pool) - C, C):
                obs = obs_pool[t:t + C]
                pools = [p.propose(o) for p, o in zip(planners, obs)]
                acts = fn(obs, pools, twin, phi, B)
                n += 1
                if psi_violated(acts, B):
                    psi_bad += 1
                if any(not admissible(o, a, phi) for o, a in zip(obs, acts)):
                    phi_bad += 1
        check(f"{scheme}: every per-cell action is Phi-admissible",
              phi_bad == 0, f"{phi_bad} of {n}")
        if expect_sound:
            check(f"{scheme}: joint action satisfies the cluster budget",
                  psi_bad == 0, f"{psi_bad} of {n}")
        else:
            check("per_cell: per-cell shielding DOES violate the budget "
                  "(the paper's negative result)", psi_bad > 0,
                  "expected violations, saw none")


# ---------------------------------------------------------------------------
# Deployment invariants I1-I3
# ---------------------------------------------------------------------------

def test_invariants():
    print("\nDeployment invariants I1--I3")
    conf = load_confusion("radioml")
    twin = Twin(TwinParams(), seed=5, confusion=conf)
    phi = PHI_AUG

    # I1: no unauthenticated planner output reaches the actuator unmodified.
    leaked = 0
    n = 0
    rng = np.random.default_rng(53)
    planner = PLANNERS["naive"](rng, k=5)
    for o in random_observations(150, seed=59):
        cands = planner.propose(o)
        a = ShieldRepairController(planner=planner, phi=phi).step(o, twin).action
        n += 1
        # if the actuated action equals a candidate, that candidate must have
        # been admissible in the first place
        for c in cands:
            if c.sig() == a.sig() and not admissible(o, c, phi):
                leaked += 1
    check("I1: no inadmissible planner candidate reaches the actuator",
          leaked == 0, f"{leaked} of {n}")

    # I2: no admitted action enlarges its own authority scope (power ceiling).
    escalated = 0
    for o in random_observations(150, seed=61):
        a = ShieldRepairController(planner=planner, phi=phi).step(o, twin).action
        if a.power_dbm > P_MAX_POLICY:
            escalated += 1
    check("I2: no admitted action exceeds the signed power ceiling",
          escalated == 0, f"{escalated}")

    # I3: the signed fallback is admissible in every reachable observation.
    bad_fb = 0
    for ps in ("phi", "phi_plus"):
        for o in random_observations(300, seed=67):
            if not admissible(o, SAFE_FALLBACK, PREDICATE_SETS[ps]):
                bad_fb += 1
    check("I3: the signed fallback is admissible in every observation",
          bad_fb == 0, f"{bad_fb}")


# ---------------------------------------------------------------------------
# Reproducibility of the twin
# ---------------------------------------------------------------------------

def test_twin_determinism():
    print("\nAssumption 2 (the twin is a function, not a fresh draw)")
    conf = load_confusion("radioml")
    t1 = Twin(TwinParams(), seed=9, confusion=conf)
    t2 = Twin(TwinParams(), seed=9, confusion=conf)
    diffs = 0
    for o in random_observations(50, seed=71):
        for a in random_actions(10, seed=73):
            if abs(t1(o, a)[0] - t1(o, a)[0]) > 0 or \
               abs(t1(o, a)[0] - t2(o, a)[0]) > 1e-12:
                diffs += 1
    check("repeated queries at the same (o, a) return the same estimate",
          diffs == 0, f"{diffs}")


# ---------------------------------------------------------------------------

def main() -> int:
    print("Invariant checks")
    test_lemma1_soundness()
    test_lemma3_repair_sound()
    test_prop1_nearest_action()
    test_prop1_single_pass()
    test_component_feasible_nonempty()
    test_prop2_coverage_monotone()
    test_thm2_cluster()
    test_invariants()
    test_twin_determinism()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: " + ", ".join(FAILURES))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
