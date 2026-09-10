"""
Journal edition -- the safety shield, the repair operator, and all
controllers under comparison.

A reject-only shield is a pure filter: inadmissible candidates are
discarded and, if none survived, the controller fell back to a signed safe
configuration.  That is sound but wasteful: the planner's *intent* is thrown
away together with its unsafe encoding.

This module adds the REPAIR operator R of Lemma 3.  Because the deployed
predicate set is component-separable -- every predicate constrains exactly one
component of the factored action -- the admissible set is a product set and can
be entered by projecting each violating component independently onto its own
feasible values.  Repair is therefore a single O(d) pass, is sound by
construction, and never returns the empty set, so the shielded controller stops
falling back.

Author: Liang Dong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core import (
    Action, Observation, LatentState, Predicate, Twin,
    PHI_BASE, PHI_AUG, SAFE_FALLBACK,
    WAVEFORMS, MOD_CODES, BEAMS, ROUTES, SLICES, WORKLOADS, POWERS,
    W_LOCKED, R_BLOCKLIST, P_MAX_POLICY, P_NOMINAL, MCS_REQ_DB,
    admissible, violated_predicates,
)

# ---------------------------------------------------------------------------
# Component algebra used by the repair operator
# ---------------------------------------------------------------------------
# Each component carries an ordered domain.  "Nearest feasible value" is
# measured by position in this order, so repair preserves as much of the
# planner's intent as the constraint permits.

COMPONENT_DOMAIN = {
    "waveform":  list(WAVEFORMS),
    "mod_code":  list(MOD_CODES),
    "beam":      list(BEAMS),
    "route":     list(ROUTES),
    "slice_":    list(SLICES),
    "workload":  list(WORKLOADS),
    "power_dbm": list(POWERS),
}
# Fallback value per component: always feasible, used as the last resort of the
# projection and as the anchor that guarantees A^j_Phi(o) is non-empty.
COMPONENT_FALLBACK = {
    "waveform": SAFE_FALLBACK.waveform, "mod_code": SAFE_FALLBACK.mod_code,
    "beam": SAFE_FALLBACK.beam, "route": SAFE_FALLBACK.route,
    "slice_": SAFE_FALLBACK.slice_, "workload": SAFE_FALLBACK.workload,
    "power_dbm": SAFE_FALLBACK.power_dbm,
}

GLOBAL_SCOPE = "*"


def split_predicates(phi: Sequence[Predicate]
                     ) -> tuple[dict[str, list[Predicate]], list[Predicate]]:
    """Partition Phi into the component-separable part (grouped by the single
    component each predicate constrains) and the global-scope part."""
    sep: dict[str, list[Predicate]] = {}
    glob: list[Predicate] = []
    for p in phi:
        if GLOBAL_SCOPE in p.scope:
            glob.append(p)
        elif len(p.scope) == 1:
            sep.setdefault(p.scope[0], []).append(p)
        elif p.terms:
            # Normalised form: project each component against only the
            # singleton-scope disjunct that constrains it.  Without this, a
            # candidate that violates the power disjunct of phi_rf would make
            # *every* waveform look infeasible and the projection would discard
            # the planner's waveform choice for no reason.
            for comp, fn in p.terms:
                sep.setdefault(comp, []).append(
                    Predicate(f"{p.name}@{comp}", (comp,), fn))
        else:
            # Not normalised: fall back to iterating the whole predicate to a
            # fixed point.  Still sound (Lemma 2), but no longer nearest-action.
            for c in p.scope:
                sep.setdefault(c, []).append(p)
    return sep, glob


def _component_feasible(o: Observation, a: Action, comp: str, value,
                        preds: Sequence[Predicate]) -> bool:
    trial = a.copy()
    setattr(trial, comp, value)
    return not any(p.fn(o, trial) for p in preds)


def repair(o: Observation, a: Action, phi: Sequence[Predicate],
           max_passes: int = 3) -> tuple[Action, list[str]]:
    """Repair operator R of Lemma 3.

    Returns (a_repaired, list_of_components_edited).  The returned action is
    Phi-admissible whenever the global-scope part of Phi does not fire; if it
    does, the signed fallback is returned, which is admissible by construction.
    """
    sep, glob = split_predicates(phi)
    if any(p.fn(o, a) for p in glob):
        return SAFE_FALLBACK.copy(), [GLOBAL_SCOPE]

    out = a.copy()
    edited: list[str] = []
    for _ in range(max_passes):
        if admissible(o, out, phi):
            break
        for comp, preds in sep.items():
            cur = getattr(out, comp)
            if _component_feasible(o, out, comp, cur, preds):
                continue
            domain = COMPONENT_DOMAIN[comp]
            try:
                i0 = domain.index(cur)
            except ValueError:
                i0 = 0
            # Search outward from the requested value: nearest feasible value
            # in the component's ordered domain.
            order = sorted(range(len(domain)), key=lambda i: (abs(i - i0), i))
            chosen = COMPONENT_FALLBACK[comp]
            for i in order:
                if _component_feasible(o, out, comp, domain[i], preds):
                    chosen = domain[i]
                    break
            setattr(out, comp, chosen)
            edited.append(comp)
    if not admissible(o, out, phi):
        return SAFE_FALLBACK.copy(), edited + ["fallback"]
    return out, edited


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    action: Action
    mode: str                  # admit | repair | fallback | unshielded | ...
    n_candidates: int = 0
    n_admitted: int = 0
    n_repaired: int = 0


class BaseController:
    """Every controller takes the same constructor signature so the experiment
    driver can build any of them uniformly."""
    name = "base"

    def __init__(self, planner=None, phi=PHI_BASE, **kw):
        self.planner = planner
        self.phi = phi

    def reset(self, rng: np.random.Generator) -> None:
        pass

    def step(self, o: Observation, twin: Twin) -> Decision:
        raise NotImplementedError

    def post(self, o: Observation, pdr: float) -> None:
        """Called after actuation with the realised outcome."""
        pass


class StaticController(BaseController):
    name = "static"

    def step(self, o, twin):
        return Decision(Action(), "unshielded")


class HeuristicController(BaseController):
    """Perception-conditioned hand rule built with oracle knowledge of which
    adaptation fits which interference class.  Strong, zero-Phi-violation,
    non-adaptive to anything outside its design envelope."""
    name = "heuristic"

    def step(self, o, twin):
        a = Action(power_dbm=26.0, slice_="urgent")
        if o.interference_class == "narrowband":
            a.waveform, a.mod_code, a.beam = "ofdm_n1", "qpsk_1_2", "narrow"
        elif o.interference_class == "wideband":
            a.waveform, a.mod_code, a.beam = "ofdm_n0", "qpsk_1_3", "narrow"
            a.route = "secondary"
        elif o.interference_class == "bursty":
            a.waveform, a.mod_code, a.beam = "ofdm_n1", "qpsk_1_2", "sector_a"
        else:
            a.mod_code = "16qam_1_2"
        if not o.backhaul_ok:
            a.route, a.workload = "mesh_a", "local"
        return Decision(a, "unshielded")


class GreedyTwinController(BaseController):
    """Unconstrained greedy on twin-predicted reward over the planner's
    candidates.  No shield.  Stands in for an unconstrained learned policy."""
    name = "greedy_twin"

    def step(self, o, twin):
        cands = self.planner.propose(o)
        best = max(cands, key=lambda a: twin(o, a)[0])
        return Decision(best, "unshielded", n_candidates=len(cands))


class LLMOnlyController(BaseController):
    """Direct actuation of the planner's own top-ranked candidate.

    This is the "let the agent drive" configuration: no digital twin, no
    re-ranking, no shield -- whatever the planner puts first reaches the radio.
    It differs from Greedy-Twin, which at least re-ranks the candidate set with
    an independent model, and it is the configuration a naive LLM-in-the-loop
    RIC deployment would produce.
    """
    name = "llm_only"

    def step(self, o, twin):
        cands = self.planner.propose(o)
        return Decision(cands[0].copy(), "unshielded", n_candidates=len(cands))


class ShieldFilterController(BaseController):
    """Conference the shielded controller: reject-only shield plus signed fallback."""
    name = "shield_filter"

    def step(self, o, twin):
        cands = self.planner.propose(o)
        adm = [a for a in cands if admissible(o, a, self.phi)]
        if not adm:
            return Decision(SAFE_FALLBACK.copy(), "fallback",
                            n_candidates=len(cands), n_admitted=0)
        best = max(adm, key=lambda a: twin(o, a)[0])
        return Decision(best, "admit", n_candidates=len(cands),
                        n_admitted=len(adm))


class ShieldRepairController(BaseController):
    """Journal the shielded controller: admissible candidates are kept, inadmissible ones
    are projected back into A_Phi(o) by the repair operator, and the signed
    fallback is always appended to the candidate set."""
    name = "shield_repair"

    def step(self, o, twin):
        cands = self.planner.propose(o)
        pool, n_adm, n_rep = [], 0, 0
        for a in cands:
            if admissible(o, a, self.phi):
                pool.append(a)
                n_adm += 1
            else:
                a_r, _ = repair(o, a, self.phi)
                pool.append(a_r)
                n_rep += 1
        pool.append(SAFE_FALLBACK.copy())
        best = max(pool, key=lambda a: twin(o, a)[0])
        mode = "admit" if n_adm and best.sig() in {c.sig() for c in cands} else "repair"
        return Decision(best, mode, n_candidates=len(cands),
                        n_admitted=n_adm, n_repaired=n_rep)


class SimplexRTAController(BaseController):
    """Classical Simplex / runtime-assurance baseline.

    The high-performance (unshielded) controller drives the plant.  A reactive
    decision module monitors realised telemetry and, when a safety condition is
    observed to have been breached, switches to the verified baseline
    controller for a dwell period.  The switch is necessarily POST hoc: the
    breach is observed only after the offending action has been actuated.  This
    is the baseline that isolates the value of pre-actuation filtering.
    """
    name = "simplex_rta"

    def __init__(self, planner=None, phi=PHI_BASE, dwell: int = 10, **kw):
        super().__init__(planner, phi)
        self.dwell = dwell
        self.timer = 0
        self.baseline = HeuristicController()
        self.last_o: Observation | None = None
        self.last_a: Action | None = None

    def reset(self, rng):
        self.timer = 0
        self.last_o = None
        self.last_a = None

    def step(self, o, twin):
        # Reactive monitor: inspect the PREVIOUS actuated action against the
        # PREVIOUS telemetry.  This is what a post-hoc monitor can observe.
        if self.last_o is not None and self.last_a is not None:
            if not admissible(self.last_o, self.last_a, self.phi):
                self.timer = self.dwell
        if self.timer > 0:
            self.timer -= 1
            d = self.baseline.step(o, twin)
            self.last_o, self.last_a = o, d.action
            return Decision(d.action, "rta_baseline")
        cands = self.planner.propose(o)
        best = max(cands, key=lambda a: twin(o, a)[0])
        self.last_o, self.last_a = o, best
        return Decision(best, "rta_performance", n_candidates=len(cands))


class LagrangianRLController(BaseController):
    """Lagrangian-constrained tabular Q-learning over the planner's candidate
    set -- the expectation-constrained safe-RL family (CPO / RCPO / PPO-Lag).

    The agent learns Q_r (reward) and Q_c (constraint cost) on a discretised
    observation, selects argmax_a [Q_r - lambda Q_c], and updates lambda by
    dual ascent towards a cost budget d.  It enforces the constraint only in
    expectation, which is exactly the property the paper argues is inadequate
    for actuation safety.
    """
    name = "lagrangian_rl"

    def __init__(self, planner=None, phi=PHI_BASE, budget: float = 0.0,
                 lr: float = 0.30, lam_lr: float = 0.05, eps: float = 0.10, **kw):
        super().__init__(planner, phi)
        self.budget = budget
        self.lr = lr
        self.lam_lr = lam_lr
        self.eps = eps
        self.lam = 1.0
        self.Qr: dict = {}
        self.Qc: dict = {}
        self.rng = np.random.default_rng(0)
        self._pending = None

    def reset(self, rng):
        self.rng = rng
        self.lam = 1.0
        self.Qr, self.Qc = {}, {}
        self._pending = None

    @staticmethod
    def _obs_key(o: Observation) -> tuple:
        return (o.interference_class, bool(o.backhaul_ok),
                int(np.clip(o.queue_priority * 4, 0, 3)),
                o.trust_state)

    @staticmethod
    def _act_key(a: Action) -> tuple:
        return (a.waveform, a.mod_code, a.beam, a.route, a.slice_,
                a.workload, round(a.power_dbm))

    def step(self, o, twin):
        cands = self.planner.propose(o)
        s = self._obs_key(o)
        if self.rng.random() < self.eps:
            best = cands[int(self.rng.integers(len(cands)))]
        else:
            def score(a):
                k = (s, self._act_key(a))
                qr = self.Qr.get(k, twin(o, a)[0])   # twin as optimistic prior
                qc = self.Qc.get(k, 0.0)
                return qr - self.lam * qc
            best = max(cands, key=score)
        self._pending = (s, self._act_key(best))
        return Decision(best, "unshielded", n_candidates=len(cands))

    def learn(self, r: float, c: float) -> None:
        if self._pending is None:
            return
        k = self._pending
        self.Qr[k] = (1 - self.lr) * self.Qr.get(k, r) + self.lr * r
        self.Qc[k] = (1 - self.lr) * self.Qc.get(k, c) + self.lr * c
        # dual ascent on the constraint budget
        self.lam = float(np.clip(self.lam + self.lam_lr * (c - self.budget),
                                 0.0, 50.0))
        self._pending = None


CONTROLLERS = {
    "static": StaticController,
    "heuristic": HeuristicController,
    "greedy_twin": GreedyTwinController,
    "llm_only": LLMOnlyController,
    "lagrangian_rl": LagrangianRLController,
    "simplex_rta": SimplexRTAController,
    "shield_filter": ShieldFilterController,
    "shield_repair": ShieldRepairController,
}

NEEDS_PLANNER = {"greedy_twin", "llm_only", "lagrangian_rl", "simplex_rta",
                 "shield_filter", "shield_repair"}

CONTROLLER_LABEL = {
    "static": "Static",
    "heuristic": "Heuristic",
    "greedy_twin": "Greedy-Twin",
    "llm_only": "LLM-only",
    "lagrangian_rl": "Lagrangian-RL",
    "simplex_rta": "Simplex-RTA",
    "shield_filter": "the shielded controller (filter)",
    "shield_repair": "the shielded controller (repair)",
}
