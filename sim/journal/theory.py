"""
Journal edition -- empirical verification of Theorem 1.

A supremum-and-exact-match instantiation of the bound uses

    xi   = sup_{o,a} |r_hat - r_bar|          (an unbounded-in-practice sup)
    cov  = Pr[ pi_dagger(o) is EXACTLY one of the K candidates ]

which produced xi ~ 1.0 and cov ~ 1e-3, hence a bound of ~2.99 on a quantity
that is bounded by 1 by construction.  The bound was true and vacuous.

This module instantiates the journal form of the bound:

    xi_delta  = the (1 - delta) empirical quantile of the per-cycle maximum
                twin error over the ADMITTED candidate pool (Assumption A2');
    cov_{K,eps} = Pr[ the candidate pool contains SOME admissible action whose
                true mean reward is within eps of pi_dagger's ]  (Assumption A3').

so that, per cycle,

    E[ r_bar(o, pi_dagger(o)) - r_bar(o, a*) ]
        <= 2 xi_delta + eps + r_max (1 - cov_{K,eps}) + r_max delta.

We report the bound and the measured regret per scenario, and -- more
informatively -- we SWEEP the two quantities the bound is stated in terms of
(twin mismatch and candidate budget K) and check that the bound tracks the
measured regret monotonically.

Author: Liang Dong.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from core import (
    Action, LatentState, Observation, Twin, TwinParams,
    SCENARIOS, SCENARIO_ORDER, SAFE_FALLBACK, SAFE_POOL,
    PHI_BASE, PREDICATE_SETS,
    step_environment, observe, realised_pdr, update_thermal,
    update_priority_ewma, admissible,
)
from shield import repair, ShieldRepairController, ShieldFilterController
from adversary import PLANNERS
from experiments import load_confusion
import conditional as cond

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True, parents=True)

R_MAX = 1.0


# ---------------------------------------------------------------------------
# Instrumented shielded run
# ---------------------------------------------------------------------------

@dataclass
class TheoryRow:
    scenario: str
    seed: int
    cycle: int
    xi_cycle: float          # max over the two points the proof of Thm 1 uses
    xi_pool: float           # max over the whole evaluated pool (diagnostic)
    r_star: float            # r_bar(o, a*)
    r_dagger: float          # r_bar(o, pi_dagger(o))
    regret: float            # on-trajectory one-step gap of Theorem 1
    cov_eps: int             # pool contains an eps-optimal admissible action
    cov_exact: int           # pi_dagger itself is in the pool
    pool_size: int
    r_clairvoyant: float     # omniscient benchmark (latent state known)
    perception_gap: float    # r_clairvoyant - r_dagger: price of misperception


def instrumented_run(scenario: str, seed: int, cycles: int = 300, k: int = 5,
                     eps: float = 0.05, planner_kind: str = "naive",
                     predicate_set: str = "phi",
                     twin_params: TwinParams | None = None,
                     confusion: dict | None = None,
                     repair_on: bool = True,
                     model: "cond.ConditionalModel | None" = None,
                     ) -> list[TheoryRow]:
    rng = np.random.default_rng(seed)
    st = LatentState(rng=rng)
    cfg = SCENARIOS[scenario]
    phi = PREDICATE_SETS[predicate_set]
    twin = Twin(twin_params or TwinParams(), seed=seed, confusion=confusion)
    planner = PLANNERS[planner_kind](rng, k=k)
    model = model or cond.build(scenario, confusion)

    a_prev = Action()
    rows: list[TheoryRow] = []

    for c in range(cycles):
        st.cycle = c
        step_environment(st, cfg)
        o = observe(st, a_prev, cfg, confusion=confusion)
        twin.observe_label(o.interference_class)

        cands = planner.propose(o)
        if repair_on:
            pool = [a if admissible(o, a, phi) else repair(o, a, phi)[0]
                    for a in cands]
            pool.append(SAFE_FALLBACK.copy())
        else:
            pool = [a for a in cands if admissible(o, a, phi)] or [SAFE_FALLBACK.copy()]

        ok = cond.obs_key(o)
        r_hat = np.array([twin(o, a)[0] for a in pool])
        r_bar = np.array([model.r_bar(ok, a) for a in pool])
        err = np.abs(r_hat - r_bar)

        j = int(np.argmax(r_hat))        # the action the shield actuates
        b = int(np.argmax(r_bar))        # the best admissible candidate present
        # Assumption A2' is only ever invoked at these two points in the proof
        # of Theorem 1, so this is the faithful empirical instantiation of xi.
        xi_cycle = float(max(err[j], err[b]))
        xi_pool = float(err.max())

        a_star, r_star = pool[j], float(r_bar[j])
        # pi_dagger of Theorem 1 is the maximum over A_Phi(o), so it searches
        # the whole action space under the observation's own predicates -- not
        # a template pool, and not without applying Phi.
        a_dag, r_dag = model.pi_dagger_full(ok, o, phi)
        _, r_clair = model.clairvoyant(st, o, phi)

        cov_eps = int(np.any(r_bar >= r_dag - eps))
        cov_exact = int(any(a.sig() == a_dag.sig() for a in pool))

        rows.append(TheoryRow(
            scenario=scenario, seed=seed, cycle=c, xi_cycle=xi_cycle,
            xi_pool=xi_pool, r_star=r_star, r_dagger=float(r_dag),
            regret=float(r_dag - r_star),
            cov_eps=cov_eps, cov_exact=cov_exact, pool_size=len(pool),
            r_clairvoyant=float(r_clair),
            # NOT clipped: r_clair uses the sampled latent state while
            # r_dag is a conditional optimum, so a single cycle may be
            # negative.  The information-value inequality holds in
            # expectation, which is what we report.
            perception_gap=float(r_clair - r_dag),
        ))

        pdr = realised_pdr(st, a_star)
        update_thermal(st, a_star)
        update_priority_ewma(st, pdr, a_star)
        a_prev = a_star

    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame, eps: float, delta: float,
              by: list[str]) -> pd.DataFrame:
    out = []
    for key, sub in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        xi_delta = float(np.quantile(sub["xi_cycle"], 1.0 - delta))
        xi_sup = float(sub["xi_cycle"].max())
        cov = float(sub["cov_eps"].mean())
        cov_exact = float(sub["cov_exact"].mean())
        measured = float(sub["regret"].mean())
        bound = 2 * xi_delta + eps + R_MAX * (1 - cov) + R_MAX * delta
        bound_sup = 2 * xi_sup + R_MAX * (1 - cov_exact)   # supremum / exact-match form
        rec = dict(zip(by, key))
        rec.update(dict(
            eps=eps, delta=delta,
            perception_gap=float(sub["perception_gap"].mean()),
            xi_delta=xi_delta, xi_sup=xi_sup, xi_mean=float(sub["xi_cycle"].mean()),
            xi_pool_delta=float(np.quantile(sub["xi_pool"], 1.0 - delta)),
            cov_eps=cov, cov_exact=cov_exact,
            measured_regret=measured,
            bound_twin=2 * xi_delta, bound_eps=eps,
            bound_cov=R_MAX * (1 - cov), bound_delta=R_MAX * delta,
            bound_total=bound, bound_sup=bound_sup,
            slack=bound - measured, tightness=measured / bound if bound > 0 else np.nan,
            n_cycles=int(len(sub)),
        ))
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--perception", default="radioml")
    args = ap.parse_args()

    conf = load_confusion(args.perception)
    seeds = list(range(args.seeds))
    models = {sc: cond.get(sc, conf, args.perception) for sc in SCENARIO_ORDER}

    # --- (T1) per-scenario verification at the deployed configuration -------
    print("[T1] per-scenario bound verification ...", flush=True)
    rows = []
    for sc in SCENARIO_ORDER:
        for sd in seeds:
            rows += instrumented_run(sc, sd, cycles=args.cycles, eps=args.eps,
                                     confusion=conf, model=models[sc])
    df = pd.DataFrame([asdict(r) for r in rows])
    t1 = summarise(df, args.eps, args.delta, ["scenario"])
    t1.to_csv(RESULTS / "theorem_scenario.csv", index=False)
    print(t1[["scenario", "xi_delta", "cov_eps", "measured_regret",
              "bound_total", "bound_sup"]].to_string(index=False))

    # --- (T2) twin-mismatch sweep ------------------------------------------
    print("\n[T2] twin-mismatch sweep ...", flush=True)
    rows = []
    for m in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
        for sigma in (0.08,):
            tp = TwinParams(mismatch=m, sigma=sigma)
            for sc in SCENARIO_ORDER:
                for sd in seeds[:5]:
                    for r in instrumented_run(sc, sd, cycles=args.cycles,
                                              eps=args.eps, twin_params=tp,
                                              confusion=conf, model=models[sc]):
                        d = asdict(r)
                        d["mismatch"] = m
                        d["sigma"] = sigma
                        rows.append(d)
    t2 = summarise(pd.DataFrame(rows), args.eps, args.delta, ["mismatch", "sigma"])
    t2.to_csv(RESULTS / "theorem_mismatch.csv", index=False)
    print(t2[["mismatch", "xi_delta", "cov_eps", "measured_regret",
              "bound_total"]].to_string(index=False))

    # --- (T3) candidate-budget sweep ---------------------------------------
    print("\n[T3] candidate-budget sweep ...", flush=True)
    rows = []
    for k in (1, 2, 3, 5, 8, 12, 20):
        for sc in SCENARIO_ORDER:
            for sd in seeds[:5]:
                for r in instrumented_run(sc, sd, cycles=args.cycles, k=k,
                                          eps=args.eps, confusion=conf,
                                          model=models[sc]):
                    d = asdict(r)
                    d["K"] = k
                    rows.append(d)
    t3 = summarise(pd.DataFrame(rows), args.eps, args.delta, ["K"])
    t3.to_csv(RESULTS / "theorem_budget.csv", index=False)
    print(t3[["K", "xi_delta", "cov_eps", "measured_regret",
              "bound_total"]].to_string(index=False))

    # --- (T4) repair vs filter: effect on coverage (Proposition 2) ---------
    print("\n[T4] repair vs filter coverage ...", flush=True)
    rows = []
    for rep in (True, False):
        for sc in SCENARIO_ORDER:
            for sd in seeds:
                for r in instrumented_run(sc, sd, cycles=args.cycles,
                                          eps=args.eps, confusion=conf,
                                          repair_on=rep, model=models[sc]):
                    d = asdict(r)
                    d["repair"] = rep
                    rows.append(d)
    t4 = summarise(pd.DataFrame(rows), args.eps, args.delta, ["repair", "scenario"])
    t4.to_csv(RESULTS / "theorem_repair.csv", index=False)
    print(t4.groupby("repair")[["cov_eps", "measured_regret", "bound_total"]]
          .mean().to_string())

    print(f"\nWrote theorem_*.csv to {RESULTS}")


if __name__ == "__main__":
    main()
