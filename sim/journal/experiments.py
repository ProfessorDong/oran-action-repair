"""
Journal edition -- experiment driver.

One episode = one (scenario, controller, planner, predicate-set, seed) tuple.
Every cycle logs, separately:

  * violation_phi   -- did the ACTUATED action violate the DEPLOYED predicate
                       set?  For any sound shield this is identically zero and
                       serves as a machine-checkable witness for Lemma 1.
  * hazard          -- did the actuated action trigger the LATENT hazard
                       oracle H?  This is the quantity that actually matters
                       and that no shield can drive to zero unless Phi covers
                       H.  The gap between the two columns is the paper's
                       predicate-coverage gap.

Author: Liang Dong.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from core import (
    Action, LatentState, Observation, Twin, TwinParams,
    SCENARIOS, SCENARIO_ORDER, SAFE_FALLBACK, HAZARD_NAMES,
    PREDICATE_SETS, PHI_BASE,
    step_environment, observe, realised_pdr, update_thermal,
    update_priority_ewma, hazards, admissible, reward,
)
from shield import CONTROLLERS, NEEDS_PLANNER, LagrangianRLController
from adversary import PLANNERS

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True, parents=True)

CONF_PATH = RESULTS / "perception_radioml.json"   # published with the code
_LEGACY_CONF = HERE.parent / "results" / "perception_radioml.json"


def load_confusion(kind: str) -> dict | None:
    """kind in {perfect, radioml}."""
    if kind == "perfect":
        return None
    p = CONF_PATH if kind == "radioml" else RESULTS / f"perception_{kind}.json"
    if not p.exists() and kind == "radioml" and _LEGACY_CONF.exists():
        p = _LEGACY_CONF
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())["perception_confusion"]


# ---------------------------------------------------------------------------

@dataclass
class EpisodeSummary:
    scenario: str
    controller: str
    planner: str
    predicate_set: str
    perception: str
    seed: int
    cycles: int
    mean_pdr: float
    p05_pdr: float
    violation_rate_phi: float      # w.r.t. the DEPLOYED predicate set
    hazard_rate: float             # w.r.t. the LATENT oracle H
    recovery_cycles: int
    fallback_rate: float
    repair_rate: float
    mean_power_dbm: float
    max_pa_temp_c: float
    mean_reward: float
    # per-hazard breakdown
    h_rf: float = 0.0
    h_pwr: float = 0.0
    h_relay: float = 0.0
    h_part: float = 0.0
    h_starve: float = 0.0
    h_therm: float = 0.0


def run_episode(scenario: str, controller: str, seed: int,
                planner_kind: str = "naive", predicate_set: str = "phi",
                eval_predicate_set: str | None = None,
                perception: str = "radioml", cycles: int = 300, k: int = 5,
                twin_params: TwinParams | None = None,
                confusion: dict | None = "auto",
                log_rows: list | None = None) -> EpisodeSummary:
    rng = np.random.default_rng(seed)
    st = LatentState(rng=rng)
    cfg = SCENARIOS[scenario]
    phi = PREDICATE_SETS[predicate_set]
    # The shield ENFORCES `phi`; violations are SCORED against `phi_eval`.
    # They differ only in the leave-one-out ablation, where the point is to
    # count the violations that the removed predicate would have caught.
    phi_eval = PREDICATE_SETS[eval_predicate_set or predicate_set]
    if confusion == "auto":
        confusion = load_confusion(perception)

    twin = Twin(twin_params or TwinParams(), rng=np.random.default_rng(seed + 9973),
                seed=seed, confusion=confusion)
    if planner_kind == "adaptive":
        # A Kerckhoffs adversary knows the deployed bundle.  Handing it
        # PHI_BASE while evaluating Phi+ would measure only how well Phi+
        # blocks an attack designed against Phi.
        planner = PLANNERS[planner_kind](rng, k=k, phi=phi)
    else:
        planner = PLANNERS[planner_kind](rng, k=k)

    ctrl = CONTROLLERS[controller](planner=planner, phi=phi)
    ctrl.reset(rng)

    a_prev = Action()
    pdrs, rewards, powers = [], [], []
    n_viol = n_haz = n_fb = n_rep = 0
    hcount = {h: 0 for h in HAZARD_NAMES}
    recovery = -1
    onset = cfg["onset"]

    for c in range(cycles):
        st.cycle = c
        step_environment(st, cfg)
        o = observe(st, a_prev, cfg, confusion=confusion)
        twin.observe_label(o.interference_class)

        d = ctrl.step(o, twin)
        a = d.action

        # --- ground truth -------------------------------------------------
        viol = not admissible(o, a, phi_eval)
        hz = hazards(st, a)
        pdr = realised_pdr(st, a)
        update_thermal(st, a)
        update_priority_ewma(st, pdr, a)
        # hazards that depend on post-update latent state
        hz = sorted(set(hz) | set(hazards(st, a)))

        if viol:
            n_viol += 1
        if hz:
            n_haz += 1
            for h in hz:
                hcount[h] += 1
        if d.mode == "fallback":
            n_fb += 1
        n_rep += d.n_repaired

        r = reward(pdr, a, bool(hz))
        pdrs.append(pdr)
        rewards.append(r)
        powers.append(a.power_dbm)

        if isinstance(ctrl, LagrangianRLController):
            ctrl.learn(pdr, float(viol))
        ctrl.post(o, pdr)

        if c > onset and recovery < 0 and pdr > 0.9:
            recovery = c - onset

        if log_rows is not None:
            log_rows.append(dict(
                scenario=scenario, controller=controller, planner=planner_kind,
                predicate_set=predicate_set, seed=seed, cycle=c,
                pdr=pdr, violation_phi=int(viol), hazard=int(bool(hz)),
                hazards=",".join(hz), mode=d.mode, power_dbm=a.power_dbm,
                pa_temp_c=st.pa_temp_c, incumbent=int(st.incumbent_active),
                interference=st.interference_class,
                perceived=o.interference_class,
            ))
        a_prev = a

    pdr_arr = np.asarray(pdrs)
    return EpisodeSummary(
        scenario=scenario, controller=controller, planner=planner_kind,
        predicate_set=predicate_set, perception=perception, seed=seed,
        cycles=cycles,
        mean_pdr=float(pdr_arr.mean()),
        p05_pdr=float(np.percentile(pdr_arr, 5)),
        violation_rate_phi=n_viol / cycles,
        hazard_rate=n_haz / cycles,
        recovery_cycles=recovery if recovery >= 0 else cycles,
        fallback_rate=n_fb / cycles,
        repair_rate=n_rep / (cycles * k),
        mean_power_dbm=float(np.mean(powers)),
        max_pa_temp_c=float(st.pa_temp_c),
        mean_reward=float(np.mean(rewards)),
        **{h: hcount[h] / cycles for h in HAZARD_NAMES},
    )


# ---------------------------------------------------------------------------

MAIN_CONTROLLERS = ["static", "heuristic", "structured", "greedy_twin",
                    "llm_only", "lagrangian_rl", "simplex_rta",
                    "shield_filter", "shield_filter_fb", "shield_repair"]


def _one(job):
    """Worker: one episode.  Module-level so it is picklable."""
    perc, pl, ps, sc, ct, sd, cycles, tp, tag = job
    r = run_episode(sc, ct, sd, planner_kind=pl, predicate_set=ps,
                    perception=perc, cycles=cycles, twin_params=tp,
                    confusion=load_confusion(perc))
    d = asdict(r)
    if tag is not None:
        d["twin"] = tag
    return d


def sweep(seeds, scenarios, controllers, planners, predicate_sets,
          perceptions, cycles, verbose=True,
          twin_params: TwinParams | None = None,
          tag: str | None = None, workers: int | None = None) -> pd.DataFrame:
    """Run the cross product of conditions, one episode per task.

    Episodes are independent, so the sweep is embarrassingly parallel; we fan
    out over processes because the inner loop is pure Python and the GIL would
    otherwise serialise it.
    """
    combos = [(perc, pl, ps, sc, ct, sd, cycles, twin_params, tag)
              for perc, pl, ps, sc, ct, sd in
              itertools.product(perceptions, planners, predicate_sets,
                                scenarios, controllers, seeds)]
    n = workers if workers is not None else max(
        1, min(len(combos), (os.cpu_count() or 2) - 2))
    if n <= 1:
        rows = [_one(j) for j in combos]
    else:
        with ProcessPoolExecutor(max_workers=n) as ex:
            rows = list(ex.map(_one, combos, chunksize=4))
    if verbose:
        print(f"  {len(rows)} episodes on {n} workers", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--out", default=str(RESULTS / "master.csv"))
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    frames = []

    # (E1) Main comparison: overt adversary, deployed Phi, both perceptions.
    print("[E1] main controller comparison ...", flush=True)
    frames.append(sweep(seeds, SCENARIO_ORDER, MAIN_CONTROLLERS, ["naive"],
                        ["phi"], ["radioml", "perfect"], args.cycles))

    # (E2) Adaptive adversary vs Phi and vs Phi+  (predicate-coverage gap).
    print("[E2] adaptive adversary, Phi vs Phi+ ...", flush=True)
    frames.append(sweep(seeds, SCENARIO_ORDER,
                        ["heuristic", "greedy_twin", "llm_only",
                         "lagrangian_rl", "simplex_rta",
                         "shield_filter", "shield_repair"],
                        ["adaptive"], ["phi", "phi_plus"], ["radioml"],
                        args.cycles))

    # (E3) Overt adversary against Phi+ (does augmentation cost throughput?).
    print("[E3] overt adversary against Phi+ ...", flush=True)
    frames.append(sweep(seeds, SCENARIO_ORDER, ["shield_repair", "shield_filter"],
                        ["naive"], ["phi_plus"], ["radioml"], args.cycles))

    # (E4) Benign planner: the price of the shield when nobody is attacking.
    print("[E4] benign planner ...", flush=True)
    frames.append(sweep(seeds, SCENARIO_ORDER,
                        ["greedy_twin", "shield_filter", "shield_repair"],
                        ["benign"], ["phi"], ["radioml"], args.cycles))

    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["scenario", "controller", "planner", "predicate_set",
                "perception", "seed"])
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(df)} episodes)")

    # (E5) Twin-calibration ablation: does perception-aware twin calibration
    # buy throughput as well as a tighter bound?
    print("\n[E5] twin-calibration ablation ...", flush=True)
    abl = []
    for aware, tag in ((True, "perception_aware"), (False, "label_as_truth")):
        abl.append(sweep(seeds, SCENARIO_ORDER,
                         ["greedy_twin", "shield_filter", "shield_repair"],
                         ["naive"], ["phi"], ["radioml"], args.cycles,
                         twin_params=TwinParams(perception_aware=aware),
                         tag=tag))
    dfa = pd.concat(abl, ignore_index=True)
    dfa.to_csv(RESULTS / "twin_ablation.csv", index=False)
    print(dfa.groupby(["twin", "controller"])[["mean_pdr", "hazard_rate"]]
          .mean().to_string())

    # (E6) Predicate leave-one-out ablation on the deployed set.
    print("\n[E6] predicate leave-one-out ...", flush=True)
    import core as _core
    rows = []
    base = list(_core.PHI_BASE)
    for drop in [None] + [p.name for p in base]:
        kept = tuple(p for p in base if p.name != drop)
        _core.PREDICATE_SETS["loo"] = kept
        for sc in SCENARIO_ORDER:
            for sd in seeds[:15]:
                r = run_episode(sc, "shield_repair", sd, planner_kind="naive",
                                predicate_set="loo", eval_predicate_set="phi",
                                perception="radioml", cycles=args.cycles)
                d = asdict(r)
                d["dropped"] = drop or "none"
                rows.append(d)
    dfl = pd.DataFrame(rows)
    dfl.to_csv(RESULTS / "predicate_loo.csv", index=False)
    print(dfl.groupby("dropped")[["violation_rate_phi", "hazard_rate",
                                  "mean_pdr"]].mean().to_string())


if __name__ == "__main__":
    main()
