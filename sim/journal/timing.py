"""
Journal edition -- latency microbenchmark for the synchronous
actuation path.

Measures, on one CPU core, the per-cycle cost of the three components that sit
between a candidate set and an actuated action: predicate evaluation, the
repair operator, and twin scoring.  The planner is deliberately excluded: it
runs on the non-real-time timescale and its output is cached, so it is not on
the synchronous path.

Author: Liang Dong.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from core import (Action, LatentState, Twin, TwinParams, SCENARIOS,
                  PHI_BASE, PHI_AUG, SAFE_FALLBACK,
                  step_environment, observe, admissible)
from shield import repair
from adversary import NaivePlanner
from experiments import load_confusion

RESULTS = Path(__file__).resolve().parent / "results"


def bench(n_cycles: int = 20000, k: int = 5) -> dict:
    conf = load_confusion("radioml")
    rng = np.random.default_rng(0)
    st = LatentState(rng=rng)
    cfg = SCENARIOS["compound"]
    twin = Twin(TwinParams(), seed=0, confusion=conf)
    planner = NaivePlanner(rng, k=k)

    # Pre-generate the workload so that generation is not timed.
    work = []
    a_prev = Action()
    for c in range(n_cycles):
        st.cycle = c % 300
        step_environment(st, cfg)
        o = observe(st, a_prev, cfg, confusion=conf)
        twin.observe_label(o.interference_class)
        work.append((o, planner.propose(o)))

    def timeit(fn):
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) / n_cycles * 1e6      # microseconds

    def predicates_only():
        for o, cands in work:
            for a in cands:
                admissible(o, a, PHI_BASE)

    def shield_repair_path():
        for o, cands in work:
            pool = [a if admissible(o, a, PHI_BASE)
                    else repair(o, a, PHI_BASE)[0] for a in cands]
            pool.append(SAFE_FALLBACK)

    def shield_repair_plus():
        for o, cands in work:
            pool = [a if admissible(o, a, PHI_AUG)
                    else repair(o, a, PHI_AUG)[0] for a in cands]
            pool.append(SAFE_FALLBACK)

    def twin_only():
        for o, cands in work:
            for a in cands:
                twin(o, a)

    out = {
        "n_cycles": n_cycles, "K": k,
        "predicates_us": timeit(predicates_only),
        "shield_repair_us": timeit(shield_repair_path),
        "shield_repair_plus_us": timeit(shield_repair_plus),
        "twin_us": timeit(twin_only),
        "platform": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    out["actuation_path_us"] = out["shield_repair_us"] + out["twin_us"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=20000)
    ap.add_argument("--repro-minutes", type=float, default=None,
                    help="wall-clock minutes for the full study, if known")
    args = ap.parse_args()
    out = bench(args.cycles)
    if args.repro_minutes is not None:
        out["repro_minutes"] = args.repro_minutes
    else:
        out["repro_minutes"] = 0.0
    (RESULTS / "timing.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(f"  {k:24s} {v}")
    print(f"\nWrote {RESULTS/'timing.json'}")


if __name__ == "__main__":
    main()
