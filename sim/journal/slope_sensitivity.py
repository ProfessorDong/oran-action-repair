"""
Journal edition -- sensitivity to the one link-model parameter the
ColO-RAN traces cannot identify.

Calibration against the measured Colosseum traces (coloran.py) fixes the
per-MCS demodulation thresholds, the delivered-fraction ceiling and the SNR
operating range.  It cannot fix the demodulation *slope*: srsRAN's closed-loop
link adaptation holds block error near target across the whole observed SNR
range, so the traces contain almost no information about how delivery decays
once SINR falls below an MCS threshold.

Rather than pick a slope and hope, we sweep it over an order of magnitude and
check that the paper's conclusions do not depend on it.  The claims under test
are the three the paper actually makes:

    C1  shielded controllers emit zero policy violations;
    C2  repair delivers at least as much throughput as filtering;
    C3  the predicate-coverage gap is present under the adaptive adversary.

Author: Liang Dong.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"

SLOPES = [0.5, 0.8, 1.4, 2.2, 3.5]
CONTROLLERS = ["heuristic", "greedy_twin", "shield_filter", "shield_repair"]


def run_for_slope(slope: float, seeds: int, cycles: int) -> pd.DataFrame:
    """Sweep with the demodulation slope set to `slope`.

    The slope is passed through the environment, not by patching a module
    attribute, because the sweep fans out over processes that import `core`
    fresh and would otherwise silently use the default.
    """
    os.environ["SHIELD_BLER_SLOPE"] = repr(slope)
    import core
    importlib.reload(core)
    import shield, adversary, experiments
    for m in (shield, adversary, experiments):
        importlib.reload(m)
    assert abs(core.BLER_SLOPE - slope) < 1e-12, core.BLER_SLOPE

    frames = []
    for planner, psets in (("naive", ["phi"]), ("adaptive", ["phi", "phi_plus"])):
        d = experiments.sweep(list(range(seeds)), experiments.SCENARIO_ORDER,
                              CONTROLLERS, [planner], psets, ["radioml"],
                              cycles, verbose=False)
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["bler_slope"] = slope
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--out", default=str(RESULTS / "slope_sensitivity.csv"))
    args = ap.parse_args()

    frames = []
    for s in SLOPES:
        print(f"[slope {s}] ...", flush=True)
        frames.append(run_for_slope(s, args.seeds, args.cycles))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out, index=False)

    main_df = df[(df.planner == "naive") & (df.predicate_set == "phi")]
    adapt = df[(df.planner == "adaptive")]

    print("\n=== C1: shielded policy-violation rate (must be 0 everywhere) ===")
    c1 = (main_df[main_df.controller.str.startswith("shield")]
          .groupby(["bler_slope", "controller"]).violation_rate_phi.max())
    print(c1.to_string())

    print("\n=== C2: mean PDR, repair vs filter ===")
    piv = main_df.pivot_table(index="bler_slope", columns="controller",
                              values="mean_pdr")
    piv["repair-filter"] = piv.shield_repair - piv.shield_filter
    print(piv.round(4).to_string())

    print("\n=== C3: coverage gap under the adaptive adversary ===")
    c3 = (adapt[adapt.controller == "shield_repair"]
          .groupby(["bler_slope", "predicate_set"])
          [["violation_rate_phi", "hazard_rate"]].mean())
    print(c3.round(4).to_string())

    ok1 = float(c1.max()) == 0.0
    ok2 = bool((piv["repair-filter"] >= -1e-9).all())
    ok3 = bool((c3.xs("phi", level="predicate_set").hazard_rate >
                c3.xs("phi_plus", level="predicate_set").hazard_rate).all())
    print(f"\nC1 zero violations everywhere : {'OK' if ok1 else 'FAIL'}")
    print(f"C2 repair >= filter everywhere: {'OK' if ok2 else 'FAIL'}")
    print(f"C3 coverage gap everywhere    : {'OK' if ok3 else 'FAIL'}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
