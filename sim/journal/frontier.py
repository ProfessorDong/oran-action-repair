"""
Risk--PDR frontier: conservative threshold tuning against predicate expansion.

The paper observes that the constructed thermal hazard is reachable only because
the signed ceiling was written from spectrum policy without reference to the
thermal envelope, and that one step down the power domain removes it.  That
raises the obvious question the augmented set has to answer: if tightening an
existing threshold removes the hazard for free, why pay Phi+ for it?

This sweeps the operator's own ceiling over the power domain under the base set
Phi and places Phi+ on the same axes.  The comparison is policy against policy,
so the Kerckhoffs adversary retargets to whichever set is enforced -- that is
what an attacker facing that policy would do.

Only P_MAX_POLICY moves.  The hazard oracle reads P_MAX_REG, the true regulatory
cap, which is a property of the environment and does not follow the operator's
threshold; without that separation h_pwr would fall to zero by construction
whenever the policy tightened, and the frontier would be circular.

Author: Liang Dong.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"

# Ceilings drawn from the action space's own power domain.  30 dBm is deployed.
CEILINGS = [23.0, 26.0, 29.0, 30.0]


def _run(ceiling: float, pset: str, seeds: int, cycles: int) -> pd.DataFrame:
    os.environ["SHIELD_P_MAX_POLICY"] = repr(ceiling)
    import core
    importlib.reload(core)
    import shield, adversary, conditional, experiments
    for m in (shield, adversary, conditional, experiments):
        importlib.reload(m)
    assert abs(core.P_MAX_POLICY - ceiling) < 1e-12, core.P_MAX_POLICY
    assert abs(core.P_MAX_REG - 30.0) < 1e-12, "the true cap must not move"

    frames = []
    # Repair under the adaptive adversary: the setting in which Phi+ is claimed.
    frames.append(experiments.sweep(
        list(range(seeds)), experiments.SCENARIO_ORDER, ["shield_repair"],
        ["adaptive"], [pset], ["radioml"], cycles, verbose=False))
    # Structured search ignores the planner, so it runs in the main setting.
    frames.append(experiments.sweep(
        list(range(seeds)), experiments.SCENARIO_ORDER, ["structured"],
        ["naive"], [pset], ["radioml"], cycles, verbose=False))
    out = pd.concat(frames, ignore_index=True)
    out["ceiling_dbm"] = ceiling
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--out", default=str(RESULTS / "frontier.csv"))
    args = ap.parse_args()

    frames = []
    for c in CEILINGS:
        print(f"[Phi, ceiling {c:.0f} dBm] ...", flush=True)
        frames.append(_run(c, "phi", args.seeds, args.cycles))
    # Phi+ is evaluated at the deployed ceiling; it buys its reduction from
    # telemetry, not from a tighter threshold.
    print("[Phi+, ceiling 30 dBm] ...", flush=True)
    frames.append(_run(30.0, "phi_plus", args.seeds, args.cycles))

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out, index=False)

    piv = (df.groupby(["controller", "predicate_set", "ceiling_dbm"])
             [["mean_pdr", "hazard_rate", "violation_rate_phi",
               "h_therm", "h_pwr", "h_relay"]].mean().round(4))
    print("\n=== risk--PDR frontier ===")
    print(piv.to_string())
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
