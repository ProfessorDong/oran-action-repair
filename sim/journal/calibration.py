"""
Journal edition -- effect of perception-aware twin calibration on
the constants of Theorem 1.

Compares two twins that differ only in how they consume the interference
classifier's output:

  label_as_truth   -- the classifier label is treated as the true class.  This
                      is the naive behavior, and it charges the
                      classifier's confusion to the twin-error constant xi.
  perception_aware -- the link model is marginalised over P(iota | iota_hat),
                      obtained by Bayes from the validation confusion matrix
                      with the class prior estimated online by EM under the
                      label-shift assumption.

Author: Liang Dong.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from core import TwinParams, SCENARIO_ORDER
from theory import instrumented_run, summarise
from experiments import load_confusion
import conditional as cond

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    args = ap.parse_args()

    conf = load_confusion("radioml")
    models = {sc: cond.get(sc, conf) for sc in SCENARIO_ORDER}

    rows = []
    for aware in (False, True):
        tp = TwinParams(perception_aware=aware)
        for sc in SCENARIO_ORDER:
            for sd in range(args.seeds):
                for r in instrumented_run(sc, sd, cycles=args.cycles,
                                          eps=args.eps, twin_params=tp,
                                          confusion=conf, model=models[sc]):
                    d = asdict(r)
                    d["perception_aware"] = aware
                    rows.append(d)

    df = pd.DataFrame(rows)
    out = summarise(df, args.eps, args.delta, ["perception_aware"])
    out.to_csv(RESULTS / "twin_calibration.csv", index=False)
    print(out[["perception_aware", "xi_delta", "xi_mean", "cov_eps",
               "measured_regret", "bound_total"]].to_string(index=False))

    per_sc = summarise(df, args.eps, args.delta, ["perception_aware", "scenario"])
    per_sc.to_csv(RESULTS / "twin_calibration_scenario.csv", index=False)
    print(f"\nWrote twin_calibration*.csv to {RESULTS}")


if __name__ == "__main__":
    main()
