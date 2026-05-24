"""
Main experiment driver for SHIELD-RIC.

Runs three experimental conditions:
    perfect    -- oracle perception of interference class
    radioml    -- perception derived from RadioML 2016.10a-trained classifier
    synthetic  -- perception derived from synthetic RF + same architecture

For each condition we sweep five scenarios × five controllers × N seeds, and
write a tidy CSV to results/master.csv.  The figures generator consumes that.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import shield_ric as sr
from shield_ric import SCENARIOS, CONTROLLERS

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
RESULTS = SIM / "results"


def load_perception(json_path: Path) -> dict | None:
    if not json_path.exists():
        return None
    with open(json_path) as f:
        d = json.load(f)
    return d.get("perception_confusion")


def run_condition(name: str, confusion: dict | None, seeds, scenarios, controllers,
                  cycles: int, out_csv: Path, append: bool = False) -> None:
    sr.PERCEPTION_CONFUSION = confusion
    mode = "a" if append else "w"
    write_header = not append
    with open(out_csv, mode, newline="") as f:
        w = None
        for sc in scenarios:
            for ct in controllers:
                for s in seeds:
                    r = sr.run(sc, ct, seed=s, cycles=cycles, results_dir=RESULTS)
                    row = {"condition": name, **vars(r)}
                    if w is None:
                        w = csv.DictWriter(f, fieldnames=list(row.keys()))
                        if write_header:
                            w.writeheader()
                    w.writerow(row)
                    print(f"[{name:>10s}] {sc:>18s}  {ct:>10s}  s={s}  "
                          f"PDR={r.mean_pdr:.3f}  viol={r.violation_rate:.3f}  "
                          f"rec={r.recovery_cycles}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default=str(RESULTS / "master.csv"))
    ap.add_argument("--conditions", nargs="+",
                    default=["perfect", "radioml", "synthetic"])
    args = ap.parse_args()

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    scenarios = tuple(SCENARIOS.keys())
    controllers = tuple(CONTROLLERS.keys())

    perception_files = {
        "radioml":   RESULTS / "perception_radioml.json",
        "synthetic": RESULTS / "perception_synth.json",
    }

    first = True
    for cond in args.conditions:
        confusion = None
        if cond in perception_files:
            confusion = load_perception(perception_files[cond])
            if confusion is None:
                print(f"[skip] perception json for {cond} not found")
                continue
        run_condition(
            cond, confusion, args.seeds, scenarios, controllers,
            cycles=args.cycles, out_csv=out_csv, append=not first,
        )
        first = False

    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
