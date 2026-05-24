"""
Bootstrap confidence intervals and Wilcoxon test for the headline numbers.

Reads results/master.csv and emits a summary table per (scenario, controller)
with 95% CI on mean PDR and on the safety-violation rate.  Also runs a
Wilcoxon signed-rank test SHIELD-RIC vs. heuristic on per-scenario PDR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
SIM  = HERE.parent
RESULTS = SIM / "results"


def bootstrap_ci(x: np.ndarray, B: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Return (mean, low, high) bootstrap CI."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float(x.mean()), float(x.mean()), float(x.mean())
    idx = rng.integers(0, len(x), size=(B, len(x)))
    boots = x[idx].mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(x.mean()), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(RESULTS / "master.csv"))
    ap.add_argument("--condition", default="radioml")
    ap.add_argument("--out", default=str(RESULTS / "stats.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.master)
    df = df[df["condition"] == args.condition]

    summary = {}
    for (sc, ct), sub in df.groupby(["scenario", "controller"]):
        pdr = sub["mean_pdr"].to_numpy()
        viol = sub["violation_rate"].to_numpy()
        pdr_mu, pdr_lo, pdr_hi = bootstrap_ci(pdr, seed=hash((sc, ct, "pdr")) & 0xffff)
        v_mu, v_lo, v_hi = bootstrap_ci(viol, seed=hash((sc, ct, "v")) & 0xffff)
        summary.setdefault(sc, {})[ct] = {
            "pdr_mean": pdr_mu, "pdr_lo": pdr_lo, "pdr_hi": pdr_hi,
            "viol_mean": v_mu, "viol_lo": v_lo, "viol_hi": v_hi,
            "n_seeds": int(len(pdr)),
        }

    # Wilcoxon SHIELD vs Heuristic on per-(scenario, seed) PDR pairs
    paired = df.pivot_table(index=["scenario", "seed"],
                            columns="controller",
                            values="mean_pdr")
    sh = paired.get("shield_ric")
    he = paired.get("heuristic")
    wilcoxon = None
    if sh is not None and he is not None:
        # Replace identical pairs (zero diff) -- scipy handles ties
        try:
            res = sps.wilcoxon(sh.values, he.values, zero_method="zsplit")
            wilcoxon = {"statistic": float(res.statistic), "pvalue": float(res.pvalue),
                        "n_pairs": int(len(sh))}
        except Exception as e:
            wilcoxon = {"error": str(e)}

    out = {"condition": args.condition, "summary": summary, "wilcoxon": wilcoxon}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    # Pretty-print headline rows
    sc_order = ["narrowband", "wideband", "bursty", "backhaul",
                "compromised_xapp", "compound"]
    ct_order = ["static", "heuristic", "rl", "llm_only", "shield_ric"]
    print(f"\n95% bootstrap CIs (condition={args.condition}):")
    hdr = f"{'scenario':>18s}  {'controller':>10s}  {'pdr_mean':>9s}  {'95% CI':>16s}  {'viol':>9s}  {'95% CI':>16s}"
    print(hdr)
    for sc in sc_order:
        for ct in ct_order:
            r = summary.get(sc, {}).get(ct)
            if not r: continue
            print(f"{sc:>18s}  {ct:>10s}  "
                  f"{r['pdr_mean']:9.3f}  [{r['pdr_lo']:.3f},{r['pdr_hi']:.3f}]  "
                  f"{r['viol_mean']:9.3f}  [{r['viol_lo']:.3f},{r['viol_hi']:.3f}]")
    if wilcoxon and "pvalue" in wilcoxon:
        print(f"\nWilcoxon (SHIELD-RIC vs heuristic): "
              f"W={wilcoxon['statistic']:.2f}, p={wilcoxon['pvalue']:.4f}, "
              f"n={wilcoxon['n_pairs']}")


if __name__ == "__main__":
    main()
