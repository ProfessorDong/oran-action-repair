"""
Journal edition -- statistics.

Bootstrap confidence intervals, paired Wilcoxon signed-rank tests with
Holm-Bonferroni correction across the family of comparisons, and Cliff's delta
effect sizes.  Every reported comparison carries a direction, an effect size
and an adjusted p-value; a bare p-value without a direction, as in the
earlier draft, does not say who won.

Author: Liang Dong.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def bootstrap_ci(x, B: int = 10000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float(x.mean()), float(x.mean()), float(x.mean())
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, x.size, size=(B, x.size))].mean(axis=1)
    return (float(x.mean()), float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)))


def cliffs_delta(a, b) -> float:
    """Non-parametric effect size in [-1, 1]; positive means a > b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (a.size * b.size))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_tests(df: pd.DataFrame, ref: str, metric: str,
                 controllers: list[str]) -> pd.DataFrame:
    """Paired comparison of `ref` against every other controller, pairing on
    (scenario, seed)."""
    piv = df.pivot_table(index=["scenario", "seed"], columns="controller",
                         values=metric)
    rows, pvals = [], []
    for ct in controllers:
        if ct == ref or ct not in piv.columns or ref not in piv.columns:
            continue
        pair = piv[[ref, ct]].dropna()
        a, b = pair[ref].to_numpy(), pair[ct].to_numpy()
        d = a - b
        if np.allclose(d, 0):
            p, stat = 1.0, 0.0
        else:
            res = sps.wilcoxon(a, b, zero_method="zsplit")
            p, stat = float(res.pvalue), float(res.statistic)
        rows.append(dict(metric=metric, reference=ref, other=ct, n_pairs=len(d),
                         mean_ref=float(a.mean()), mean_other=float(b.mean()),
                         mean_diff=float(d.mean()),
                         direction=("ref_higher" if d.mean() > 0 else
                                    "other_higher" if d.mean() < 0 else "tie"),
                         cliffs_delta=cliffs_delta(a, b),
                         wilcoxon_W=stat, p_raw=p))
        pvals.append(p)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(pvals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(RESULTS / "master.csv"))
    ap.add_argument("--out", default=str(RESULTS / "stats.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.master)
    main_df = df[(df.planner == "naive") & (df.predicate_set == "phi") &
                 (df.perception == "radioml")]
    adapt_df = df[(df.planner == "adaptive") & (df.perception == "radioml")]

    # ---- per-cell CIs ----------------------------------------------------
    summary = {}
    for (sc, ct), sub in main_df.groupby(["scenario", "controller"]):
        # Stable seeds: Python's string hash is salted per process, so hash()
        # would make the published intervals irreproducible.
        sd = lambda tag: zlib.crc32(f"{sc}|{ct}|{tag}".encode())
        pdr = bootstrap_ci(sub.mean_pdr, seed=sd("pdr"))
        vio = bootstrap_ci(sub.violation_rate_phi, seed=sd("viol"))
        haz = bootstrap_ci(sub.hazard_rate, seed=sd("haz"))
        summary.setdefault(sc, {})[ct] = dict(
            n=int(len(sub)), pdr=pdr, violation=vio, hazard=haz,
            recovery=float(sub.recovery_cycles.mean()))

    controllers = sorted(main_df.controller.unique())
    tests = {}
    for metric in ("mean_pdr", "hazard_rate"):
        t = paired_tests(main_df, "shield_repair", metric, controllers)
        tests[metric] = t.to_dict(orient="records")
        print(f"\n=== paired tests, the shielded controller (repair) vs baselines, {metric} "
              f"(overt adversary, Phi, RadioML) ===")
        print(t[["other", "mean_ref", "mean_other", "mean_diff", "direction",
                 "cliffs_delta", "p_holm"]].round(4).to_string(index=False))

    # ---- predicate-coverage gap under the adaptive adversary -------------
    gap = (adapt_df.groupby(["predicate_set", "controller"])
           [["mean_pdr", "violation_rate_phi", "hazard_rate"]].mean())
    print("\n=== adaptive adversary: Phi vs Phi+ ===")
    print(gap.round(4).to_string())

    out = dict(summary=summary, tests=tests,
               coverage_gap=json.loads(gap.reset_index().to_json(orient="records")))
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
