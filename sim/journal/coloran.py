"""
Journal edition -- link-model calibration against measured
Colosseum/ColO-RAN traces.

The evaluation's weakest point was that the mapping from SINR and modulation
and coding scheme to delivered packets was a logistic curve we chose.  This
module replaces the guess with a fit to real measurements.

Source: the public ColO-RAN dataset (Polese et al., IEEE TMC 2023), 7 base
stations, 42 static UEs, 3 slices, 3 scheduling policies and 28 RBG allocations
on the Colosseum "Rome static medium" RF scenario, logged by srsRAN.  Each
per-UE CSV records, per 250 ms report, the downlink SNR (dB), the scheduled
MCS index, and the downlink block error rate (percent).

What can and cannot be identified
---------------------------------
These traces are logged under *closed-loop link adaptation*: srsRAN's outer
loop steers the MCS to hold block error near a target, so measured BLER sits
around 3 % across the entire observed SNR range and PDR is 0.96-1.00 almost
everywhere.  A demodulation curve -- how delivery degrades as SINR falls below
an MCS threshold -- is therefore **not identifiable** from this dataset, and we
say so in the paper rather than fitting a slope to noise.

Three things are identifiable, and we calibrate all three:

  1. Per-MCS SNR operating thresholds.  The lowest SNR at which the scheduler
     is willing to select MCS m is an empirical demodulation threshold; we take
     a low weighted quantile of the SNR distribution conditioned on m and fit a
     line in the MCS index, the shape the 3GPP tables have by construction.
  2. The delivered-fraction ceiling.  A link operating inside its adaptation
     envelope delivers about 0.97, not 1.0; our model previously saturated at
     one.
  3. The SNR operating range and its centre, which fix the environment's
     nominal link budget.

Author: Liang Dong.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATASET = HERE.parent.parent / "colosseum-oran-coloran-dataset-master" / "rome_static_medium"

COLS = ["dl_snr", "dl_mcs", "dl_bler", "dl_brate", "is_attached"]
MIN_CELL = 40          # reports needed before a (SNR, MCS) cell is used


def sample_files(n: int, seed: int = 0) -> list[Path]:
    """Stratified sample: walk every scheduler/allocation/experiment/BS
    directory and take a bounded number of UE traces from each, so the sample
    spans the whole design rather than one corner of it."""
    rng = random.Random(seed)
    groups: list[list[Path]] = []
    for sched in sorted(DATASET.glob("sched*")):
        for tr in sorted(sched.glob("tr*")):
            for exp in sorted(tr.glob("exp*")):
                for bs in sorted(exp.glob("bs*")):
                    ues = sorted(bs.glob("ue*.csv"))
                    if ues:
                        groups.append(ues)
    rng.shuffle(groups)
    out: list[Path] = []
    per = max(1, n // max(len(groups), 1))
    for g in groups:
        out.extend(rng.sample(g, min(per, len(g))))
        if len(out) >= n:
            break
    return out[:n]


def load(files: list[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=COLS)
        except Exception:
            continue
        d = d[(d.is_attached > 0) & (d.dl_mcs > 0)]
        if len(d):
            frames.append(d[["dl_snr", "dl_mcs", "dl_bler", "dl_brate"]])
    if not frames:
        raise RuntimeError("no usable rows; check the dataset path")
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.dl_bler >= 0) & (d.dl_bler <= 100)]
    d["pdr"] = 1.0 - d.dl_bler / 100.0
    d["mcs"] = d.dl_mcs.round().astype(int)
    d["snr"] = d.dl_snr.round().astype(int)
    return d


def fit(d: pd.DataFrame, q: float = 0.05) -> dict:
    """Calibrate the three identifiable link-model quantities."""
    cells = (d.groupby(["mcs", "snr"])
               .agg(pdr=("pdr", "mean"), n=("pdr", "size"))
               .reset_index())
    cells = cells[cells.n >= MIN_CELL]

    # (1) Per-MCS operating threshold: low weighted quantile of SNR | MCS.
    theta = {}
    support = {}
    for m, g in cells.groupby("mcs"):
        g = g.sort_values("snr")
        w = g.n.to_numpy(dtype=float)
        cw = np.cumsum(w) / w.sum()
        theta[int(m)] = float(np.interp(q, cw, g.snr.to_numpy(dtype=float)))
        support[int(m)] = int(w.sum())

    # Fit a line over the well-sampled MCS indices only.
    well = [m for m in theta if support[m] >= 20000]
    ms = np.array(sorted(well), dtype=float)
    ts = np.array([theta[int(m)] for m in ms])
    wts = np.array([support[int(m)] for m in ms], dtype=float)
    slope, icept = np.polyfit(ms, ts, 1, w=np.sqrt(wts))
    resid = ts - (slope * ms + icept)

    # (2) Delivered-fraction ceiling: weighted mean PDR where the link is
    # operating inside its adaptation envelope (SNR at or above threshold).
    inside = cells[cells.apply(
        lambda r: r.snr >= theta.get(int(r.mcs), -99), axis=1)]
    ceiling = float(np.average(inside.pdr, weights=inside.n))

    # (3) Operating range of the realised SNR.
    snr_w = d.groupby("snr").size()
    vals = snr_w.index.to_numpy(dtype=float); wt = snr_w.to_numpy(dtype=float)
    cw = np.cumsum(wt) / wt.sum()
    q05, q50, q95 = (float(np.interp(x, cw, vals)) for x in (0.05, 0.5, 0.95))

    return dict(
        theta=theta, theta_support=support,
        theta_slope=float(slope), theta_intercept=float(icept),
        theta_resid_rms=float(np.sqrt(np.mean(resid ** 2))),
        well_sampled_mcs=[int(m) for m in well],
        pdr_ceiling=ceiling,
        snr_q05=q05, snr_median=q50, snr_q95=q95,
        bler_mean_pct=float(d.dl_bler.mean()),
        frac_cells_below_0p9=float((cells.pdr < 0.9).mean()),
        slope_identifiable=False,
        n_reports=int(len(d)), n_cells=int(len(cells)),
        mcs_range=[int(d.mcs.min()), int(d.mcs.max())],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "coloran_fit.json"))
    args = ap.parse_args()

    if not DATASET.exists():
        raise SystemExit(f"dataset not found at {DATASET}")
    files = sample_files(args.files, args.seed)
    print(f"sampling {len(files)} UE traces ...", flush=True)
    d = load(files)
    print(f"  {len(d):,} attached reports; "
          f"SNR {d.snr.min()}..{d.snr.max()} dB, MCS {d.mcs.min()}..{d.mcs.max()}, "
          f"mean BLER {d.dl_bler.mean():.2f}%")
    f = fit(d)
    f["n_files"] = len(files)
    Path(args.out).write_text(json.dumps(f, indent=2))
    print(f"\n  MCS threshold line: {f['theta_slope']:.3f} dB/index + "
          f"{f['theta_intercept']:.2f} dB  (RMS resid {f['theta_resid_rms']:.2f} dB,"
          f" {len(f['well_sampled_mcs'])} well-sampled MCS)")
    print(f"  delivered-fraction ceiling: {f['pdr_ceiling']:.4f}")
    print(f"  SNR operating range: q05 {f['snr_q05']:.1f}, median "
          f"{f['snr_median']:.1f}, q95 {f['snr_q95']:.1f} dB")
    print(f"  mean BLER {f['bler_mean_pct']:.2f}% -- demodulation slope NOT "
          f"identifiable under closed-loop link adaptation")
    print(f"\nWrote {args.out}")
    cells = (d.groupby(["mcs", "snr"])
               .agg(pdr=("pdr", "mean"), n=("pdr", "size"))
               .reset_index())
    cells[cells.n >= MIN_CELL].to_csv(RESULTS / "coloran_cells.csv", index=False)


if __name__ == "__main__":
    main()
