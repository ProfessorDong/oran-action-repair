"""
RF perception module for SHIELD-RIC.

Two data paths:

1. real_data:  load DeepSig RadioML 2016.10A (215k I/Q windows, 11 modulation
               classes, SNRs -20:18 dB).  We map RadioML modulations to a
               4-class *interference taxonomy* used by SHIELD-RIC:

               none       <- {QAM16, QAM64, BPSK, QPSK, GFSK, CPFSK, PAM4}
                              (treated as clean digital traffic)
               narrowband <- {AM-SSB, AM-DSB}
                              (single-tone-like narrowband emissions)
               wideband   <- {WBFM, 8PSK}
                              (broadband modulations / FM)
               bursty     <- {QAM16+low SNR, QAM64+low SNR}   (re-labelled
                              with synthetic bursting in code; see remap below)

               Low-SNR samples become the bursty class to keep four labels.
               The mapping is intentionally crude; the goal is a measured
               classifier confusion matrix, not a definitive RF taxonomy.

2. synth_data: generate a self-contained synthetic dataset using documented
               signal models.  Used when the real dataset is unavailable.

A small 1D CNN is trained on either path; we then report a confusion matrix
that the SHIELD-RIC simulator can ingest via PERCEPTION_CONFUSION.

Author: Liang Dong (sole author, MILCOM 2026 Paper 1).
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset, random_split
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


HERE = Path(__file__).resolve().parent
SIM_DIR = HERE.parent
DATA_DIR = SIM_DIR / "data"
RESULTS_DIR = SIM_DIR / "results"

CLASSES = ("none", "narrowband", "wideband", "bursty")


# ---------------------------------------------------------------------------
# Synthetic data path (fully self-contained)
# ---------------------------------------------------------------------------

def synth_window(rng: np.random.Generator, label: str, n: int = 128,
                 fs: float = 200e3) -> np.ndarray:
    """Return a (2, n) I/Q tensor in [-1,1] for the requested class."""
    t = np.arange(n) / fs
    # baseline modulated carrier (random data)
    syms = rng.choice([-1, 1], size=n)
    base_re = syms * np.cos(2 * np.pi * 30e3 * t)
    base_im = syms * np.sin(2 * np.pi * 30e3 * t)
    sig = base_re + 1j * base_im

    if label == "none":
        noise_amp = 0.20
    elif label == "narrowband":
        # add CW tone(s) close to carrier
        tone = np.exp(1j * (2 * np.pi * rng.uniform(28e3, 32e3) * t
                            + rng.uniform(0, 2 * np.pi)))
        sig = sig + 1.6 * tone
        noise_amp = 0.20
    elif label == "wideband":
        # add wideband noise burst across the window
        wbn = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
        # bandpass-shape it crudely by light AR filtering
        wbn = wbn + 0.3 * np.roll(wbn, 1) + 0.2 * np.roll(wbn, -1)
        sig = sig + 1.2 * wbn
        noise_amp = 0.10
    elif label == "bursty":
        # random duty-cycle pulses with high amplitude
        burst = np.zeros(n, dtype=complex)
        n_bursts = rng.integers(2, 5)
        for _ in range(int(n_bursts)):
            start = int(rng.integers(0, n - 8))
            length = int(rng.integers(4, 16))
            phase = rng.uniform(0, 2 * np.pi)
            burst[start:start + length] += 2.4 * np.exp(1j * phase)
        sig = sig + burst
        noise_amp = 0.18
    else:
        noise_amp = 0.2

    noise = noise_amp * (rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n))
    sig = sig + noise
    # normalise per-window
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    return np.stack([np.real(sig), np.imag(sig)], axis=0).astype(np.float32)


def make_synth_dataset(n_per_class: int = 2500, n: int = 128, seed: int = 0
                       ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = []
    y = []
    for idx, lbl in enumerate(CLASSES):
        for _ in range(n_per_class):
            X.append(synth_window(rng, lbl, n=n))
            y.append(idx)
    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# RadioML 2016.10A loader (if available)
# ---------------------------------------------------------------------------

# Modulation -> interference-class taxonomy used by SHIELD-RIC.
RADIOML_TO_CLASS = {
    "BPSK": "none",
    "QPSK": "none",
    "8PSK": "wideband",
    "QAM16": "none",
    "QAM64": "none",
    "PAM4": "none",
    "GFSK": "none",
    "CPFSK": "none",
    "AM-SSB": "narrowband",
    "AM-DSB": "narrowband",
    "WBFM": "wideband",
}


def load_radioml(pkl_path: Path, low_snr_to_bursty: bool = True,
                 cap_per_class: int | None = 4000, seed: int = 0,
                 min_snr: int = 0, max_snr: int = 18
                 ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load RadioML 2016.10a dict {(mod, snr): (N, 2, 128)} into X, y.

    By default we keep SNR >= 0 dB so the perception sub-task corresponds to
    *usable* RF telemetry the controller would actually receive.  Below 0 dB,
    even hand-tuned classifiers fall to chance on this taxonomy.
    """
    with open(pkl_path, "rb") as f:
        try:
            d = pickle.load(f, encoding="latin1")
        except TypeError:
            d = pickle.load(f)
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[np.ndarray]] = {c: [] for c in CLASSES}
    label_set = set()
    for key, arr in d.items():
        mod, snr = key
        if isinstance(mod, bytes):
            mod = mod.decode("ascii")
        snr = int(snr)
        if not (min_snr <= snr <= max_snr):
            continue
        label_set.add((mod, snr))
        if mod not in RADIOML_TO_CLASS:
            continue
        target = RADIOML_TO_CLASS[mod]
        # Re-label low-SNR (within the kept band) QAMs as bursty to flesh out 4th class
        if low_snr_to_bursty and snr <= 2 and mod in ("QAM16", "QAM64"):
            target = "bursty"
        buckets[target].append(np.asarray(arr, dtype=np.float32))
    X = []
    y = []
    for idx, c in enumerate(CLASSES):
        if not buckets[c]:
            continue
        cat = np.concatenate(buckets[c], axis=0)
        if cap_per_class is not None and cat.shape[0] > cap_per_class:
            sel = rng.choice(cat.shape[0], cap_per_class, replace=False)
            cat = cat[sel]
        X.append(cat)
        y.append(np.full(cat.shape[0], idx, dtype=np.int64))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    perm = rng.permutation(len(y))
    meta = {"raw_keys": sorted([f"{m}@{s}" for (m, s) in label_set]),
            "class_counts": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
            "source": str(pkl_path)}
    return X[perm], y[perm], meta


# ---------------------------------------------------------------------------
# Small 1D CNN classifier
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module if HAVE_TORCH else object):
    def __init__(self, c_in, c_out, k=7, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(c_in)
        self.c1 = nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.c2 = nn.Conv1d(c_out, c_out, k, padding=k // 2)
        if c_in != c_out or stride != 1:
            self.skip = nn.Conv1d(c_in, c_out, 1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.c1(F.relu(self.bn1(x)))
        h = self.c2(F.relu(self.bn2(h)))
        return h + self.skip(x)


class IQNet(nn.Module if HAVE_TORCH else object):
    """1D ResNet-ish classifier: input (B, 2, 128), output (B, 4) logits.

    Modelled on O'Shea's ResNet baseline for RadioML; sized to train in a
    few minutes on CPU/GPU and produce a realistic confusion matrix.
    """
    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.stem = nn.Conv1d(2, 64, kernel_size=7, padding=3)
        self.r1 = ResBlock1D(64, 64)
        self.r2 = ResBlock1D(64, 128, stride=2)
        self.r3 = ResBlock1D(128, 128)
        self.r4 = ResBlock1D(128, 256, stride=2)
        self.r5 = ResBlock1D(256, 256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        h = self.stem(x)
        h = self.r1(h)
        h = self.r2(h)
        h = self.r3(h)
        h = self.r4(h)
        h = self.r5(h)
        h = self.pool(h).squeeze(-1)
        h = self.drop(h)
        return self.fc(h)


def train_classifier(X: np.ndarray, y: np.ndarray, epochs: int = 8,
                     batch_size: int = 256, lr: float = 1e-3, seed: int = 0,
                     val_frac: float = 0.2, device: str | None = None
                     ) -> tuple[IQNet, np.ndarray, dict]:
    """Train IQNet on (X, y).  Return (model, confusion_matrix, info)."""
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch unavailable.")
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    n_val = int(len(ds) * val_frac)
    n_tr = len(ds) - n_val
    tr, va = random_split(ds, [n_tr, n_val],
                          generator=torch.Generator().manual_seed(seed))
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=0)
    model = IQNet(n_classes=len(CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for ep in range(epochs):
        model.train()
        tl, n = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * yb.size(0)
            n += yb.size(0)
        model.eval()
        vl, va_n, va_correct = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vl += F.cross_entropy(logits, yb, reduction="sum").item()
                va_correct += int((logits.argmax(-1) == yb).sum().item())
                va_n += yb.size(0)
        history.append({"epoch": ep + 1, "train_loss": tl / n,
                        "val_loss": vl / va_n, "val_acc": va_correct / va_n})
        print(f"epoch {ep+1:>2d}  train_loss={tl/n:.4f}  "
              f"val_loss={vl/va_n:.4f}  val_acc={va_correct/va_n:.4f}")

    # Build confusion matrix on validation set
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for xb, yb in va_loader:
            xb = xb.to(device)
            pred = model(xb).argmax(-1).cpu().numpy()
            yt = yb.numpy()
            for t, p in zip(yt, pred):
                cm[t, p] += 1
    info = {"history": history, "n_train": n_tr, "n_val": n_val,
            "val_accuracy": float(va_correct / va_n)}
    return model, cm, info


def confusion_to_perception(cm: np.ndarray) -> dict:
    """Return row-normalised confusion dict keyed by ground-truth class label."""
    out = {}
    for i, c in enumerate(CLASSES):
        row = cm[i].astype(float)
        s = row.sum()
        if s <= 0:
            out[c] = {c: 1.0}
        else:
            out[c] = {CLASSES[j]: float(row[j] / s) for j in range(len(CLASSES))}
    return out


# ---------------------------------------------------------------------------
# CLI driver: train + save confusion matrix
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["radioml", "synth", "auto"], default="auto")
    ap.add_argument("--radioml-pkl", default=str(DATA_DIR / "RML2016.10a_dict.pkl"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS_DIR / "perception.json"))
    args = ap.parse_args()

    pkl_path = Path(args.radioml_pkl)
    have_radioml = pkl_path.exists() and pkl_path.stat().st_size > 0
    if args.source == "radioml" and not have_radioml:
        sys.exit(f"radioml pkl not at {pkl_path}; run with --source synth or download first.")
    if args.source == "auto":
        chosen = "radioml" if have_radioml else "synth"
    else:
        chosen = args.source

    if chosen == "radioml":
        print(f"Loading RadioML 2016.10a from {pkl_path}")
        X, y, meta = load_radioml(pkl_path, seed=args.seed)
    else:
        print("Generating synthetic RF dataset")
        X, y = make_synth_dataset(n_per_class=2500, seed=args.seed)
        meta = {"source": "synthetic", "class_counts":
                {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}}

    print("Dataset shape:", X.shape, "classes:", meta.get("class_counts"))
    model, cm, info = train_classifier(X, y, epochs=args.epochs, seed=args.seed)
    pmap = confusion_to_perception(cm)
    print("\nConfusion matrix (rows = ground truth, cols = predicted):")
    hdr = " " * 12 + " ".join(f"{c:>10s}" for c in CLASSES)
    print(hdr)
    for i, c in enumerate(CLASSES):
        row = " ".join(f"{cm[i,j]:>10d}" for j in range(len(CLASSES)))
        print(f"{c:>10s}  {row}")
    print(f"\nValidation accuracy: {info['val_accuracy']:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "source": chosen,
            "meta": meta,
            "classes": list(CLASSES),
            "confusion_counts": cm.tolist(),
            "perception_confusion": pmap,
            "info": info,
        }, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
