"""
Generate vector PDF figures for the SHIELD-RIC manuscript.

Inputs:
    results/master.csv          (sweep over condition x scenario x controller x seed)
    results/perception_radioml.json   (RadioML confusion matrix)

Outputs (each as both .pdf and .png in figs/):
    fig_violation_rate.pdf      headline: violations/cycle by controller, by scenario
    fig_perception_confusion.pdf  RadioML-trained classifier confusion
    fig_recovery_pdr.pdf        recovery time and PDR trade-off
    fig_shield_architecture.pdf  (preserved or regenerated -- block diagram)
    fig_shield_eval.pdf          (preserved or regenerated)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
RESULTS = SIM / "results"
FIGS = SIM / "figs"
FIGS.mkdir(exist_ok=True, parents=True)
PAPER_DIR = SIM.parent
TARGET_FIGS = PAPER_DIR  # final paper figures live next to main.tex

# -----  styling  ------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SCENARIO_ORDER = ["narrowband", "wideband", "bursty",
                  "backhaul", "compromised_xapp", "compound"]
SCENARIO_LABEL = {
    "narrowband": "Narrowband",
    "wideband": "Wideband",
    "bursty": "Bursty",
    "backhaul": "Backhaul",
    "compromised_xapp": "Comp. xApp",
    "compound": "Compound",
}
CTRL_ORDER = ["static", "heuristic", "rl", "llm_only", "shield_ric"]
CTRL_LABEL = {
    "static": "Static",
    "heuristic": "Heuristic",
    "rl": "Unconstr. RL",
    "llm_only": "LLM only",
    "shield_ric": "SHIELD-RIC",
}
# Colorblind-friendly palette (Okabe-Ito) with hatches retained for B&W fallback
CTRL_HATCH = {
    "static": "////",
    "heuristic": "\\\\\\\\",
    "rl": "xxxx",
    "llm_only": "....",
    "shield_ric": "",
}
CTRL_FACE = {
    "static":     "#56B4E9",   # sky blue
    "heuristic":  "#009E73",   # bluish green
    "rl":         "#E69F00",   # orange
    "llm_only":   "#D55E00",   # vermillion
    "shield_ric": "#0072B2",   # deep blue -- hero
}
CTRL_EDGE = {
    "static":     "#22526a",
    "heuristic":  "#005c3f",
    "rl":         "#8a5e00",
    "llm_only":   "#7a3500",
    "shield_ric": "#003255",
}


def load_master() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "master.csv")
    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER)
    df["controller"] = pd.Categorical(df["controller"], categories=CTRL_ORDER)
    return df


# ---------------------------------------------------------------------------
# Figure 1: Safety violations under realistic perception
# ---------------------------------------------------------------------------

def fig_violation_rate(df: pd.DataFrame, condition: str = "radioml",
                       out_name: str = "fig_violation_rate"):
    sub = df[df["condition"] == condition]
    agg = sub.groupby(["scenario", "controller"], observed=True)["violation_rate"]\
             .agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    width = 0.15
    x = np.arange(len(SCENARIO_ORDER))
    # zero-floor stub: invisible bars get a thin visible footprint so the
    # color/hatch coding remains identifiable in the figure.
    zero_stub = 0.012
    for i, ctrl in enumerate(CTRL_ORDER):
        s = agg[agg["controller"] == ctrl].set_index("scenario").reindex(SCENARIO_ORDER)
        means = s["mean"].values
        stds = s["std"].values
        plot_heights = np.where(means < 1e-6, zero_stub, means)
        ax.bar(x + (i - 2) * width, plot_heights, width,
               yerr=stds, capsize=2,
               label=CTRL_LABEL[ctrl],
               edgecolor=CTRL_EDGE[ctrl], facecolor=CTRL_FACE[ctrl],
               hatch=CTRL_HATCH[ctrl], linewidth=0.9)
        # annotate the zero bars so the reader sees the 0.000 explicitly
        for j, v in enumerate(means):
            if v < 1e-6:
                ax.text(x[j] + (i - 2) * width, zero_stub + 0.008, "0",
                        ha="center", va="bottom", fontsize=8,
                        color=CTRL_EDGE[ctrl])
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABEL[s] for s in SCENARIO_ORDER], fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("Safety violations per cycle", fontsize=11)
    ax.set_ylim(0, max(0.62, agg["mean"].max() * 1.18))
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              frameon=False, fontsize=10)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGS / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{out_name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Perception confusion (RadioML-trained classifier)
# ---------------------------------------------------------------------------

def fig_perception(out_name: str = "fig_perception_confusion"):
    pj = RESULTS / "perception_radioml.json"
    if not pj.exists():
        print("[skip] no perception_radioml.json")
        return
    d = json.loads(pj.read_text())
    cm = np.array(d["confusion_counts"], dtype=float)
    classes = d["classes"]
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    val_acc = d.get("info", {}).get("val_accuracy", float("nan"))
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="row-norm.")
    fig.tight_layout()
    fig.savefig(FIGS / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{out_name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: PDR vs violations -- the safety/throughput Pareto
# ---------------------------------------------------------------------------

def fig_pdr_vs_violations(df: pd.DataFrame, out_name: str = "fig_pdr_vs_violations"):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    markers = {"perfect": "o", "radioml": "s", "synthetic": "^"}
    for cond in ["perfect", "radioml"]:
        sub = df[df["condition"] == cond]
        for ctrl in CTRL_ORDER:
            s = sub[sub["controller"] == ctrl]
            mu_pdr = s["mean_pdr"].mean()
            mu_v = s["violation_rate"].mean()
            ax.scatter(mu_v, mu_pdr, marker=markers[cond],
                       facecolor=CTRL_FACE[ctrl],
                       edgecolor=CTRL_EDGE[ctrl], s=70, linewidth=0.9,
                       label=f"{CTRL_LABEL[ctrl]} ({cond})" if cond == "radioml" else None)
    # only annotate shield_ric+heuristic for clarity
    annot = {("shield_ric", "radioml"): "SHIELD-RIC",
             ("heuristic", "radioml"): "Heuristic",
             ("llm_only", "radioml"): "LLM only",
             ("static", "radioml"): "Static"}
    for (ctrl, cond), txt in annot.items():
        s = df[(df["condition"] == cond) & (df["controller"] == ctrl)]
        ax.annotate(txt, (s["violation_rate"].mean(), s["mean_pdr"].mean()),
                    xytext=(6, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Mean safety-violation rate")
    ax.set_ylabel("Mean priority-flow PDR")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(0.5, 1.02)
    ax.axvspan(-0.02, 0.001, color="#f0f0f0", alpha=0.6, label="zero-violation region")
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGS / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{out_name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Recovery time across scenarios (static fails)
# ---------------------------------------------------------------------------

def fig_recovery(df: pd.DataFrame, condition: str = "radioml",
                 out_name: str = "fig_recovery"):
    sub = df[df["condition"] == condition]
    agg = sub.groupby(["scenario", "controller"], observed=True)["recovery_cycles"]\
             .mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    width = 0.15
    x = np.arange(len(SCENARIO_ORDER))
    for i, ctrl in enumerate(CTRL_ORDER):
        s = agg[agg["controller"] == ctrl].set_index("scenario").reindex(SCENARIO_ORDER)
        ax.bar(x + (i - 2) * width, s["recovery_cycles"].values, width,
               label=CTRL_LABEL[ctrl],
               edgecolor=CTRL_EDGE[ctrl], facecolor=CTRL_FACE[ctrl],
               hatch=CTRL_HATCH[ctrl], linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABEL[s] for s in SCENARIO_ORDER], fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("Recovery cycles to PDR > 0.9", fontsize=11)
    ax.set_yscale("log")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              frameon=False, fontsize=10)
    ax.grid(axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGS / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{out_name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Architecture diagram (regenerated as vector)
# ---------------------------------------------------------------------------

def fig_architecture(out_name: str = "fig_shield_architecture"):
    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.set_axis_off()

    def box(x, y, w, h, txt, face="#ffffff", edge="#222222", fontsize=8):
        from matplotlib.patches import FancyBboxPatch
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.18",
                           ec=edge, fc=face, lw=1.0)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fontsize, wrap=True)

    def arrow(x1, y1, x2, y2, label="", color="#333333"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color=color))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label,
                    ha="center", va="bottom", fontsize=7, style="italic",
                    color=color)

    # Telemetry (cool gray)
    box(0.2, 1.6, 1.7, 1.2, "Signed\ntelemetry\n$o_t$",
        face="#E8EEF3", edge="#5C7080")
    # LLM planner (yellow/amber -- "reasoning")
    box(2.3, 1.6, 2.0, 1.2, "LLM\nplanner\n(propose $\\mathcal{A}^{cand}$)",
        face="#FFE6A8", edge="#A8771C")
    # Digital twin (green)
    box(4.7, 0.4, 2.3, 1.2, "Digital twin\n$\\hat{r},\\hat{v},\\hat{d}$",
        face="#C8E6C9", edge="#1B5E20")
    # Policy / identity (red -- gating)
    box(4.6, 2.8, 2.5, 1.2,
        "Policy predicates $\\Phi$\n(Eqs. 3--8)",
        face="#FBD0C2", edge="#8B0000")
    # Safety shield (deep blue -- hero).  Custom box draws white text on dark fill.
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((7.4, 1.6), 2.6, 1.2,
                       boxstyle="round,pad=0.05,rounding_size=0.18",
                       ec="#003255", fc="#0072B2", lw=1.2)
    ax.add_patch(p)
    ax.text(7.4 + 1.3, 1.6 + 0.6, "Safety shield\n$\\Pi_\\Phi$ (Eq. 9)",
            ha="center", va="center", fontsize=8.5, color="white",
            fontweight="bold")
    # Actuation (lighter blue)
    box(10.4, 1.6, 2.0, 1.2, "Bounded\nRIC actuation",
        face="#9BC4E2", edge="#1F4E79")
    # Rollback (purple)
    box(12.7, 1.6, 1.2, 1.2, "Rollback\nmonitor",
        face="#D5C2EE", edge="#4B2E83")

    arrow(1.9, 2.2, 2.3, 2.2)
    arrow(4.3, 2.2, 4.6, 1.2)
    arrow(4.3, 2.2, 4.6, 3.5)
    arrow(4.3, 2.2, 7.4, 2.2)
    arrow(7.1, 1.0, 7.4, 1.9, color="#1B5E20")
    arrow(7.1, 3.4, 7.4, 2.5, color="#8B0000")
    arrow(10.0, 2.2, 10.4, 2.2, "$a^\\star$", color="#003255")
    arrow(12.4, 2.2, 12.7, 2.2)
    arrow(13.3, 1.6, 13.3, 0.4, color="#4B2E83")
    ax.annotate("rollback if monitored utility falls below guard",
                xy=(13.3, 0.4), xytext=(8.5, 0.05),
                fontsize=7, style="italic", color="#4B2E83",
                arrowprops=dict(arrowstyle="-", lw=0.5, ls=":", color="#4B2E83"))

    fig.savefig(FIGS / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{out_name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Eval flow diagram (compact)
# ---------------------------------------------------------------------------

def fig_eval(out_name: str = "fig_shield_eval"):
    fig, ax = plt.subplots(figsize=(6.8, 1.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 3); ax.set_axis_off()
    from matplotlib.patches import FancyBboxPatch
    def box(x, y, w, h, txt, face, edge, fontsize=8):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.18",
                           ec=edge, fc=face, lw=1.0)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fontsize)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#333333"))
    box(0.05, 0.5, 2.5, 2.0,
        "RadioML\n2016.10A\n($\\geq$ 0 dB SNR)",
        face="#E8EEF3", edge="#5C7080", fontsize=8.5)
    box(2.9, 0.5, 2.2, 2.0, "ResNet-1D\nclassifier",
        face="#FFE6A8", edge="#A8771C", fontsize=8.5)
    box(5.5, 0.5, 2.6, 2.0,
        "SHIELD-RIC\nsimulator\n(6 scenarios)",
        face="#C8E6C9", edge="#1B5E20", fontsize=8.5)
    box(8.4, 0.5, 2.5, 2.0,
        "5 controllers\nablation",
        face="#9BC4E2", edge="#1F4E79", fontsize=8.5)
    box(11.2, 0.5, 2.7, 2.0,
        "PDR, recovery,\nsafety viol.,\nrollback rate",
        face="#D5C2EE", edge="#4B2E83", fontsize=8.5)
    arrow(2.55, 1.5, 2.9, 1.5)
    arrow(5.10, 1.5, 5.5, 1.5)
    arrow(8.10, 1.5, 8.4, 1.5)
    arrow(10.9, 1.5, 11.2, 1.5)
    fig.savefig(FIGS / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{out_name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Master tableau printed to stdout (for manuscript)
# ---------------------------------------------------------------------------

def print_table(df: pd.DataFrame):
    for cond in ["perfect", "radioml"]:
        print(f"\n=== Condition: {cond} ===")
        sub = df[df["condition"] == cond]
        agg = sub.groupby(["scenario", "controller"], observed=True).agg(
            pdr=("mean_pdr", "mean"),
            pdr_std=("mean_pdr", "std"),
            viol=("violation_rate", "mean"),
            rec=("recovery_cycles", "mean"),
            rew=("mean_reward", "mean"),
        ).round(3)
        print(agg.to_string())


def main():
    df = load_master()
    fig_violation_rate(df, condition="radioml")
    fig_perception()
    fig_pdr_vs_violations(df)
    fig_recovery(df, condition="radioml")
    fig_architecture()
    fig_eval()
    print_table(df)
    # Copy the key figures into the paper directory
    for name in ("fig_shield_architecture.pdf", "fig_shield_eval.pdf",
                 "fig_violation_rate.pdf", "fig_perception_confusion.pdf",
                 "fig_pdr_vs_violations.pdf", "fig_recovery.pdf"):
        src = FIGS / name
        dst = TARGET_FIGS / name
        if src.exists():
            dst.write_bytes(src.read_bytes())
            print(f"copied {src} -> {dst}")


if __name__ == "__main__":
    main()
