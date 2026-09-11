"""
Journal edition -- figure generation.

All figures are vector PDF, sized for IEEE two-column (single column 3.45 in,
double column 7.16 in), typeset in a serif face to match the body text, and
coloured with the Okabe-Ito colour-blind-safe palette with redundant hatching
so that they survive greyscale printing.

Author: Liang Dong.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE / "figs"
PAPER = HERE.parent.parent
FIGS.mkdir(exist_ok=True, parents=True)

COL1, COL2 = 3.45, 7.16

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.1,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Okabe-Ito
OI = {"blue": "#0072B2", "orange": "#E69F00", "sky": "#56B4E9",
      "green": "#009E73", "yellow": "#F0E442", "vermillion": "#D55E00",
      "purple": "#CC79A7", "grey": "#7F7F7F", "black": "#000000"}

CTRL_ORDER = ["static", "heuristic", "greedy_twin", "llm_only",
              "lagrangian_rl", "simplex_rta", "shield_filter", "shield_repair"]
CTRL_LABEL = {"static": "Static", "heuristic": "Heuristic",
              "greedy_twin": "Greedy-Twin", "llm_only": "LLM-only",
              "lagrangian_rl": "Lagrangian-RL", "simplex_rta": "Reactive-RTA",
              "shield_filter": "Shield (filter)", "shield_repair": "Shield (repair)"}
CTRL_COLOR = {"static": OI["grey"], "heuristic": OI["yellow"],
              "greedy_twin": OI["vermillion"], "llm_only": OI["orange"],
              "lagrangian_rl": OI["purple"], "simplex_rta": OI["sky"],
              "shield_filter": OI["green"], "shield_repair": OI["blue"]}
CTRL_HATCH = {"static": "", "heuristic": "..", "greedy_twin": "//",
              "llm_only": "\\\\", "lagrangian_rl": "xx", "simplex_rta": "--",
              "shield_filter": "++", "shield_repair": ""}

SC_ORDER = ["narrowband", "wideband", "bursty", "backhaul",
            "compromised_xapp", "compound"]
SC_LABEL = {"narrowband": "Narrowband", "wideband": "Wideband", "bursty": "Bursty",
            "backhaul": "Backhaul", "compromised_xapp": "Comp. xApp",
            "compound": "Compound"}
HAZ_LABEL = {"h_rf": "Locked waveform", "h_pwr": "Reg. power cap",
             "h_relay": "Unauth. relay", "h_part": "Partition breach",
             "h_starve": "Priority starvation", "h_therm": "PA over-temp."}
HAZ_COLOR = {"h_rf": OI["vermillion"], "h_pwr": OI["orange"],
             "h_relay": OI["purple"], "h_part": OI["sky"],
             "h_starve": OI["green"], "h_therm": OI["blue"]}


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}")
    (PAPER / f"{name}.pdf").write_bytes((FIGS / f"{name}.pdf").read_bytes())
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def _grid(ax, axis="y"):
    ax.grid(axis=axis, linestyle=":", linewidth=0.4, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _ci(x, B=4000, seed=0):
    x = np.asarray(x, float)
    if x.size < 2:
        return 0.0, 0.0
    r = np.random.default_rng(seed)
    b = x[r.integers(0, x.size, (B, x.size))].mean(1)
    m = x.mean()
    return m - np.quantile(b, .025), np.quantile(b, .975) - m


# ---------------------------------------------------------------------------
# F3: policy violations vs latent hazards
# ---------------------------------------------------------------------------

def fig_safety(master: pd.DataFrame):
    d = master[(master.planner == "naive") & (master.predicate_set == "phi") &
               (master.perception == "radioml")]
    fig, axes = plt.subplots(2, 1, figsize=(COL2, 2.55), sharex=True)
    w = 0.10
    x = np.arange(len(SC_ORDER))
    for ax, metric, ylab in zip(
            axes, ["violation_rate_phi", "hazard_rate"],
            ["Policy violations\nper cycle", "Latent hazards\nper cycle"]):
        for i, ct in enumerate(CTRL_ORDER):
            sub = d[d.controller == ct]
            vals, err = [], [[], []]
            for k, sc in enumerate(SC_ORDER):
                v = sub[sub.scenario == sc][metric].to_numpy()
                vals.append(v.mean() if v.size else np.nan)
                lo, hi = _ci(v, seed=i * 101 + k)
                err[0].append(lo)
                err[1].append(hi)
            pos = x + (i - 3.5) * w
            ax.bar(pos, vals, w, label=CTRL_LABEL[ct], color=CTRL_COLOR[ct],
                   edgecolor="black", linewidth=0.35, hatch=CTRL_HATCH[ct],
                   zorder=3)
            ax.errorbar(pos, vals, yerr=err, fmt="none", ecolor="black",
                        elinewidth=0.4, capsize=0.8, zorder=4)
            # A bar of height zero is invisible; mark the exact zeros so the
            # reader can tell "zero" from "absent".  Black, not the bar colour:
            # yellow on white is unreadable, and the x position already says
            # which controller a mark belongs to.
            top = ax.get_ylim()[1] or 1.0
            for px, v in zip(pos, vals):
                if v == 0.0:
                    ax.text(px, 0.006 * top, "0", ha="center", va="bottom",
                            fontsize=5.2, color="black", fontweight="bold",
                            zorder=5)
        ax.set_ylabel(ylab, fontsize=7.6)
        _grid(ax)
    axes[0].set_ylim(0, max(0.56, axes[0].get_ylim()[1]))
    axes[0].annotate("Static, Heuristic and both shielded controllers:\n"
                     "exactly $0.000$ in every scenario and every seed",
                     xy=(0.012, 0.94), xycoords="axes fraction", ha="left",
                     va="top", fontsize=6.6, style="italic",
                     bbox=dict(boxstyle="round,pad=0.28", fc="white",
                               ec=OI["blue"], lw=0.5))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([SC_LABEL[s] for s in SC_ORDER])
    axes[0].legend(ncol=4, frameon=False, loc="upper left",
                   bbox_to_anchor=(0.0, 1.40), columnspacing=1.1,
                   handlelength=1.5, handletextpad=0.4)
    fig.tight_layout()
    _save(fig, "fig_safety")


# ---------------------------------------------------------------------------
# F4: safety-utility frontier
# ---------------------------------------------------------------------------

def fig_frontier(master: pd.DataFrame):
    d = master[(master.planner == "naive") & (master.predicate_set == "phi") &
               (master.perception == "radioml")]
    g = d.groupby("controller").agg(pdr=("mean_pdr", "mean"),
                                    haz=("hazard_rate", "mean"),
                                    vio=("violation_rate_phi", "mean"))
    fig, ax = plt.subplots(figsize=(COL1, 2.15))
    ax.axvspan(-0.010, 0.004, color=OI["green"], alpha=0.13, zorder=0)
    for ct in CTRL_ORDER:
        if ct not in g.index:
            continue
        r = g.loc[ct]
        m = "*" if ct == "shield_repair" else ("D" if ct.startswith("shield") else "o")
        ax.scatter(r.vio, r.pdr, s=90 if ct == "shield_repair" else 42,
                   marker=m, color=CTRL_COLOR[ct], edgecolor="black",
                   linewidth=0.4, zorder=4, label=CTRL_LABEL[ct])
    ax.set_xlabel(r"Policy-violation rate (lower better)")
    ax.set_ylabel("Mean priority-flow PDR")
    ax.text(0.008, ax.get_ylim()[1] - 0.004, "operationally\nadmissible",
            fontsize=6.3, color=OI["green"], ha="left", va="top")
    _grid(ax, axis="both")
    ax.legend(frameon=False, fontsize=6.3, ncol=2, loc="lower right",
              handletextpad=0.2, columnspacing=0.6, borderpad=0.2)
    fig.tight_layout()
    _save(fig, "fig_frontier")


# ---------------------------------------------------------------------------
# F5: predicate-coverage gap under the adaptive adversary
# ---------------------------------------------------------------------------

def fig_coverage_gap(master: pd.DataFrame):
    d = master[(master.planner == "adaptive") & (master.perception == "radioml")]
    hz = list(HAZ_LABEL)
    # Only controllers that APPLY an admission test belong here.  The adaptive
    # adversary restricts itself to admissible candidates because a shield
    # would reject anything else; against a victim that applies no test it
    # would simply attack directly, which is the overt adversary already
    # reported.  Including unshielded controllers here would credit them with
    # safety supplied by the attacker's own restraint.
    ctrls = ["heuristic", "simplex_rta", "shield_filter", "shield_repair"]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 1.85), sharey=True)
    for ax, ps, title in zip(axes, ["phi", "phi_plus"],
                             [r"deployed set $\Phi$ (6 predicates)",
                              r"augmented set $\Phi^{+}$ (10 predicates)"]):
        sub = d[d.predicate_set == ps]
        x = np.arange(len(ctrls))
        bottom = np.zeros(len(ctrls))
        for h in hz:
            vals = np.array([sub[sub.controller == c][h].mean() for c in ctrls])
            ax.bar(x, vals, 0.62, bottom=bottom, label=HAZ_LABEL[h],
                   color=HAZ_COLOR[h], edgecolor="black", linewidth=0.3, zorder=3)
            bottom += vals
        for xi, v in zip(x, bottom):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.4)
        ax.set_xticks(x)
        ax.set_xticklabels([CTRL_LABEL[c] for c in ctrls], rotation=18,
                           ha="right")
        ax.set_title(title, fontsize=7.8)
        _grid(ax)
    axes[0].set_ylabel("Latent hazards per cycle")
    axes[1].legend(frameon=False, fontsize=6.4, ncol=1, loc="upper right",
                   handlelength=1.1, handletextpad=0.4, borderpad=0.2)
    fig.tight_layout()
    _save(fig, "fig_coverage_gap")


# ---------------------------------------------------------------------------
# F6: Theorem 1
# ---------------------------------------------------------------------------

def fig_theorem():
    t1 = pd.read_csv(RESULTS / "theorem_scenario.csv")
    t2 = pd.read_csv(RESULTS / "theorem_mismatch.csv")
    t3 = pd.read_csv(RESULTS / "theorem_budget.csv")
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.62))

    # (a) per-scenario decomposition vs measured
    ax = axes[0]
    t1 = t1.set_index("scenario").loc[SC_ORDER].reset_index()
    x = np.arange(len(t1))
    w = 0.36
    ax.bar(x - w / 2, t1.bound_twin, w, label=r"$2\xi_\delta$",
           color=OI["sky"], edgecolor="black", linewidth=0.3, zorder=3)
    ax.bar(x - w / 2, t1.bound_eps, w, bottom=t1.bound_twin,
           label=r"$\epsilon$", color=OI["yellow"], edgecolor="black",
           linewidth=0.3, zorder=3)
    ax.bar(x - w / 2, t1.bound_cov, w, bottom=t1.bound_twin + t1.bound_eps,
           label=r"$r_{\max}(1-\mathrm{cov}_{K,\epsilon})$", color=OI["orange"],
           edgecolor="black", linewidth=0.3, zorder=3)
    ax.bar(x - w / 2, t1.bound_delta, w,
           bottom=t1.bound_twin + t1.bound_eps + t1.bound_cov,
           label=r"$r_{\max}\delta$", color=OI["purple"], edgecolor="black",
           linewidth=0.3, zorder=3)
    ax.bar(x + w / 2, t1.measured_regret, w, label="measured gap",
           color=OI["blue"], edgecolor="black", linewidth=0.3, zorder=3)
    ax.axhline(1.0, color=OI["vermillion"], lw=0.7, ls="--", zorder=2)
    ax.text(-0.35, 1.02, r"$r_{\max}$", fontsize=6.4,
            color=OI["vermillion"], ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels([SC_LABEL[s] for s in t1.scenario], rotation=32,
                       ha="right", fontsize=6.3)
    ax.set_ylabel("Per-cycle safe-action gap")
    ax.set_title("(a) bound vs. measurement", fontsize=7.4)
    TOP = 1.15
    ax.set_ylim(0, TOP)
    # Bars taller than the axis are clipped; print their value so a truncated
    # bar is never mistaken for one that stops at the limit.
    total = (t1.bound_twin + t1.bound_eps + t1.bound_cov + t1.bound_delta)
    for xi, v in zip(x, total):
        if v > TOP:
            ax.annotate(f"{v:.2f}", xy=(xi - w / 2, TOP), xytext=(0, 1.5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=5.4, fontweight="bold", clip_on=False)
    ax.legend(frameon=False, fontsize=5.9, loc="upper right", handlelength=1.0,
              handletextpad=0.35, labelspacing=0.25, borderpad=0.15,
              bbox_to_anchor=(1.02, 1.02))
    _grid(ax)

    # (b) twin-mismatch sweep
    ax = axes[1]
    ax.plot(t2.mismatch, t2.bound_total, "o-", color=OI["orange"],
            label="bound", ms=3)
    ax.plot(t2.mismatch, t2.measured_regret, "s-", color=OI["blue"],
            label="measured", ms=3)
    ax.plot(t2.mismatch, t2.xi_delta, "^--", color=OI["sky"],
            label=r"$\xi_\delta$", ms=3)
    ax.set_xlabel("twin structural mismatch $m$")
    ax.set_title("(b) twin quality", fontsize=7.4)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=6.4, loc="center right")
    _grid(ax, "both")

    # (c) candidate-budget sweep
    ax = axes[2]
    ax.plot(t3.K, t3.bound_total, "o-", color=OI["orange"], label="bound", ms=3)
    ax.plot(t3.K, t3.measured_regret, "s-", color=OI["blue"], label="measured", ms=3)
    ax2 = ax.twinx()
    ax2.plot(t3.K, t3.cov_eps, "^:", color=OI["green"], ms=3,
             label=r"$\mathrm{cov}_{K,\epsilon}$")
    ax2.set_ylabel(r"$\mathrm{cov}_{K,\epsilon}$", color=OI["green"], fontsize=7.4)
    ax2.tick_params(axis="y", colors=OI["green"], labelsize=6.6)
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel("candidate budget $K$")
    ax.set_xscale("log")
    ax.set_xticks(t3.K)
    ax.set_xticklabels([str(int(v)) for v in t3.K])
    ax.set_yscale("log")
    ax.set_title("(c) planner coverage", fontsize=7.4)
    ax.legend(frameon=False, fontsize=6.4, loc="center right")
    _grid(ax, "both")

    fig.tight_layout()
    _save(fig, "fig_theorem")


# ---------------------------------------------------------------------------
# F7: repair vs filter
# ---------------------------------------------------------------------------

def fig_repair(master: pd.DataFrame):
    t4 = pd.read_csv(RESULTS / "theorem_repair.csv")
    d = master[(master.planner == "naive") & (master.predicate_set == "phi") &
               (master.perception == "radioml")]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.78))
    x = np.arange(len(SC_ORDER))
    w = 0.38
    pairs = [(False, "filter only", OI["green"], "++"),
             (True, "with repair", OI["blue"], "")]
    FLOOR = 1e-4

    # (a) coverage DEFICIT on a log axis: the improvement is multiplicative and
    #     invisible on a linear coverage axis near 1.  A log axis cannot draw
    #     zero, so exact zeros are clamped to the floor and marked "0" -- the
    #     reader must not read the floor as a measured value.
    ax = axes[0]
    for j, (rep, lab, col, ha) in enumerate(pairs):
        s4 = t4[t4.repair == rep].set_index("scenario").reindex(SC_ORDER)
        deficit = 1.0 - s4.cov_eps.to_numpy()
        pos = x + (j - .5) * w
        ax.bar(pos, np.maximum(deficit, FLOOR), w, label=lab, color=col,
               edgecolor="black", linewidth=0.3, hatch=ha, zorder=3)
        for px, v in zip(pos, deficit):
            if v <= 0:
                ax.text(px, FLOOR * 1.35, "0", ha="center", va="bottom",
                        fontsize=5.4, fontweight="bold", color="black",
                        zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.75, None)
    ax.set_ylabel(r"$1-\mathrm{cov}_{K,\epsilon}$  (lower better)", fontsize=7.2)
    ax.set_title("(a) planner coverage", fontsize=7.4)

    # (b) measured one-step gap
    ax = axes[1]
    for j, (rep, lab, col, ha) in enumerate(pairs):
        s4 = t4[t4.repair == rep].set_index("scenario").reindex(SC_ORDER)
        ax.bar(x + (j - .5) * w, s4.measured_regret, w, color=col,
               edgecolor="black", linewidth=0.3, hatch=ha, zorder=3)
    ax.set_ylabel("measured gap / cycle", fontsize=7.4)
    ax.set_title("(b) one-step regret", fontsize=7.4)

    # (c) delivered throughput
    ax = axes[2]
    for j, (ct, col, ha) in enumerate(
            [("shield_filter", OI["green"], "++"),
             ("shield_repair", OI["blue"], "")]):
        vals = [d[(d.controller == ct) & (d.scenario == sc)].mean_pdr.mean()
                for sc in SC_ORDER]
        ax.bar(x + (j - .5) * w, vals, w, color=col, edgecolor="black",
               linewidth=0.3, hatch=ha, zorder=3)
    hv = [d[(d.controller == "heuristic") & (d.scenario == sc)].mean_pdr.mean()
          for sc in SC_ORDER]
    ax.plot(x, hv, "k_", ms=13, mew=1.1, zorder=6)
    ax.set_ylabel("mean PDR", fontsize=7.4)
    lo = min([min(hv)] + [d[(d.controller == c) & (d.scenario == sc)].mean_pdr.mean()
                          for c in ("shield_filter", "shield_repair")
                          for sc in SC_ORDER])
    ax.set_ylim(max(0.0, lo - 0.03), 1.02)
    ax.set_title("(c) delivered throughput", fontsize=7.4)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([SC_LABEL[s] for s in SC_ORDER], rotation=34,
                           ha="right", fontsize=6.2)
        _grid(ax)

    # one legend for all three panels; the in-axes version collided with the
    # tall wideband and compound bars in (a)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=OI["green"], edgecolor="black",
                              linewidth=0.3, hatch="++", label="filter only"),
                        Patch(facecolor=OI["blue"], edgecolor="black",
                              linewidth=0.3, label="with repair"),
                        Line2D([0], [0], color="black", marker="_", ms=9,
                               mew=1.1, ls="none",
                               label="Heuristic, oracle-designed (c)")],
               loc="upper center", ncol=3, frameon=False, fontsize=6.4,
               handlelength=1.3, handletextpad=0.4, columnspacing=1.4,
               bbox_to_anchor=(0.5, 1.015))
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    _save(fig, "fig_repair")


# ---------------------------------------------------------------------------
# F8: multi-cell
# ---------------------------------------------------------------------------

def fig_multicell():
    d = pd.read_csv(RESULTS / "multicell.csv")
    s = pd.read_csv(RESULTS / "multicell_scale.csv")
    order = ["per_cell", "static_share", "sequential"]
    lab = {"per_cell": "Per-cell shield", "static_share": "Static share",
           "sequential": "Alg. 3 (budgeted)"}
    col = {"per_cell": OI["vermillion"], "static_share": OI["orange"],
           "sequential": OI["blue"]}
    hat = {"per_cell": "//", "static_share": "..", "sequential": ""}

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.78))
    x = np.arange(len(SC_ORDER))
    w = 0.26
    for ax, metric, ylab, title in zip(
            axes[:2], ["psi_violation_rate", "mean_pdr"],
            [r"cluster-budget violation rate", "mean PDR"],
            [r"(a) coupled predicate $\psi$", "(b) throughput"]):
        for j, sch in enumerate(order):
            v = [d[(d.scheme == sch) & (d.scenario == sc)][metric].mean()
                 for sc in SC_ORDER]
            pos = x + (j - 1) * w
            ax.bar(pos, v, w, label=lab[sch], color=col[sch],
                   edgecolor="black", linewidth=0.3, hatch=hat[sch], zorder=3)
            if metric == "psi_violation_rate":
                for px, vv in zip(pos, v):
                    if vv == 0.0:
                        ax.text(px, 0.006, "0", ha="center", va="bottom",
                                fontsize=5.4, color=col[sch],
                                fontweight="bold", zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([SC_LABEL[c] for c in SC_ORDER], rotation=34,
                           ha="right", fontsize=6.2)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=7.4)
        _grid(ax)
    lo = float(d.groupby(["scheme", "scenario"]).mean_pdr.mean().min())
    axes[1].set_ylim(max(0.0, lo - 0.03), 1.0)
    axes[0].legend(frameon=False, fontsize=6.3, loc="upper left")

    ax = axes[2]
    for sch in order:
        g = s[s.scheme == sch].groupby("n_cells").mean_pdr.mean()
        ax.plot(g.index, g.values, "o-", color=col[sch], ms=3, label=lab[sch])
    ax.set_xlabel("cells in cluster $C$")
    ax.set_ylabel("mean PDR")
    ax.set_title("(c) cluster size", fontsize=7.4)
    _grid(ax, "both")
    ax.legend(frameon=False, fontsize=6.3, loc="lower left")
    fig.tight_layout()
    _save(fig, "fig_multicell")


# ---------------------------------------------------------------------------
# F9: LLM planner and prompt injection
# ---------------------------------------------------------------------------

def fig_llm():
    p = RESULTS / "llm.csv"
    if not p.exists():
        print("  [skip] llm.csv absent")
        return
    d = pd.read_csv(p)
    g = d.groupby(["inject", "mode"]).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(COL1, 1.85))
    x = np.arange(2)
    w = 0.36

    def zeros(ax, pos, vals, col):
        for px, v in zip(pos, vals):
            if v == 0.0:
                ax.text(px, 0.012, "0", ha="center", va="bottom", fontsize=5.4,
                        color=col, fontweight="bold", zorder=6)

    ax = axes[0]
    for j, (mode, lab, c, h) in enumerate(
            [("unshielded", "LLM-only", OI["vermillion"], "//"),
             ("shield_repair", "Shield (repair)", OI["blue"], "")]):
        pos = x + (j - .5) * w
        cand = [g.loc[(inj, mode), "cand_violation_rate"] for inj in (False, True)]
        act = [g.loc[(inj, mode), "violation_rate_phi"] for inj in (False, True)]
        ax.bar(pos, cand, w, color="white", edgecolor=c, linewidth=1.0,
               hatch=h, zorder=3, label=f"{lab}: candidates")
        ax.bar(pos, act, w * 0.55, color=c, edgecolor="black", linewidth=0.35,
               zorder=4, label=f"{lab}: actuated")
        zeros(ax, pos, act, c)
    ax.set_ylabel(r"fraction violating $\Phi^{+}$", fontsize=7.0)
    ax.set_title("(a) planner vs.\nactuator", fontsize=7.0)
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    for j, (mode, lab, c, h) in enumerate(
            [("unshielded", "LLM-only", OI["vermillion"], "//"),
             ("shield_repair", "Shield (repair)", OI["blue"], "")]):
        pos = x + (j - .5) * w
        v = [g.loc[(inj, mode), "hazard_rate"] for inj in (False, True)]
        ax.bar(pos, v, w, color=c, edgecolor="black", linewidth=0.35, hatch=h,
               label=lab, zorder=3)
        zeros(ax, pos, v, c)
    ax.set_ylabel("latent hazard rate", fontsize=7.0)
    ax.set_title("(b) latent hazards", fontsize=7.0)
    ax.set_ylim(0, 1.05)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(["benign", "injected"], fontsize=6.6)
        _grid(ax)
    # one legend for both panels: outlined = candidate level, solid = actuated
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="white", edgecolor=OI["vermillion"],
                              linewidth=1.0, hatch="//", label="LLM-only"),
                        Patch(facecolor="white", edgecolor=OI["blue"],
                              linewidth=1.0, label="Shield (repair)")],
               loc="upper center", ncol=2, frameon=False, fontsize=6.0,
               handlelength=1.2, handletextpad=0.35, columnspacing=1.2,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    _save(fig, "fig_llm")


# ---------------------------------------------------------------------------
# F10: per-cycle trace
# ---------------------------------------------------------------------------

def fig_trace(scenario: str = "compound", seed: int = 3, cycles: int = 300):
    """Per-cycle dynamics for one episode: running counts of policy violations
    and latent hazards.  Drawn at SINGLE-column width, because that is how the
    manuscript includes it -- a COL2 figure squeezed into \\columnwidth halves
    every font, and 6.4 pt legend text becomes 3.2 pt on the page."""
    import core
    import experiments as ex

    conf = ex.load_confusion("radioml")
    shown = [("heuristic", OI["yellow"], "-."),
             ("greedy_twin", OI["vermillion"], "--"),
             ("shield_repair", OI["blue"], "-")]
    traces = {}
    for ct, _, _ in shown:
        rows = []
        ex.run_episode(scenario, ct, seed, planner_kind="naive",
                       predicate_set="phi", perception="radioml",
                       cycles=cycles, confusion=conf, log_rows=rows)
        traces[ct] = pd.DataFrame(rows)

    onset, end = core.SCENARIOS[scenario]["onset"], core.SCENARIOS[scenario]["end"]
    fig, ax = plt.subplots(figsize=(COL1, 1.72))
    ax.axvspan(onset, end, color="black", alpha=0.045, zorder=0)
    for ct, col, ls in shown:
        t = traces[ct]
        ax.plot(t.cycle, t.violation_phi.cumsum(), ls, color=col, lw=1.1,
                zorder=3, label=CTRL_LABEL[ct])
        ax.plot(t.cycle, t.hazard.cumsum(), ls, color=col, lw=0.8, alpha=0.42,
                zorder=2)
    ax.set_ylabel("cumulative count", fontsize=7.0)
    ax.set_xlabel("control cycle", fontsize=7.0)
    ax.tick_params(labelsize=6.4)
    ax.set_xticks([0, 100, 200, 300])
    ax.legend(ncol=3, frameon=False, loc="lower center", fontsize=6.0,
              bbox_to_anchor=(0.5, 1.00), handlelength=1.6,
              handletextpad=0.35, columnspacing=0.9)
    ax.text(onset + 6, ax.get_ylim()[1] * 0.94, "impairment", fontsize=6.0,
            color="black", alpha=0.65, va="top")
    _grid(ax, "both")
    fig.tight_layout()
    _save(fig, "fig_trace")


# ---------------------------------------------------------------------------
# F2: perception confusion matrix
# ---------------------------------------------------------------------------

def fig_confusion():
    p = RESULTS / "perception_radioml.json"
    d = json.loads(p.read_text())
    classes = d["classes"]
    M = np.array([[d["perception_confusion"][t][pr] for pr in classes]
                  for t in classes])
    fig, ax = plt.subplots(figsize=(COL1 * 0.74, COL1 * 0.62))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.8, color="white" if M[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel(r"predicted class $\hat\iota$")
    ax.set_ylabel(r"true class $\iota$")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6.4)
    fig.tight_layout()
    _save(fig, "fig_perception_confusion")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(RESULTS / "master.csv"))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    master = pd.read_csv(args.master)
    jobs = {
        "confusion": lambda: fig_confusion(),
        "safety": lambda: fig_safety(master),
        "frontier": lambda: fig_frontier(master),
        "coverage": lambda: fig_coverage_gap(master),
        "theorem": lambda: fig_theorem(),
        "repair": lambda: fig_repair(master),
        "multicell": lambda: fig_multicell(),
        "llm": lambda: fig_llm(),
        "trace": lambda: fig_trace(),
    }
    for name, fn in jobs.items():
        if args.only and name not in args.only:
            continue
        try:
            fn()
        except Exception as e:
            print(f"  [fail] {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
