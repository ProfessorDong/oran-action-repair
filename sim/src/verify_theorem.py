"""
Numerical verification of Theorem 1 (Bounded regret) for SHIELD-RIC.

For each cycle of a shield_ric run we compute:
    1.  The realised reward of the chosen action a*       (true reward).
    2.  The twin's predicted reward of a*                 (-> twin error xi).
    3.  An ORACLE best safe action pi^*(o_t) found by exhaustive search
        over a fixed pool of safe templates                (-> coverage).
    4.  Whether pi^*(o_t) was a planner candidate          (-> cov_K(o_t)).
    5.  Per-cycle regret r(o_t,pi^*) - r(o_t,a*).

We aggregate over 10 seeds x 6 scenarios x 300 cycles and report, per
scenario:

    xi_max          = max_{cycles, seeds} |hat r - r|       (Assumption A2)
    xi_mean         = mean over cycles |hat r - r|
    cov_K_mean      = mean over cycles of 1[pi^* in A^cand]
    G_measured      = mean cumulative regret
    bound_twin      = 2*xi*T  (Theorem 1 twin term)
    bound_coverage  = r_max * Sum_t (1 - cov_K(o_t))  (Theorem 1 coverage term)
    bound_total     = bound_twin + bound_coverage

We then plot per-scenario measured regret vs. the bound and dump a CSV.

Author: Liang Dong, MILCOM 2026 Paper 1.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shield_ric as sr
from shield_ric import (
    Action, Observation, NetworkState, observe, twin_predict,
    is_violation, SCENARIOS, LLMPlannerStub, ShieldConfig,
    shield_admit_set,
)

HERE = Path(__file__).resolve().parent
SIM  = HERE.parent
RESULTS = SIM / "results"
FIGS    = SIM / "figs"
RESULTS.mkdir(exist_ok=True, parents=True)
FIGS.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------------------
# Oracle: enumerate "natural" safe action templates and pick the best on truth
# ---------------------------------------------------------------------------

def safe_action_pool() -> list[Action]:
    """Enumerate a fixed set of natural safe action templates.

    Every template satisfies phi_rf (no dsss, P<=26), phi_id (no mesh_b),
    phi_qos (slice=balanced never demotes priority), phi_bh (workload=local
    so it does not violate when backhaul partitions).  Cardinality ~120.
    """
    waveforms = ["ofdm_n0", "ofdm_n1", "ofdm_n2"]
    mod_codes = ["qpsk_1_3", "qpsk_1_2", "16qam_1_2", "16qam_3_4", "64qam_3_4"]
    beams     = ["omni", "sector_a", "narrow"]
    routes    = ["primary", "secondary", "mesh_a"]
    pool = []
    for w, m, b, r in product(waveforms, mod_codes, beams, routes):
        pool.append(Action(
            waveform=w, mod_code=m, beam=b, route=r,
            slice_="balanced", workload="local", power_dbm=23.0,
        ))
    return pool


SAFE_POOL = safe_action_pool()


def oracle_best_action(net: NetworkState, o: Observation
                       ) -> tuple[Action, float]:
    """Exhaustively score the safe pool with the THEOREM's reward, namely
    realised priority-flow PDR in [0,1].  By construction no template in
    SAFE_POOL violates, so the violation-cost term is identically zero.
    Returns (best safe action, best PDR)."""
    best_a = SAFE_POOL[0]
    best_r = -1.0
    for a in SAFE_POOL:
        pdr = net.realised_pdr(a)
        if pdr > best_r:
            best_r = pdr
            best_a = a
    return best_a, float(best_r)


def action_sig(a: Action) -> str:
    return f"{a.waveform}|{a.mod_code}|{a.beam}|{a.route}|{a.slice_}|{a.workload}|{a.power_dbm:.1f}"


# ---------------------------------------------------------------------------
# Per-cycle theorem-instrumented run (shield_ric controller only)
# ---------------------------------------------------------------------------

@dataclass
class CycleLog:
    cycle: int
    scenario: str
    seed: int
    twin_r_hat: float
    true_r_chosen: float
    oracle_r: float
    oracle_in_candidates: int       # 0/1
    regret: float                   # max(0, oracle_r - true_r_chosen)
    twin_abs_err: float             # |hat r - true r| at the chosen action


def verify_run(scenario: str, seed: int, cycles: int = 300,
               k: int = 5, poison_p: float = 0.3,
               perception_confusion: dict | None = None
               ) -> list[CycleLog]:
    rng = np.random.default_rng(seed)
    net = NetworkState(rng=rng)
    scenario_cfg = SCENARIOS[scenario]
    planner = LLMPlannerStub(rng=rng, k=k, poison_p=poison_p)
    shield_cfg = ShieldConfig()
    sr.PERCEPTION_CONFUSION = perception_confusion

    last_action = Action()
    logs: list[CycleLog] = []
    fallback = Action()

    for c in range(cycles):
        net.cycle = c
        net.step_environment(scenario_cfg)
        o = observe(net, last_action, scenario_cfg)

        cands = planner.propose(o, scenario_cfg)
        admitted = shield_admit_set(o, cands, shield_cfg)
        if admitted:
            chosen_a = max(admitted, key=lambda t: t[1][0])[0]
        else:
            chosen_a = fallback

        # Theorem-aligned reward: priority-flow PDR in [0,1].  Lemma 1
        # ensures chosen_a is admissible so the violation cost is zero.
        true_r = net.realised_pdr(chosen_a)

        # Twin's predicted PDR.  twin_predict now returns the PDR
        # estimate directly (matching the controller's argmax target).
        hat_r, _, _ = twin_predict(o, chosen_a)

        # Oracle best safe action over the natural pool
        oracle_a, oracle_r = oracle_best_action(net, o)

        # Was the oracle action in the candidate set?
        cand_sigs = {action_sig(a) for a in cands}
        in_cands = int(action_sig(oracle_a) in cand_sigs)

        logs.append(CycleLog(
            cycle=c, scenario=scenario, seed=seed,
            twin_r_hat=float(hat_r),
            true_r_chosen=float(true_r),
            oracle_r=float(oracle_r),
            oracle_in_candidates=int(in_cands),
            regret=float(max(0.0, oracle_r - true_r)),
            twin_abs_err=float(abs(hat_r - true_r)),
        ))
        last_action = chosen_a

    return logs


# ---------------------------------------------------------------------------
# Aggregate over seeds and scenarios
# ---------------------------------------------------------------------------

@dataclass
class ScenarioAgg:
    scenario: str
    cycles: int
    xi_max: float
    xi_mean: float
    cov_K_mean: float
    G_measured_per_cycle: float
    bound_twin_per_cycle: float
    bound_cov_per_cycle: float
    bound_total_per_cycle: float


def aggregate(rows: list[CycleLog], r_max: float = 1.0) -> list[ScenarioAgg]:
    df = pd.DataFrame([r.__dict__ for r in rows])
    out = []
    for sc, sub in df.groupby("scenario"):
        T = sub["cycle"].nunique()
        seeds = sub["seed"].nunique()
        xi_max = sub["twin_abs_err"].max()
        xi_mean = sub["twin_abs_err"].mean()
        cov_K_mean = sub["oracle_in_candidates"].mean()
        # Measured regret summed per seed then averaged
        per_seed_regret = sub.groupby("seed")["regret"].sum().mean()
        G_per_cycle = per_seed_regret / T
        bound_twin_per_cycle = 2.0 * xi_max
        bound_cov_per_cycle = r_max * (1.0 - cov_K_mean)
        out.append(ScenarioAgg(
            scenario=str(sc),
            cycles=int(T),
            xi_max=float(xi_max),
            xi_mean=float(xi_mean),
            cov_K_mean=float(cov_K_mean),
            G_measured_per_cycle=float(G_per_cycle),
            bound_twin_per_cycle=float(bound_twin_per_cycle),
            bound_cov_per_cycle=float(bound_cov_per_cycle),
            bound_total_per_cycle=float(bound_twin_per_cycle + bound_cov_per_cycle),
        ))
    return out


# ---------------------------------------------------------------------------
# Figure: measured regret vs. theoretical bound
# ---------------------------------------------------------------------------

SCENARIO_ORDER = ["narrowband", "wideband", "bursty",
                  "backhaul", "compromised_xapp", "compound"]
SCENARIO_LABEL = {
    "narrowband": "Narrowband",  "wideband": "Wideband",
    "bursty": "Bursty",          "backhaul": "Backhaul",
    "compromised_xapp": "Comp. xApp",  "compound": "Compound",
}


def make_figure(agg: list[ScenarioAgg], out_name: str = "fig_theorem_verify"):
    df = pd.DataFrame([a.__dict__ for a in agg])
    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER)
    df = df.sort_values("scenario")

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    x = np.arange(len(df))
    width = 0.32
    ax.bar(x - width/2, df["bound_twin_per_cycle"], width,
           label=r"Bound term $2\xi$ (twin)",
           color="#56B4E9", edgecolor="#22526a", linewidth=0.8)
    ax.bar(x - width/2, df["bound_cov_per_cycle"], width,
           bottom=df["bound_twin_per_cycle"],
           label=r"Bound term $r_{\max}(1-\mathrm{cov}_K)$ (coverage)",
           color="#E69F00", edgecolor="#8a5e00", linewidth=0.8)
    ax.bar(x + width/2, df["G_measured_per_cycle"], width,
           label="Measured regret",
           color="#0072B2", edgecolor="#003255", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABEL[str(s)] for s in df["scenario"]],
                       fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("Per-cycle regret", fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=10, ncol=1)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGS / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{out_name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--perception", choices=["perfect", "radioml"], default="radioml")
    ap.add_argument("--out-csv", default=str(RESULTS / "theorem_verify.csv"))
    args = ap.parse_args()

    conf = None
    if args.perception == "radioml":
        p = RESULTS / "perception_radioml.json"
        if p.exists():
            conf = json.loads(p.read_text())["perception_confusion"]

    all_rows: list[CycleLog] = []
    for sc in SCENARIO_ORDER:
        for s in args.seeds:
            rows = verify_run(sc, s, cycles=args.cycles,
                              perception_confusion=conf)
            all_rows.extend(rows)
            xi = max(r.twin_abs_err for r in rows)
            cov = sum(r.oracle_in_candidates for r in rows) / len(rows)
            reg = sum(r.regret for r in rows) / len(rows)
            print(f"[verify] {sc:>18s}  s={s}  xi_max={xi:.3f}  "
                  f"cov_K={cov:.3f}  regret/cycle={reg:.3f}")

    agg = aggregate(all_rows)
    df = pd.DataFrame([a.__dict__ for a in agg])
    df.to_csv(args.out_csv, index=False)
    print(f"\nAggregate per scenario:\n{df.to_string(index=False)}")

    make_figure(agg)
    # also copy fig into paper directory
    src = FIGS / "fig_theorem_verify.pdf"
    dst = SIM.parent / "fig_theorem_verify.pdf"
    dst.write_bytes(src.read_bytes())
    print(f"\nWrote {args.out_csv} and {dst}")


if __name__ == "__main__":
    main()
