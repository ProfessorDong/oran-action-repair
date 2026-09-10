"""
Journal edition -- multi-cell clusters and coupled predicates.

The single-cell predicate set is component-separable, which is what makes the
repair operator a one-pass coordinate projection.  Cluster-level policy is not:
a total-radiated-power budget

    psi(o, a^1..a^C) = 1[ sum_i g(a^i) > B ]

couples the cells, so a shield that independently enforces the per-cell set Phi
at every cell is UNSOUND for psi -- each cell is individually legal while the
cluster is over budget.  This module implements and measures three schemes:

  per_cell   -- independent per-cell shielding (unsound for psi);
  static     -- a fixed per-cell share B/C (sound, but wastes budget on cells
                that do not need it);
  sequential -- Algorithm 3: budgeted greedy ascent.  Every cell starts at its
                minimum-power admissible action and the controller grants the
                upgrade with the best predicted benefit per unit of radiated
                power that still fits the budget.  Sound for any monotone
                coupled predicate (Theorem 2 in the paper).

The cluster also has a genuine power externality: a cell's SINR is degraded by
its neighbours' transmit power, so uncapped power racing is self-defeating.
The budget therefore does not only constrain -- it can improve throughput.

Author: Liang Dong.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from core import (
    Action, LatentState, Observation, Twin, TwinParams,
    SCENARIOS, SCENARIO_ORDER, SAFE_FALLBACK, PREDICATE_SETS, PHI_BASE,
    step_environment, observe, realised_pdr, update_thermal,
    update_priority_ewma, hazards, admissible, P_NOMINAL,
)
from shield import repair
from adversary import PLANNERS
from experiments import load_confusion

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------------------
# Coupled predicate
# ---------------------------------------------------------------------------

def g_power(a: Action) -> float:
    """Radiated power of one cell in linear mW -- the coupling functional."""
    return float(10.0 ** (a.power_dbm / 10.0))


def cluster_budget(n_cells: int, per_cell_dbm: float = 26.0) -> float:
    """Total radiated-power budget B for the cluster."""
    return n_cells * 10.0 ** (per_cell_dbm / 10.0)


def psi_violated(actions: list[Action], budget: float) -> bool:
    return sum(g_power(a) for a in actions) > budget + 1e-9


# ---------------------------------------------------------------------------
# Admission schemes
# ---------------------------------------------------------------------------

def _best_admissible(o, pool, twin, phi):
    cand = [a if admissible(o, a, phi) else repair(o, a, phi)[0] for a in pool]
    cand.append(SAFE_FALLBACK.copy())
    return cand


def admit_per_cell(obs, pools, twin, phi, budget):
    """Independent per-cell shielding.  Sound for Phi, unsound for psi."""
    out = []
    for o, pool in zip(obs, pools):
        cand = _best_admissible(o, pool, twin, phi)
        out.append(max(cand, key=lambda a: twin(o, a)[0]))
    return out


def admit_static_share(obs, pools, twin, phi, budget):
    """Fixed per-cell share of the budget.  Sound, but cannot move headroom
    from an idle cell to a congested one."""
    share = budget / len(obs)
    out = []
    for o, pool in zip(obs, pools):
        cand = [a for a in _best_admissible(o, pool, twin, phi)
                if g_power(a) <= share]
        if not cand:
            cand = [SAFE_FALLBACK.copy()]
        out.append(max(cand, key=lambda a: twin(o, a)[0]))
    return out


def admit_sequential(obs, pools, twin, phi, budget):
    """Algorithm 3: budgeted greedy ascent.

    Every cell starts at its minimum-power admissible action, which is jointly
    feasible by assumption (A4).  The cluster controller then repeatedly grants
    the single power upgrade with the largest predicted benefit per unit of
    radiated power that still fits the remaining budget.  The running total is
    an invariant of the loop, so the joint action satisfies the coupled
    predicate at every iteration and in particular on exit (Theorem 3), while
    each per-cell action stays Phi-admissible because upgrades are drawn only
    from the per-cell admissible sets.
    """
    n = len(obs)
    cand_sets = []
    for o, pool in zip(obs, pools):
        cand = _best_admissible(o, pool, twin, phi)
        # Keep, for each distinct power level, only the best-scoring action.
        best_at: dict[float, tuple[float, Action]] = {}
        for a in cand:
            key = round(a.power_dbm, 3)
            r = twin(o, a)[0]
            if key not in best_at or r > best_at[key][0]:
                best_at[key] = (r, a)
        cand_sets.append(sorted(((g_power(a), r, a) for r, a in best_at.values()),
                                key=lambda t: t[0]))

    idx = [0] * n                        # each cell at its cheapest action
    spent = sum(cand_sets[i][0][0] for i in range(n))

    while True:
        best_ratio, best_move = 0.0, None
        for i in range(n):
            g_i, r_i, _ = cand_sets[i][idx[i]]
            for j in range(idx[i] + 1, len(cand_sets[i])):
                g_j, r_j, _ = cand_sets[i][j]
                dg, dr = g_j - g_i, r_j - r_i
                if dg <= 0 or dr <= 0:
                    continue
                if spent - g_i + g_j > budget + 1e-9:
                    continue
                ratio = dr / dg
                if ratio > best_ratio:
                    best_ratio, best_move = ratio, (i, j)
        if best_move is None:
            break
        i, j = best_move
        spent += cand_sets[i][j][0] - cand_sets[i][idx[i]][0]
        idx[i] = j

    return [cand_sets[i][idx[i]][2] for i in range(n)]


SCHEMES = {"per_cell": admit_per_cell, "static_share": admit_static_share,
           "sequential": admit_sequential}
SCHEME_LABEL = {"per_cell": "Per-cell shield", "static_share": "Static share",
                "sequential": "Sequential (Alg.~3)"}


# ---------------------------------------------------------------------------
# Cluster episode
# ---------------------------------------------------------------------------

@dataclass
class ClusterSummary:
    scenario: str
    scheme: str
    n_cells: int
    seed: int
    cycles: int
    mean_pdr: float
    psi_violation_rate: float
    phi_violation_rate: float
    hazard_rate: float
    mean_power_dbm: float
    budget_use: float


def run_cluster(scenario: str, scheme: str, seed: int, n_cells: int = 7,
                cycles: int = 300, k: int = 5, planner_kind: str = "naive",
                predicate_set: str = "phi", confusion: dict | None = None,
                coupling: float = 0.35) -> ClusterSummary:
    rng = np.random.default_rng(seed)
    cfg = SCENARIOS[scenario]
    phi = PREDICATE_SETS[predicate_set]
    budget = cluster_budget(n_cells)
    twin = Twin(TwinParams(), seed=seed, confusion=confusion)
    planners = [PLANNERS[planner_kind](np.random.default_rng(seed * 100 + i), k=k)
                for i in range(n_cells)]
    # Heterogeneous cluster: cells differ in nominal link budget and in how
    # strongly the scenario's impairment reaches them, which is what makes
    # budget reallocation worth anything.
    states = [LatentState(rng=np.random.default_rng(seed * 1000 + i),
                          base_sinr_db=10.5 + 1.1 * (i % 5))
              for i in range(n_cells)]
    exposure = [(1.0, 0.45, 0.0)[i % 3] for i in range(n_cells)]
    prev = [Action() for _ in range(n_cells)]

    pdrs, powers, use = [], [], []
    n_psi = n_phi = n_haz = 0

    for c in range(cycles):
        obs = []
        for i, st in enumerate(states):
            st.cycle = c
            step_environment(st, cfg)
            st.interference_strength *= exposure[i]
            if exposure[i] == 0.0:
                st.interference_class = "none"
            obs.append(observe(st, prev[i], cfg, confusion=confusion))
        twin.observe_label(obs[0].interference_class)

        pools = [p.propose(o) for p, o in zip(planners, obs)]
        acts = SCHEMES[scheme](obs, pools, twin, phi, budget)

        total_g = sum(g_power(a) for a in acts)
        if total_g > budget + 1e-9:
            n_psi += 1
        if any(not admissible(o, a, phi) for o, a in zip(obs, acts)):
            n_phi += 1
        use.append(total_g / budget)

        # Inter-cell power externality: a cell's SINR is degraded by the mean
        # transmit power of its neighbours.
        for i, (st, a) in enumerate(zip(states, acts)):
            nb = [acts[j].power_dbm for j in range(n_cells) if j != i]
            ext = coupling * max(0.0, float(np.mean(nb)) - P_NOMINAL)
            pdr = realised_pdr(st, a, interf_ext_db=ext)
            if hazards(st, a):
                n_haz += 1
            update_thermal(st, a)
            update_priority_ewma(st, pdr, a)
            pdrs.append(pdr)
            powers.append(a.power_dbm)
        prev = acts

    return ClusterSummary(
        scenario=scenario, scheme=scheme, n_cells=n_cells, seed=seed,
        cycles=cycles, mean_pdr=float(np.mean(pdrs)),
        psi_violation_rate=n_psi / cycles,
        phi_violation_rate=n_phi / cycles,
        hazard_rate=n_haz / (cycles * n_cells),
        mean_power_dbm=float(np.mean(powers)),
        budget_use=float(np.mean(use)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--cells", type=int, nargs="+", default=[3, 5, 7, 12, 19])
    args = ap.parse_args()
    conf = load_confusion("radioml")

    rows = []
    for sc in SCENARIO_ORDER:
        for scheme in SCHEMES:
            for sd in range(args.seeds):
                rows.append(asdict(run_cluster(sc, scheme, sd, n_cells=7,
                                               cycles=args.cycles,
                                               confusion=conf)))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "multicell.csv", index=False)
    print(df.groupby("scheme")[["mean_pdr", "psi_violation_rate",
                                "hazard_rate", "budget_use"]].mean()
          .to_string())

    rows = []
    for n in args.cells:
        for scheme in SCHEMES:
            for sd in range(args.seeds):
                rows.append(asdict(run_cluster("compound", scheme, sd, n_cells=n,
                                               cycles=args.cycles,
                                               confusion=conf)))
    df2 = pd.DataFrame(rows)
    df2.to_csv(RESULTS / "multicell_scale.csv", index=False)
    print("\n" + df2.groupby(["n_cells", "scheme"])[
        ["mean_pdr", "psi_violation_rate"]].mean().to_string())
    print(f"\nWrote multicell*.csv to {RESULTS}")


if __name__ == "__main__":
    main()
