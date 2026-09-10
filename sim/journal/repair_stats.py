"""
Journal edition -- what the repair operator actually edits.

Counts, over a full sweep of scenarios and seeds, which action components the
repair operator rewrites and which predicate triggered the rewrite.  The paper
quotes this distribution rather than asserting it.

Author: Liang Dong.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from core import (Action, LatentState, Twin, TwinParams, SCENARIOS,
                  SCENARIO_ORDER, PREDICATE_SETS, SAFE_FALLBACK,
                  step_environment, observe, realised_pdr, update_thermal,
                  update_priority_ewma, admissible, violated_predicates)
from shield import repair
from adversary import PLANNERS
from experiments import load_confusion

RESULTS = Path(__file__).resolve().parent / "results"

COMP_TEX = {"power_dbm": "transmit power", "waveform": "waveform",
            "route": "route", "slice_": "slice", "workload": "placement",
            "mod_code": "MCS", "beam": "beam", "*": "global veto",
            "fallback": "fallback"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--planner", default="naive")
    ap.add_argument("--predicate-set", default="phi")
    args = ap.parse_args()

    conf = load_confusion("radioml")
    phi = PREDICATE_SETS[args.predicate_set]
    edits, trigs, n_cand, n_repaired = Counter(), Counter(), 0, 0
    # Action aliasing (Markgraf et al.): how many distinct actions survive in
    # the pool after repair, relative to the raw candidate set.  Projection
    # collapses several unsafe candidates onto the same safe action, which
    # shrinks the effective pool even though its cardinality is unchanged.
    raw_distinct = rep_distinct = n_cycles = 0

    for sc in SCENARIO_ORDER:
        cfg = SCENARIOS[sc]
        for sd in range(args.seeds):
            rng = np.random.default_rng(sd)
            st = LatentState(rng=rng)
            twin = Twin(TwinParams(), seed=sd, confusion=conf)
            planner = PLANNERS[args.planner](rng, k=5)
            a_prev = Action()
            for c in range(args.cycles):
                st.cycle = c
                step_environment(st, cfg)
                o = observe(st, a_prev, cfg, confusion=conf)
                twin.observe_label(o.interference_class)
                cands = planner.propose(o)
                pool = []
                for a in cands:
                    n_cand += 1
                    if admissible(o, a, phi):
                        pool.append(a)
                        continue
                    n_repaired += 1
                    for p in violated_predicates(o, a, phi):
                        trigs[p] += 1
                    a_r, ed = repair(o, a, phi)
                    for e in set(ed):
                        edits[e] += 1
                    pool.append(a_r)
                raw_distinct += len({a.sig() for a in cands})
                rep_distinct += len({a.sig() for a in pool})
                n_cycles += 1
                pool.append(SAFE_FALLBACK.copy())
                best = max(pool, key=lambda x: twin(o, x)[0])
                pdr = realised_pdr(st, best)
                update_thermal(st, best)
                update_priority_ewma(st, pdr, best)
                a_prev = best

    rows = [dict(kind="component", name=k, label=COMP_TEX.get(k, k),
                 count=v, share=v / max(n_repaired, 1))
            for k, v in edits.most_common()]
    rows += [dict(kind="predicate", name=k, label=k, count=v,
                  share=v / max(n_repaired, 1)) for k, v in trigs.most_common()]
    rows.append(dict(kind="totals", name="candidates", label="candidates",
                     count=n_cand, share=1.0))
    rows.append(dict(kind="totals", name="repaired", label="repaired",
                     count=n_repaired, share=n_repaired / max(n_cand, 1)))
    rows.append(dict(kind="aliasing", name="raw_distinct_per_cycle",
                     label="distinct candidates before repair",
                     count=raw_distinct, share=raw_distinct / max(n_cycles, 1)))
    rows.append(dict(kind="aliasing", name="rep_distinct_per_cycle",
                     label="distinct candidates after repair",
                     count=rep_distinct, share=rep_distinct / max(n_cycles, 1)))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "repair_stats.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {RESULTS/'repair_stats.csv'}")


if __name__ == "__main__":
    main()
