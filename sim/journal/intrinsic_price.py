"""The intrinsic price of the constraint, and the degeneracy of P(latent | o).

The repair-versus-Greedy-Twin difference mixes three things: the constraint
itself, the planner's candidate generation, and twin ranking error.  Only the
first is a property of Phi.  This module measures it directly as

    Delta_Phi(o) = max_{a in A} r_bar(o, a) - max_{a in A_Phi(o)} r_bar(o, a),

which needs neither a planner nor a twin.  It also reports the entropy of the
posterior P(latent | o), which is what makes pi_dagger coincide with the
clairvoyant optimum in this simulator.

Author: Liang Dong.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import core, conditional as cond, experiments as ex
from core import PHI_BASE, observe, step_environment, LatentState, SCENARIOS

RESULTS = Path(__file__).resolve().parent / "results"


def main(seeds: int = 10, cycles: int = 300) -> None:
    conf = ex.load_confusion("radioml")
    rows, ent_all = [], []
    for sc in core.SCENARIO_ORDER:
        model = cond.get(sc, conf)
        for w in model.weights.values():
            p = np.array(list(w.values()), float); p = p / p.sum()
            ent_all.append(float(-(p * np.log2(p + 1e-12)).sum()))
        cfg = SCENARIOS[sc]
        deltas = []
        for sd in range(seeds):
            st = LatentState(rng=np.random.default_rng(1000 + sd))
            a_prev = core.SAFE_FALLBACK.copy()
            for c in range(cycles):
                st.cycle = c
                step_environment(st, cfg)
                o = observe(st, a_prev, cfg, confusion=conf)
                ok = cond.obs_key(o)
                rvec = model._rbar_vector(ok)
                mask = cond._admissible_mask(
                    (o.queue_priority > core.THETA_Q,
                     o.queue_priority > core.THETA_M,
                     o.twin_uncertainty > core.U_BAR,
                     bool(o.backhaul_ok)), o, PHI_BASE)
                if mask.any():
                    deltas.append(float(rvec.max() - rvec[mask].max()))
        rows.append(dict(scenario=sc, delta_phi=float(np.mean(deltas)),
                         delta_phi_p95=float(np.quantile(deltas, 0.95)),
                         n=len(deltas)))
    out = {"per_scenario": rows,
           "mean_delta_phi": float(np.mean([r["delta_phi"] for r in rows])),
           "posterior_entropy_bits": float(np.mean(ent_all)),
           "posterior_entropy_max": float(np.max(ent_all))}
    (RESULTS / "intrinsic_price.json").write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"  {r['scenario']:17s} Delta_Phi = {r['delta_phi']:.4f}"
              f"  (p95 {r['delta_phi_p95']:.4f})")
    print(f"\nmean Delta_Phi = {out['mean_delta_phi']:.4f}")
    print(f"posterior entropy: mean {out['posterior_entropy_bits']:.3f} bits, "
          f"max {out['posterior_entropy_max']:.3f}")


if __name__ == "__main__":
    main()
