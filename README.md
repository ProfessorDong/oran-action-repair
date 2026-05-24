# SHIELD-RIC — Provably Safe Agentic Control of Self-Healing Open RAN Tactical Edge Networks

Code and aggregated results accompanying the IEEE MILCOM 2026 submission
*"SHIELD-RIC: Provably Safe Agentic Control of Self-Healing Open RAN Tactical
Edge Networks"* by Liang Dong (Baylor University).

> **SHIELD** stands for *Shielded Hypothesis-Intent, Evidenced LLM Decisions*.
> An LLM proposes cross-layer recovery actions; a deterministic safety shield
> filters every candidate against a predicate set $\Phi$; the controller
> issues only admitted actions and rolls back if a guard fails.  Soundness is
> proved (Lemma 1) and regret is bounded by twin error and planner coverage
> (Theorem 1).  Both are verified numerically.

This repository contains the discrete-event simulator, the perception
classifier, the verification scripts, the bootstrap-CI / Wilcoxon analysis,
the real-LLM driver, and the figure generators.  All aggregated results
cited in the paper are committed as small CSV/JSON files so reviewers can
re-derive every reported number without re-running the simulator.

---

## What is in this repo

```
sim/
  src/
    shield_ric.py        # discrete-event Open RAN simulator (cross-layer cPOMDP)
    perception.py        # RadioML 2016.10A loader + ResNet-1D classifier trainer
    run_experiments.py   # sweeps 3 perception conditions x 6 scenarios x 5 controllers x N seeds
    verify_theorem.py    # numerical verification of Theorem 1 (xi, cov_K, regret bound)
    stats.py             # bootstrap 95% CIs and Wilcoxon signed-rank test
    llm_planner.py       # real local LLM (Qwen2.5-1.5B-Instruct) driving the planner
    make_figures.py      # vector-PDF figure generation
  results/
    master.csv           # aggregated PDR / violation-rate per (controller, scenario, seed, condition)
    theorem_verify.csv   # xi_max, cov_K, measured regret, theoretical bound per scenario
    stats.json           # bootstrap CIs and Wilcoxon test output
    llm_planner.csv      # Qwen-driven planner results: PDR, violation rate, cov_K, latency
    perception_radioml.json  # classifier accuracy + row-normalised confusion matrix
  figs/                  # generated PDF/PNG figures used in the manuscript
scripts/
  download_radioml.sh    # fetch RadioML 2016.10A from Zenodo
  download_qwen.sh       # fetch Qwen2.5-1.5B-Instruct from ModelScope
LICENSE                  # MIT
README.md
```

The manuscript LaTeX source is **not** redistributed; only data, code,
and aggregated outputs are released here.

---

## Datasets and model weights

Two external assets are not committed and must be downloaded separately:

| Asset | Size | Source | Script |
|---|---|---|---|
| DeepSig RadioML 2016.10A | 213 MB tar.bz2 | [Zenodo 18397070](https://zenodo.org/records/18397070) | `scripts/download_radioml.sh` |
| Qwen2.5-1.5B-Instruct weights | 2.9 GB | [ModelScope mirror](https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct) | `scripts/download_qwen.sh` |

After running both scripts you should have:

```
sim/data/RML2016.10a_dict_optimized.pkl
sim/models/Qwen2.5-1.5B-Instruct/{config.json, model.safetensors, ...}
```

> The HuggingFace public endpoint rate-limits anonymous model downloads to a
> few KB/s, so `download_qwen.sh` uses the ModelScope mirror, which serves
> the Qwen weights at 10-30 MB/s.

---

## Software requirements

Tested on Ubuntu 24.10, Python 3.12, single NVIDIA RTX 4060 Laptop GPU (8 GB).

```
pip install numpy scipy pandas matplotlib torch
pip install transformers accelerate   # only for llm_planner.py
```

Anything able to run PyTorch 2.x with CUDA will work; CPU-only runs of the
simulator are also fine (just slower for `perception.py`).

---

## Reproducing every number in the paper

A single shell session that re-derives every value cited in the manuscript
(headline table, Theorem 1 verification, bootstrap CIs, real-LLM result):

```bash
cd sim
# 1. Train the RadioML interference classifier (~15 min on GPU)
python3 src/perception.py --source radioml \
        --radioml-pkl data/RML2016.10a_dict_optimized.pkl \
        --epochs 15 --out results/perception_radioml.json

# 2. Main sweep: 3 conditions x 6 scenarios x 5 controllers x 10 seeds
python3 src/run_experiments.py --seeds 0 1 2 3 4 5 6 7 8 9 --cycles 300

# 3. Theorem 1 numerical verification (xi, cov_K, regret vs. bound)
python3 src/verify_theorem.py --seeds 0 1 2 3 4 5 6 7 8 9 --cycles 300 --perception radioml

# 4. Bootstrap 95% CIs and Wilcoxon test
python3 src/stats.py

# 5. Real local-LLM driver (Qwen2.5-1.5B-Instruct, ~25 min for 480 cycles)
python3 src/llm_planner.py --cycles 80 --seeds 0 1 \
        --scenarios narrowband wideband compromised_xapp

# 6. Generate all figures
python3 src/make_figures.py
```

Total runtime is about **45 minutes** on a single 8 GB GPU.  If you only
want to regenerate figures or aggregate analysis from the committed CSVs,
skip steps 1, 2, 3, 5 and run steps 4 and 6 only (a few seconds).

### Mapping every paper claim to a CSV cell

| Manuscript claim | File | Cell |
|---|---|---|
| 0.000 violation rate, every scenario, RadioML perception | `master.csv` | `controller==shield_ric & condition==radioml` rows |
| 34.7-52.3 % unshielded-baseline violation range | `master.csv` | `controller in {rl, llm_only}` per-scenario means |
| 79.06 % perception accuracy | `perception_radioml.json` | `info.val_accuracy` |
| xi in [1.13, 1.48], cov_K in [0.0003, 0.008] | `theorem_verify.csv` | `xi_max`, `cov_K_mean` |
| Measured cumulative regret 0.07-0.23 per cycle | `theorem_verify.csv` | `G_measured_per_cycle` |
| Bootstrap CI half-width <= 0.010 | `stats.json` | `summary.<scenario>.shield_ric` |
| Wilcoxon p < 1e-4 | `stats.json` | `wilcoxon.pvalue` |
| 480 LLM cycles, 0.000 violations | `llm_planner.csv` | `violation_rate` column |
| -phi_rf admits 20.8-34.2 %, -phi_id 15.5-23.8 %, ... | reproduce with leave-one-out flag in `shield_ric.py`; see Section V ablation paragraph | |

---

## License

MIT, see [LICENSE](LICENSE).  The bundled RadioML 2016.10A dataset is
distributed under CC BY-NC-SA 4.0 by DeepSig and is **not** included in
this repository; please respect its license terms when downloading from
Zenodo.

## Citation

```
@inproceedings{dong_shield_ric_2026,
  author    = {Liang Dong},
  title     = {{SHIELD-RIC}: Provably Safe Agentic Control of Self-Healing
               {Open RAN} Tactical Edge Networks},
  booktitle = {Proc. IEEE Military Communications Conference (MILCOM)},
  year      = {2026}
}
```

## Contact

Liang Dong, Department of Electrical and Computer Engineering, Baylor
University, Waco, Texas 76798, USA.  `Liang_Dong@baylor.edu`.
