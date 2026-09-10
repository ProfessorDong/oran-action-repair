# Action repair and predicate-coverage analysis for Open RAN control

Provably sound agentic control for self-healing Open RAN: an untrusted planner
proposes cross-layer recovery actions and a deterministic shield decides what is
actuated, repairing inadmissible candidates rather than discarding them.

Accompanies a manuscript submitted to *IEEE Transactions on Network and Service
Management*. Sole author: Liang Dong, Baylor University.

This repository contains the manuscript, the simulator, and everything needed
to regenerate every number, table, and figure in the paper. No value quoted in
the manuscript is typed by hand: `sim/journal/make_numbers.py` emits
`numbers.tex`, `tab_headline.tex`, and `tab_loo.tex` from the result CSVs, and
the paper reads only those.

## What is and is not in this repository

Included: the manuscript source and PDF, the simulator used for every result,
and the aggregated result files the numbers are generated from.

Not included, and to be fetched separately:

| Item | Where |
|---|---|
| Qwen2.5-1.5B-Instruct weights | `scripts/download_qwen.sh` (Hugging Face) |
| DeepSig RadioML 2016.10A | `scripts/download_radioml.sh` |
| ColO-RAN dataset (7.6 GB) | github.com/wineslab/colosseum-oran-coloran-dataset |

Also not included: the source PDFs of the reviewed literature, which are
copyrighted and cannot be redistributed.

An earlier simulator is retained locally but deliberately not published: it
contains a circular evaluation, in which the controller was scored against the
same predicate set the shield enforced, that this work exists to correct.

## Layout

```
main.tex                 manuscript (IEEEtran, journal class)
references.bib           bibliography
numbers.tex              GENERATED -- every numeric claim in the paper
tab_headline.tex         GENERATED -- Table III (headline results)
tab_loo.tex              GENERATED -- Table IV (predicate ablation)
fig_*.pdf                GENERATED -- figures copied from sim/journal/figs/

sim/journal/             journal-edition simulator (this paper)
  core.py                network model, latent hazard oracle H, predicate sets,
                         perception-aware digital twin
  shield.py              shield, repair operator, all eight controllers
  adversary.py           benign / overt / adaptive (Kerckhoffs) planners
  conditional.py         estimator of r_bar(o,a) = E[reward | o, a]
  experiments.py         main sweeps E1-E6
  theory.py              Theorem 1 verification and its sweeps
  calibration.py         perception-aware twin calibration study
  multicell.py           cluster-coupled predicates, Algorithm 3
  llm_planner.py         local LLM planner + prompt-injection experiment
  repair_stats.py        what the repair operator actually edits
  stats_j.py             bootstrap CIs, Wilcoxon + Holm, Cliff's delta
  timing.py              latency microbenchmark for the actuation path
  figures.py             all figures
  make_numbers.py        CSV -> LaTeX macros, with consistency checks

sim/src/                 earlier conference-draft simulator, kept for reference
sim/data/                RadioML 2016.10A (not redistributed here)
sim/models/              local LLM weights (not redistributed here)
```

## Reproducing

```bash
cd sim/journal

# 1. main sweeps: 8 controllers x 6 scenarios x 30 seeds x 300 cycles,
#    over three adversaries and two predicate sets   (~35 min, CPU)
python3 experiments.py --seeds 30 --cycles 300

# 2. Theorem 1: per-scenario bound, twin-mismatch sweep, candidate-budget
#    sweep, repair-vs-filter coverage                 (~6 min, CPU)
python3 theory.py --seeds 10 --cycles 300

# 3. perception-aware twin calibration                (~3 min, CPU)
python3 calibration.py --seeds 10 --cycles 300

# 4. cluster-coupled policy, C = 3..19                (~10 min, CPU)
python3 multicell.py --seeds 15 --cycles 300

# 5. repair-edit statistics                           (~5 min, CPU)
python3 repair_stats.py --seeds 30 --cycles 300

# 6. statistics: bootstrap CIs, paired tests          (~1 min)
python3 stats_j.py

# 7. actuation-path latency microbenchmark            (~1 min)
python3 timing.py

# 8. local LLM planner + prompt injection             (~2 h, one GPU)
python3 llm_planner.py --cycles 60 --seeds 0 1 2 \
        --scenarios narrowband wideband compound

# 9. figures, LaTeX macros, consistency checks, and the PDF
cd ../.. && ./build.sh
```

`make_numbers.py` prints a block of consistency checks at the end. Every one
must read `OK`; they assert the qualitative claims the prose makes (that the
repaired shield has the highest mean PDR, that the bound is non-vacuous, that
the shielded controllers record exactly zero policy violations, and so on).

## What is measured, and what it means

The simulator reports two distinct safety metrics, and the distinction is the
paper's main point.

* `violation_rate_phi` is the fraction of cycles on which the **actuated**
  action violated the **deployed** predicate set `Phi`. For any correctly
  implemented shield this is identically zero. It is a machine-checkable
  witness for Lemma 1 and nothing more; scoring a shield against its own
  predicates cannot produce any other answer.

* `hazard_rate` is the fraction of cycles on which the actuated action
  triggered the **latent hazard oracle** `H`, which reads the latent state and
  is strictly richer than `Phi`: a regulatory power cap that tightens while a
  protected incumbent transmits, a relay compromised after admission, a power
  amplifier above its thermal limit, sustained priority starvation. No
  controller can see `H`. This is the number that matters operationally.

Against the adaptive (Kerckhoffs) adversary, which knows the predicate schema
and therefore proposes only `Phi`-admissible actions, the first column is exactly
zero while the second is not. That gap is the paper's central result.

## Data and models

* Interference perception is trained on DeepSig **RadioML 2016.10A**, which
  must be downloaded separately into `sim/data/`. The trained classifier's
  validation confusion matrix is checked in at
  `sim/results/perception_radioml.json`, so every experiment except retraining
  the classifier runs without the raw dataset.
* The LLM experiment uses a local **Qwen2.5-1.5B-Instruct** checkout in
  `sim/models/`. Weights are not redistributed here. Nothing leaves the
  machine at any point.

## Requirements

Python 3.12 with `numpy`, `pandas`, `scipy`, `matplotlib`; `torch` and
`transformers` for the classifier and the LLM planner; TeX Live with
`IEEEtran`, `algorithmicx`, `subfig`, `booktabs`.
