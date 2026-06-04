# Multi-Task Extension (RadarMTL) — Design

**Date:** 2026-06-04
**Goal:** Extend the project's own 1D CNN into a multi-task model that solves the
full RadChar task — signal-type classification **plus** estimation of the four
pulse parameters — so it can be benchmarked against the *entire* RadChar Table 1,
and so we can measure what multi-task learning does to classification accuracy.

---

## Context

The project currently classifies radar **signal type** only (`RadarCNN`,
notebooks 04–05) and benchmarks against the classification-accuracy column of the
RadChar paper (arXiv:2306.13105). The paper's models are **multi-task**: one
shared backbone feeding 5 heads — 1 classification (signal type) + 4 regression
(number of pulses, pulse width, PRI, time delay) — which together form the Pulse
Descriptor Word used in radar signal characterisation.

This extension adds an equivalent multi-task model built on *our own* 1D CNN
backbone, enabling a full Table-1 comparison and a single-task-vs-multi-task
study.

## Goals

- **A — Full benchmark.** Compare our model against the paper's complete Table 1
  (regression MAE + classification accuracy), not just the classification column.
- **B — Learning experiment.** Measure whether adding the 4 regression tasks
  helps or hurts classification accuracy, by comparing single-task `RadarCNN`
  against multi-task `RadarMTL` under an identical training recipe.

## Decisions (agreed)

| Decision | Choice |
|----------|--------|
| Model organisation | Keep `RadarCNN` untouched; **add** a new `RadarMTL` class reusing the same backbone design + 5 heads |
| Notebook placement | **New** `06_multitask.ipynb`; notebooks 01–05 unchanged |
| Single-task baseline | Reuse the existing `RadarCNN`, re-run at the matched recipe (no separate single-task build) |
| Regression recipe | Match the paper's recipe (label-norm to [0,1], L1 loss, weights 0.1/0.225×4) **with task weights exposed as a config constant** for experimentation |

## Architecture

### `src/radar/model.py` — add `RadarMTL`

Reuses the same convolutional backbone design as `RadarCNN` (3 conv blocks,
2 → 32 → 64 → 128 channels, adaptive average pool → 128-dim shared features),
then **5 parallel heads** off the shared feature vector:

- 1 **classification head** → 5 logits (signal type)
- 4 **regression heads** → 1 scalar each (number of pulses, pulse width, PRI,
  time delay)

`forward()` returns the class logits plus the four regression scalars (e.g. a
dict or tuple with a fixed, documented order). `RadarCNN` is left exactly as is
to serve as the single-task baseline.

### `src/radar/data.py` — add target helpers

`radar.data` owns "raw file → model-ready tensors + the scaling needed to report
results." Two additions:

- **`regression_targets(labels)`** → `(N, 4)` float array of the four parameters
  in **real units** (µs for pulse width / PRI / time delay; raw count for number
  of pulses). Single source of truth for target order, so the notebook and the
  metrics never disagree. (Time fields are stored in seconds and converted to µs
  here.)
- **A small normaliser** — fits **min/max on training rows only**, transforms all
  splits to [0, 1] for training, and provides `inverse_transform` to convert
  predictions back to real units for MAE reporting. Fit-on-train-only preserves
  the project's no-leakage discipline.

The classification target stays `labels["signal_type"]`.

## Training — `notebooks/06_multitask.ipynb`

Mirrors notebook 04's structure for familiarity.

- **Setup:** same recipe constants as the nb04 re-run so single-vs-multi is fair —
  `SEED=42`, batch size 64, Adam, `lr=5e-4`, `EPOCHS=100`, 70/15/15 split via
  `make_split`, same input handling. Reuses the saved split indices so the test
  set is identical across models.
- **Targets:** classification label + the 4 normalised regression targets from
  the `radar.data` helpers.
- **Loss:** cross-entropy (classification) + L1 (each regression), combined as a
  weighted sum. Weights exposed as a named constant:

  ```python
  TASK_WEIGHTS = {"type": 0.1, "n_pulses": 0.225, "pw": 0.225, "pri": 0.225, "td": 0.225}
  ```

  Reproduces the paper by default; the single knob to change for the goal-B
  weight experiment (cf. paper §3.3 ablation).
- **Training loop:** same shape as nb04 (train/val, early stopping on total val
  loss, restore best weights), additionally tracking the 5 sub-losses, val
  classification accuracy, and val regression MAEs so each task's learning is
  visible.
- **Saves:** `results/radar_mtl.pth` and the fitted normaliser stats (needed to
  de-normalise predictions during evaluation) into `results/`.

## Evaluation & outputs (in nb06, after training)

- **Reproduce Table 1** — for each task at **−10 / 0 / +10 dB**: classification
  accuracy and regression MAE in real units (de-normalised). Printed with our row
  alongside the paper's CNN1D / CNN2D / IQST-S / IQST-L rows.
- **Reproduce Fig. 3** — 5 subplots across −20…+20 dB: 4 MAE curves + 1
  classification-accuracy curve. Saved to `results/`.
- **Goal-B comparison** — overlay single-task `RadarCNN` accuracy (from the nb04
  re-run) against multi-task `RadarMTL` accuracy vs SNR on one plot, answering
  "does multi-tasking help or hurt classification?" Saved to `results/`.
- **README update** — fill the full Table 1 (all 5 tasks) and add the
  single-vs-multi finding, replacing the classification-only placeholder.

## Files touched

| File | Change |
|------|--------|
| `src/radar/model.py` | add `RadarMTL` (shared backbone + 5 heads); `RadarCNN` untouched |
| `src/radar/data.py` | add `regression_targets(labels)` + fit-on-train normaliser |
| `notebooks/06_multitask.ipynb` | new — train `RadarMTL`, reproduce Table 1 + Fig. 3, single-vs-multi plot |
| `notebooks/04_train_cnn.ipynb` | (prerequisite, separate) re-run at matched recipe to refresh the single-task baseline |
| `results/` | new `radar_mtl.pth`, normaliser stats, multitask plots |
| `README.md` | update full Table 1 + single-vs-multi finding |

## Dependencies / sequencing

- Goal B requires the **nb04 single-task re-run at the matched recipe**
  (`lr=5e-4`, 100 epochs, same data/split). This is the benchmark-tuning change
  discussed previously, now a prerequisite for a fair single-vs-multi comparison.
- The GPU training runs (nb04 re-run and nb06) happen on Kaggle, outside this
  implementation work.

## Out of scope

- Changing `RadarCNN`'s architecture.
- Auto-balanced loss weighting (GradNorm, uncertainty weighting) — fixed
  configurable weights only.
- Replicating the paper's exact CNN1D / IQST backbones — we use our own backbone.
- The GPU training execution itself.
- A unit-test suite (consistent with the project's notebook-first approach;
  modules are verified by running them against the dataset).
