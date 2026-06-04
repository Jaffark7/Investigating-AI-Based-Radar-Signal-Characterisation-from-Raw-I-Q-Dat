# Codebase Cleanup & Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the working radar project into a clean, best-practice, GitHub-ready portfolio repo by extracting shared code into an installable `radar` package and polishing all notebooks.

**Architecture:** Reusable code (data loading, the CNN, the traditional classifier) moves into `src/radar/` as an installable package (`pip install -e .`). The five notebooks keep their narrative role but import from `radar` instead of duplicating code. A README and dependency files are added.

**Tech Stack:** Python, PyTorch, NumPy/SciPy, h5py, Matplotlib, Jupyter, setuptools.

**Verification note:** This is a notebook-first portfolio project; per the approved spec there is no pytest suite. Each extracted module is verified by importing it and running it against `data/RadChar-Tiny.h5` (the small, fast dataset) and checking output shapes/ranges. That functional check is this project's equivalent of a test.

---

## File map

| File | Status | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | create | Declares the `radar` package under `src/`, deps |
| `requirements.txt` | create | Pinned dependencies for reproducibility |
| `.gitignore` | modify | Add packaging/tooling artifacts |
| `src/radar/__init__.py` | create | Marks package |
| `src/radar/data.py` | create | `load_radchar`, `make_split` |
| `src/radar/model.py` | create | `RadarCNN` |
| `src/radar/traditional.py` | create | `spectral_feature`, `build_templates`, `classify` |
| `docs/project_plan.md` | move | from root `radar_ai_project_plan.md` |
| `notebooks/01..05` | modify | Import from `radar`, fix comments, dedupe |
| `README.md` | create | Portfolio pitch (drafted with user) |

---

## Task 1: Packaging scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "radar"
version = "0.1.0"
description = "Detecting and classifying radar signals from raw I/Q data: DSP baseline vs 1D CNN."
requires-python = ">=3.9"
dependencies = ["numpy", "scipy", "h5py", "torch", "matplotlib"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Capture installed versions and write `requirements.txt`**

Run this to see the versions actually installed, then paste them into the file:

```bash
python -c "import numpy, scipy, h5py, torch, matplotlib; print('numpy=='+numpy.__version__); print('scipy=='+scipy.__version__); print('h5py=='+h5py.__version__); print('torch=='+torch.__version__); print('matplotlib=='+matplotlib.__version__)"
```

Create `requirements.txt` with the printed pins, plus jupyter, e.g.:

```
numpy==<printed>
scipy==<printed>
h5py==<printed>
torch==<printed>
matplotlib==<printed>
jupyter
```

- [ ] **Step 3: Add packaging artifacts to `.gitignore`**

Append to `.gitignore`:

```
# Packaging / tooling
*.egg-info/
build/
dist/
.firecrawl/
```

- [ ] **Step 4: Install the package in editable mode**

Run: `pip install -e .`
Expected: ends with `Successfully installed radar-0.1.0`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt .gitignore
git commit -m "build: add installable radar package scaffold and deps"
```

---

## Task 2: Extract `data.py`

**Files:**
- Create: `src/radar/__init__.py`
- Create: `src/radar/data.py`

- [ ] **Step 1: Create `src/radar/__init__.py`**

```python
"""radar — shared code for the RadChar DSP-vs-CNN project."""
```

- [ ] **Step 2: Create `src/radar/data.py`**

```python
"""Loading and splitting the RadChar dataset."""

import h5py
import numpy as np


def load_radchar(path):
    """Load raw I/Q signals and labels from a RadChar .h5 file.

    Returns (iq, labels): iq is complex, shape (N, 512); labels is a
    structured array with fields such as 'signal_type' and
    'signal_to_noise_ratio'.
    """
    with h5py.File(path, "r") as f:
        iq = f["iq"][:]
        labels = f["labels"][:]
    return iq, labels


def make_split(n, seed=42, fractions=(0.70, 0.15, 0.15)):
    """Deterministic train/val/test index split.

    Uses the legacy global RNG so the split reproduces the original
    notebooks and the index files already saved in results/.
    Returns (train_idx, val_idx, test_idx).
    """
    train_frac, val_frac, _ = fractions
    np.random.seed(seed)
    perm = np.random.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    return (
        perm[:n_train],
        perm[n_train:n_train + n_val],
        perm[n_train + n_val:],
    )
```

- [ ] **Step 3: Verify against the Tiny dataset**

Run:

```bash
python -c "from radar.data import load_radchar, make_split; iq, labels = load_radchar('data/RadChar-Tiny.h5'); print('iq', iq.shape, iq.dtype); print('fields', labels.dtype.names); tr, va, te = make_split(len(iq)); print('split', len(tr), len(va), len(te)); assert len(tr)+len(va)+len(te)==len(iq)"
```

Expected: prints `iq (50000, 512) complex128`, the field names, `split 35000 7500 7500`, no assertion error.

- [ ] **Step 4: Commit**

```bash
git add src/radar/__init__.py src/radar/data.py
git commit -m "refactor: extract data loading and split into radar.data"
```

---

## Task 3: Extract `model.py`

**Files:**
- Create: `src/radar/model.py`

- [ ] **Step 1: Create `src/radar/model.py`**

```python
"""The 1D CNN that classifies radar signal type from raw I/Q."""

import torch.nn as nn


class RadarCNN(nn.Module):
    """Input (batch, 2, 512) -> output (batch, 5) class scores.

    Two input channels (I and Q), five radar signal-type classes.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=11, padding=5), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.5),       # regularisation: reduces overfitting
            nn.Linear(64, 5),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

- [ ] **Step 2: Verify a forward pass**

Run:

```bash
python -c "import torch; from radar.model import RadarCNN; m = RadarCNN(); out = m(torch.randn(8, 2, 512)); print(out.shape); assert tuple(out.shape) == (8, 5)"
```

Expected: prints `torch.Size([8, 5])`, no assertion error.

- [ ] **Step 3: Commit**

```bash
git add src/radar/model.py
git commit -m "refactor: extract RadarCNN into radar.model"
```

---

## Task 4: Extract `traditional.py`

**Files:**
- Create: `src/radar/traditional.py`

- [ ] **Step 1: Create `src/radar/traditional.py`**

```python
"""Traditional (non-neural) spectral template classifier.

Each modulation type has a distinct frequency fingerprint. We build one
mean spectrum per class from training data, then label each test sample by
its nearest template. This is the baseline the CNN is compared against.
"""

import numpy as np


def spectral_feature(x):
    """Normalised magnitude spectrum per row (time-shift invariant)."""
    s = np.abs(np.fft.fft(x, axis=-1))
    return s / (s.sum(axis=-1, keepdims=True) + 1e-12)


def build_templates(iq_train, y_train, n_classes=5):
    """Mean spectral fingerprint per class, fitted on training data only."""
    feats = spectral_feature(iq_train)
    return np.stack([feats[y_train == c].mean(0) for c in range(n_classes)])


def classify(iq_samples, templates):
    """Label each sample by its nearest template (smallest squared distance)."""
    feats = spectral_feature(iq_samples)
    d = ((feats[:, None, :] - templates[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)
```

- [ ] **Step 2: Verify against the Tiny dataset**

Run:

```bash
python -c "import numpy as np; from radar.data import load_radchar, make_split; from radar.traditional import build_templates, classify; iq, labels = load_radchar('data/RadChar-Tiny.h5'); y = labels['signal_type'].astype(np.int64); tr, va, te = make_split(len(iq)); T = build_templates(iq[tr], y[tr]); pred = classify(iq[te][:100], T); print('templates', T.shape); print('pred range', pred.min(), pred.max()); assert T.shape == (5, 512) and pred.min() >= 0 and pred.max() <= 4"
```

Expected: prints `templates (5, 512)` and a `pred range` within 0–4, no assertion error.

- [ ] **Step 3: Commit**

```bash
git add src/radar/traditional.py
git commit -m "refactor: extract spectral template classifier into radar.traditional"
```

---

## Task 5: Move the project plan into `docs/`

**Files:**
- Move: `radar_ai_project_plan.md` -> `docs/project_plan.md`

- [ ] **Step 1: Move the file with git**

```bash
git mv radar_ai_project_plan.md docs/project_plan.md
```

(If git reports the file is untracked, use a plain move: `mkdir -p docs && mv radar_ai_project_plan.md docs/project_plan.md`.)

- [ ] **Step 2: Verify**

Run: `ls docs/project_plan.md`
Expected: the path prints, file exists.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "docs: move project plan into docs/"
```

---

## Task 6: Clean notebook 01 (explore data)

**Files:**
- Modify: `notebooks/01_explore_data.ipynb`

- [ ] **Step 1: Update the imports cell**

Replace the imports/`DATA_PATH` cell with:

```python
import numpy as np
import h5py
from radar.data import load_radchar

DATA_PATH = "../data/RadChar-Small.h5"
```

(`h5py` stays because the next cell lists the file's internal structure directly.)

- [ ] **Step 2: Replace the load cell**

Replace the cell that opens the file and reads `iq`/`labels` (the one with the stale `# shape: (50000, 512) — 50k samples` comment) with:

```python
# Load the raw I/Q signals and their labels.
iq, labels = load_radchar(DATA_PATH)
print("IQ shape:", iq.shape)        # (500000, 512) complex
print("Labels shape:", labels.shape)
```

- [ ] **Step 3: Fix the obscure indexing cell**

In the "how many samples of each signal type" cell, replace
`types = labels[labels.dtype.names[1]]` with:

```python
types = labels["signal_type"]
```

- [ ] **Step 4: Verify the notebook executes**

Run: `jupyter nbconvert --to notebook --execute notebooks/01_explore_data.ipynb --output 01_explore_data.ipynb --ExecutePreprocessor.timeout=120`
Expected: completes with no error (note: requires `pip install -e .` from Task 1 so `radar` imports).

- [ ] **Step 5: Commit**

```bash
git add notebooks/01_explore_data.ipynb
git commit -m "refactor(nb01): use radar.load_radchar, fix stale comment and indexing"
```

---

## Task 7: Clean notebook 02 (plot signals)

**Files:**
- Modify: `notebooks/02_plot_signals.ipynb`

- [ ] **Step 1: Update imports + load cells**

Replace the imports/`DATA_PATH` cell with:

```python
import numpy as np
import matplotlib.pyplot as plt
from radar.data import load_radchar

DATA_PATH = "../data/RadChar-Small.h5"
```

Replace the data-load cell with:

```python
iq, labels = load_radchar(DATA_PATH)
```

- [ ] **Step 2: Remove the "pitch" analogy**

In the cell that splits the first sample, replace the I/Q comment lines:

```python
I = sample.real   # in-phase component
Q = sample.imag   # quadrature component
```

- [ ] **Step 3: Verify the notebook executes**

Run: `jupyter nbconvert --to notebook --execute notebooks/02_plot_signals.ipynb --output 02_plot_signals.ipynb --ExecutePreprocessor.timeout=120`
Expected: completes with no error.

- [ ] **Step 4: Commit**

```bash
git add notebooks/02_plot_signals.ipynb
git commit -m "refactor(nb02): use radar.load_radchar, drop pitch analogy"
```

---

## Task 8: Clean notebook 03 (traditional detector)

**Files:**
- Modify: `notebooks/03_traditional_detector.ipynb`

- [ ] **Step 1: Update imports + load cells**

Replace the imports/`DATA_PATH` cell with:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from radar.data import load_radchar

DATA_PATH = "../data/RadChar-Small.h5"
```

Replace the data-load cell with:

```python
iq, labels = load_radchar(DATA_PATH)
```

- [ ] **Step 2: Leave the matched-filter demo inline**

No change to the matched-filter/threshold cells — this is a one-off teaching
visual, not reused elsewhere, so it stays in the notebook.

- [ ] **Step 3: Verify the notebook executes**

Run: `jupyter nbconvert --to notebook --execute notebooks/03_traditional_detector.ipynb --output 03_traditional_detector.ipynb --ExecutePreprocessor.timeout=120`
Expected: completes with no error.

- [ ] **Step 4: Commit**

```bash
git add notebooks/03_traditional_detector.ipynb
git commit -m "refactor(nb03): use radar.load_radchar"
```

---

## Task 9: Clean notebook 04 (train CNN)

**Files:**
- Modify: `notebooks/04_train_cnn.ipynb`

- [ ] **Step 1: Update the imports cell**

Replace the top imports cell with:

```python
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

from radar.data import load_radchar, make_split
from radar.model import RadarCNN

DATA_PATH = "../data/RadChar-Small.h5"
SEED = 42
```

(Keep the existing `set_seed`/device cell as is.)

- [ ] **Step 2: Use `load_radchar` for the data-load cell**

Replace the `with h5py.File(...)` load cell with:

```python
iq, labels = load_radchar(DATA_PATH)
```

- [ ] **Step 3: Delete the duplicated `RadarCNN` class cell**

Remove the entire cell that defines `class RadarCNN(nn.Module): ...`.
Replace it with a one-line instantiation cell:

```python
model = RadarCNN().to(device)
print(model)
```

- [ ] **Step 4: Use `make_split` in the split cell**

In the split cell, replace the manual `perm = np.random.permutation(N)` block and
the `n_train`/`n_val`/slicing lines with:

```python
train_idx, val_idx, test_idx = make_split(N, seed=SEED)
```

Keep the lines that save `train_idx`/`test_idx` to `results/` and build the
`Subset`s and `DataLoader`s.

- [ ] **Step 5: Verify imports resolve (no full training run here)**

Run: `jupyter nbconvert --to script --stdout notebooks/04_train_cnn.ipynb > /dev/null`
Expected: completes with no error (full execution/training happens later on GPU).

- [ ] **Step 6: Commit**

```bash
git add notebooks/04_train_cnn.ipynb
git commit -m "refactor(nb04): import RadarCNN and split/load helpers from radar"
```

---

## Task 10: Clean notebook 05 (compare)

**Files:**
- Modify: `notebooks/05_compare.ipynb`

- [ ] **Step 1: Update the imports cell**

Replace the top imports cell with:

```python
import numpy as np
import torch
import matplotlib.pyplot as plt

from radar.data import load_radchar
from radar.model import RadarCNN
from radar.traditional import build_templates, classify

DATA_PATH = "../data/RadChar-Small.h5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- [ ] **Step 2: Use `load_radchar` for the data-load cell**

Replace the `with h5py.File(...)` load cell with:

```python
iq, labels = load_radchar(DATA_PATH)
y = labels["signal_type"].astype(np.int64)
snr = labels["signal_to_noise_ratio"]
```

- [ ] **Step 3: Delete the duplicated `RadarCNN` class cell**

Remove the cell that re-defines `class RadarCNN(nn.Module): ...`. The model-load
cell stays but now relies on the imported class:

```python
model = RadarCNN().to(device)
model.load_state_dict(torch.load("../results/radar_cnn.pth", map_location=device))
model.eval()
print("Model loaded.")
```

- [ ] **Step 4: Replace the inline spectral-classifier cell with imports**

Remove the cell defining `spectral_feature` and the inline template/`traditional_classify`
logic. Replace with:

```python
# Build the spectral templates from TRAIN ONLY (no leakage), then the
# comparison loop below labels test samples with radar.traditional.classify.
templates = build_templates(iq[train_idx], y[train_idx])
print("Built spectral templates for 5 classes from training data.")
```

In the per-SNR comparison loop, replace the `trad_pred = traditional_classify(idx)`
line with:

```python
trad_pred = classify(iq[idx], templates)
```

- [ ] **Step 5: Verify imports resolve**

Run: `jupyter nbconvert --to script --stdout notebooks/05_compare.ipynb > /dev/null`
Expected: completes with no error.

- [ ] **Step 6: Commit**

```bash
git add notebooks/05_compare.ipynb
git commit -m "refactor(nb05): import model + traditional classifier from radar"
```

---

## Task 11: Write the README (collaborative)

**Files:**
- Create: `README.md`

- [ ] **Step 1: Ask the user for the three narrative pieces**

Before writing, ask the user (one at a time) for:
1. The one-paragraph pitch — what the project is and why it exists (the Saab thesis inspiration).
2. The headline result they want to lead with.
3. The "what I learned" note (radar/DSP from scratch, regularisation, fair comparison).

Do not auto-generate these; use the user's words.

- [ ] **Step 2: Assemble `README.md`**

Use this skeleton, filling the prose from Step 1 and the table from known results:

```markdown
# Radar Signal Classification — DSP Baseline vs 1D CNN

<!-- pitch paragraph from user -->

## Result

<!-- headline result from user -->

| Model | -10 dB | 0 dB | +10 dB |
|-------|--------|------|--------|
| CNN1D (RadChar paper) | 0.757 | 0.998 | 1.000 |
| IQST-L (RadChar paper) | 0.791 | 0.998 | 1.000 |
| **This project (1D CNN)** | **~0.82** | **~1.00** | **~1.00** |

![Comparison](results/comparison_plot.png)

Benchmark source: RadChar paper, arXiv:2306.13105.

## Project structure

- `notebooks/` — the story, 01 (explore) -> 05 (compare)
- `src/radar/` — reusable code (data, model, traditional classifier)
- `results/` — trained model, split indices, plots
- `docs/project_plan.md` — the full project plan

## Setup

```bash
pip install -e .
```

Download RadChar into `data/` (see https://github.com/abcxyzi/RadChar).

## What I learned

<!-- lessons from user -->
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add portfolio README"
```

---

## Task 12: Final verification

- [ ] **Step 1: Clean install check**

Run: `pip install -e . && python -c "import radar.data, radar.model, radar.traditional; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 2: Confirm no leftover duplication**

Run: `grep -rn "class RadarCNN" notebooks/`
Expected: no matches (the class now lives only in `src/radar/model.py`).

- [ ] **Step 3: Note the remaining manual step**

Notebooks 04 (training) and 05 (comparison) must be re-run on a GPU (e.g. Kaggle)
to regenerate `results/radar_cnn.pth`, `comparison_plot.png`, and capture exact
per-SNR decimals for the README table. This is intentionally outside this plan.

- [ ] **Step 4: Final commit (if anything pending)**

```bash
git add -A
git commit -m "chore: final cleanup pass"
```

---

## Self-review notes

- **Spec coverage:** packaging (T1), data/model/traditional extraction (T2–T4),
  plan move (T5), all five notebooks (T6–T10), README (T11), git init done before
  planning, final verification + re-run flag (T12). All spec sections covered.
- **Type consistency:** `load_radchar(path) -> (iq, labels)`, `make_split(n, seed)
  -> (train_idx, val_idx, test_idx)`, `build_templates(iq_train, y_train)`,
  `classify(iq_samples, templates)`, `RadarCNN()` — names used consistently in
  notebook tasks.
- **Placeholders:** README prose is genuinely collaborative (Step 1 gathers the
  user's words first), not a code placeholder.
