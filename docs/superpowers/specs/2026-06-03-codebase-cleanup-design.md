# Codebase Cleanup & Restructure — Design

**Date:** 2026-06-03
**Goal:** Turn the working-but-rough radar project into a clean, coherent,
best-practice repository ready to publish on GitHub as a portfolio piece.

---

## Context

The project works end to end (notebooks 01–05) and produces a strong result:
a 1D CNN matches/beats the published RadChar CNN1D baseline at low SNR. But the
codebase has rough edges that undercut it as a portfolio piece:

- No README, no `requirements.txt`, repo not git-initialized
- `RadarCNN` class duplicated across notebooks 04 and 05
- `h5py` data-loading boilerplate repeated in all 5 notebooks
- A stale comment (01: "50k" — actually 500k) and an inconsistent "pitch"
  analogy (02)
- Obscure indexing: `labels[labels.dtype.names[1]]` instead of `labels['signal_type']`

## Decisions (agreed)

| Decision | Choice |
|----------|--------|
| Cleanup depth | Polish notebooks **and** extract shared code into a `src/` module |
| Language | English everywhere; README content drafted collaboratively |
| Import mechanism | Installable package — `pip install -e .`, then `from radar... import ...` |

## Best-practice anchors

- **Comments explain *why*, not *what*** (Real Python, PEP 8). Cut comments that
  only restate code. Keep ≤72 chars, space after `#`, English, grammatical, never stale.
- **Preserve the teaching voice** where it explains radar concepts — this is a
  learning portfolio, so concept-explaining comments add value. Only redundant
  narration is removed.
- **`src/` holds reusable code; `notebooks/` tells the story** (Towards Data Science).
  No duplicated definitions across notebooks.

## Target structure

```
ai-radar/
├── README.md              # portfolio pitch (drafted together)
├── pyproject.toml         # makes `radar` installable
├── requirements.txt       # pinned dependencies
├── .gitignore             # tidied
├── docs/
│   └── project_plan.md    # moved from root radar_ai_project_plan.md
├── data/                  # unchanged (gitignored)
├── notebooks/             # 01–05, cleaned; import from `radar`
├── src/
│   └── radar/
│       ├── __init__.py
│       ├── data.py        # load_radchar(path), make_split(n, seed)
│       ├── model.py       # RadarCNN
│       └── traditional.py # spectral_feature, build_templates, classify
└── results/               # model weights, split indices, plots
```

## What moves into `src/radar/`

- **`data.py`** — `load_radchar(path)` wraps the `h5py` open/read; `make_split`
  produces the deterministic 70/15/15 index split (seeded). Replaces boilerplate
  in all notebooks.
- **`model.py`** — the `RadarCNN` class. Single source of truth; notebooks 04 and
  05 both import it (currently copy-pasted).
- **`traditional.py`** — spectral template classifier (`spectral_feature`,
  `build_templates`, `classify`) used by the notebook 05 comparison.

Notebook 03's matched-filter demo stays inline — one-off teaching visual, not reused.

## Notebook cleanup (all 5)

- Fix stale comment (01), remove "pitch" analogy (02)
- `labels[labels.dtype.names[1]]` → `labels['signal_type']`
- Replace duplicated/boilerplate code with imports from `radar`
- Consistent markdown intro + header style across all notebooks
- Apply the comment principles above

## README

Drafted collaboratively — the user is asked what to write for the pitch, the
headline result, and the "what I learned" section, rather than auto-generated.
Will include the CNN-vs-published comparison table and the comparison plot.

## Sequencing / open items

1. Restructure changes the code the notebooks ran. To publish honestly, notebooks
   must be **re-run top-to-bottom** afterward to regenerate outputs, the plot, and
   the model with the final (regularized) code.
2. That re-run includes GPU training (notebook 04) — likely the Kaggle run.
3. Optional: re-run notebook 05 to capture **exact** per-SNR decimals for the
   README table (currently read from the plot).

These are flagged as final steps; the cleanup itself does not block on them.

## Out of scope

- No unit-test suite (portfolio learning project; not warranted yet)
- No CI/CD, no config-file system (single `DATA_PATH` constant is fine)
- No change to the model architecture or experimental method
