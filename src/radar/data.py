"""Loading and splitting the RadChar dataset."""

import json

import h5py
import numpy as np

# order of the four regression targets, shared by the model heads, the
# loss weights and the metrics
REGRESSION_TASKS = ("n_pulses", "pw", "pri", "td")

_SECONDS_TO_US = 1e6


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

    Uses the global RNG so the split matches the indices saved in results/.
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


def regression_targets(labels):
    """The four RadChar pulse parameters as an (N, 4) float array.

    Columns follow REGRESSION_TASKS order: number of pulses (raw count),
    then pulse width, PRI and time delay. The three time fields are stored
    in seconds and converted to microseconds here, to match the paper's units.
    """
    return np.stack([
        labels["number_of_pulses"].astype(np.float64),
        labels["pulse_width"] * _SECONDS_TO_US,
        labels["pulse_repetition_interval"] * _SECONDS_TO_US,
        labels["time_delay"] * _SECONDS_TO_US,
    ], axis=1)


class MinMaxNormaliser:
    """Scales regression targets to [0, 1] using train-set min/max.

    Fit on training rows only (no leakage), transform every split for
    training, and inverse_transform predictions back to real units for MAE
    reporting. Stats are saved as plain JSON so they're easy to read and reload.
    """

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)

    @classmethod
    def fit(cls, targets):
        """Fit per-column min/max on training targets only."""
        return cls(targets.min(axis=0), targets.max(axis=0))

    def transform(self, targets):
        return (targets - self.lo) / (self.hi - self.lo)

    def inverse_transform(self, scaled):
        return scaled * (self.hi - self.lo) + self.lo

    def save(self, path):
        """Write min/max to JSON so notebook 04 reloads the exact train ranges."""
        with open(path, "w") as f:
            json.dump({"lo": self.lo.tolist(), "hi": self.hi.tolist()}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            stats = json.load(f)
        return cls(stats["lo"], stats["hi"])
