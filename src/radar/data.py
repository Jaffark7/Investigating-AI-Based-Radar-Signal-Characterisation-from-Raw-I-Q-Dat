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
