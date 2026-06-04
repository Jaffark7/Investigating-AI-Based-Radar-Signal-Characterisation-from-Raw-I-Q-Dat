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
