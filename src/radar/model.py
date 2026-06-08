"""The multi-task 1D CNN that solves the full RadChar task from raw I/Q."""

import torch.nn as nn


class RadarMTL(nn.Module):
    """Input (batch, 2, 512) -> dict of signal-type logits + 4 pulse params.

    A shared convolutional backbone (3 conv blocks, 2 -> 32 -> 64 -> 128
    channels, adaptive average pool -> 128-dim features) feeds five heads:
    one classification head (5 signal-type logits) and four regression
    heads, one scalar each for the RadChar pulse parameters. Together these
    five outputs form the Pulse Descriptor Word.

    forward() returns a dict keyed by task name so callers are
    order-independent: {"signal_type", "n_pulses", "pw", "pri", "td"}.
    The regression keys follow radar.data.REGRESSION_TASKS.
    """

    REGRESSION_TASKS = ("n_pulses", "pw", "pri", "td")

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=11, padding=5), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        # One classification head plus a head per regression task, all
        # branching off the same shared features.
        self.signal_type = self._head(5)
        self.regression = nn.ModuleDict(
            {task: self._head(1) for task in self.REGRESSION_TASKS}
        )

    @staticmethod
    def _head(out_features):
        """A prediction head off the 128-dim shared feature vector."""
        return nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.5),       # regularisation: reduces overfitting
            nn.Linear(64, out_features),
        )

    def forward(self, x):
        z = self.features(x)
        out = {"signal_type": self.signal_type(z)}
        for task, head in self.regression.items():
            out[task] = head(z).squeeze(-1)   # (batch, 1) -> (batch,)
        return out
