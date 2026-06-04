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
