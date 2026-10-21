"""JigNet: Siamese CNN that classifies the spatial relation of two 128x128 tiles.

Output classes: 0 NO_MATCH, 1 LEFT, 2 RIGHT, 3 UP, 4 DOWN
(class k means: tile B is <k> of tile A).

NOTE: layer order and attribute names are kept identical to the training notebook so that
existing `jignet_weights.pth` / checkpoint files load without any key remapping.
"""
import torch
from torch import nn


class JigEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # Input: (B, 3, 128, 128)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                           # (B, 32, 32, 32)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                           # (B, 64, 16, 16)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (B, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                           # (B, 128, 8, 8)

            # Reduce spatial dimensions without losing all spatial layout
            nn.AdaptiveAvgPool2d(output_size=(2, 2)),              # (B, 128, 2, 2)
            nn.Flatten(),                                          # (B, 512)
        )

    def forward(self, x):
        return self.encoder(x)


class JigNet(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.encoder = JigEncoder()
        feature_dim = 512                 # 128 * 2 * 2
        combined_dim = feature_dim * 4    # [F_a, F_b, |F_a - F_b|, F_a * F_b]

        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(128, num_classes),
        )

    def forward(self, tile_a, tile_b):
        feature_a = self.encoder(tile_a)
        feature_b = self.encoder(tile_b)
        diff = torch.abs(feature_a - feature_b)
        prod = feature_a * feature_b
        combined = torch.cat([feature_a, feature_b, diff, prod], dim=1)
        return self.mlp(combined)
