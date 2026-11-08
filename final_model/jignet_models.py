import torch
import torch.nn as nn

class JigEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.AdaptiveAvgPool2d(output_size=(2, 2)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.encoder(x)

class JigNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.encoder = JigEncoder()
        feature_dim = 512
        combined_dim = feature_dim * 4

        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, tile_a, tile_b):
        feature_a = self.encoder(tile_a)
        feature_b = self.encoder(tile_b)
        diff = torch.abs(feature_a - feature_b)
        prod = feature_a * feature_b
        combined = torch.cat([feature_a, feature_b, diff, prod], dim=1)
        return self.mlp(combined)
