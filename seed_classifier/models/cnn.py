"""Small CNN backbone + CORN ordinal head for blend-severity classification
on 64x64 multi-band cutouts."""
import torch.nn as nn


class BlendCNN(nn.Module):
    def __init__(self, n_bands, num_classes=4, base_channels=32):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(n_bands, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(c, c * 2, 3, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(c * 2, c * 4, 3, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(c * 4, c * 8, 3, padding=1), nn.BatchNorm2d(c * 8), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 8, c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(c * 4, num_classes - 1),  # CORN logits
        )

    def forward(self, x):
        return self.head(self.features(x))
