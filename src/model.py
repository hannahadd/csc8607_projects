"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESC50CNN2D(nn.Module):
    """
    CNN 2D for log-mel spectrogram "images"
    Input: (B, 1, 64, T)
    Output: (B, num_classes)
    """
    def __init__(self, in_channels: int, num_classes: int, channels: list[int], kernel_size: int = 3):
        super().__init__()
        assert len(channels) == 3, "channels must be [C1, C2, C3]"
        assert kernel_size in (3, 5), "kernel_size must be 3 or 5"

        c1, c2, c3 = channels
        padding = 1 if kernel_size == 3 else 2

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=kernel_size, padding=padding, bias=True),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=kernel_size, padding=padding, bias=True),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=kernel_size, padding=padding, bias=True),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling then linear head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)           # (B, C3, 1, 1)
        x = torch.flatten(x, 1)   # (B, C3)
        logits = self.fc(x)       # (B, num_classes)
        return logits


def build_model(config: dict, num_classes: int | None = None) -> nn.Module:
    m = config["model"]
    channels = [int(v) for v in m.get("channels", [32, 64, 128])]
    kernel_size = int(m.get("kernel_size", 3))

    if num_classes is None:
        # fallback if train.py forgets to pass it
        num_classes = int(m.get("num_classes", 50))

    return ESC50CNN2D(in_channels=1, num_classes=int(num_classes), channels=channels, kernel_size=kernel_size)
