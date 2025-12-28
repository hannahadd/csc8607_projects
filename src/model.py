"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

import torch
import torch.nn as nn


class ESC50CNN2D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, channels, kernel_size: int):
        super().__init__()
        c1, c2, c3 = channels
        padding = 1 if kernel_size == 3 else 2

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_model(config):
    m = config["model"]
    num_classes = int(m["num_classes"])
    channels = m["channels"]
    kernel_size = int(m["kernel_size"])
    return ESC50CNN2D(1, num_classes, channels, kernel_size)
