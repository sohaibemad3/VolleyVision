
"""Baseline 1: Full-image group activity classification."""

import torch
import torch.nn as nn
from torchvision import models


class Baseline1(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        # Load ResNet50 pretrained on ImageNet.
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        # Remove the original classification layer.
        # Output shape: [B, 2048, 1, 1]
        self.model = nn.Sequential(*list(resnet.children())[:-1])

        # Regularize the extracted feature representation.
        self.dropout = nn.Dropout(0.2)

        # Map the 2048-dimensional feature vector to group classes.
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Input: [B, C, H, W]
        x = self.model(x)

        # Flatten spatial dimensions.
        # Shape: [B, 2048]
        x = torch.flatten(x, 1)

        x = self.dropout(x)

        # Output: [B, num_classes] (raw logits).
        x = self.fc(x)

        return x

