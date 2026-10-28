
"""Baseline 5: Team-level temporal modeling with an LSTM."""

import torch
import torch.nn as nn
from torchvision import models


class Baseline5(nn.Module):
    def __init__(self, hidden_size1=2048, num_classes=8):
        super().__init__()

        # Pretrained CNN for extracting features from player images.
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )
        self.model = nn.Sequential(*list(resnet.children())[:-1])

        # Model the temporal sequence after aggregating player features.
        self.image_lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size1,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.3)

        # Predict one group activity class for the complete sequence.
        self.fc = nn.Linear(hidden_size1, num_classes)

    def forward(self, x, player_mask):
        # x: [B, N, F_S, C, H, W]
        # player_mask: [B, N]
        # True means a valid player; False means a padded slot.
        B, N, F_S, C, H, W = x.shape

        # Extract a CNN feature vector for every player in every frame.
        # Shape: [B * N * F_S, C, H, W]
        x = x.reshape(B * N * F_S, C, H, W)
        x = self.model(x)

        # Shape: [B * N * F_S, 2048]
        x = x.flatten(1)
        x = self.dropout(x)

        # Restore batch, player, and time dimensions.
        # Shape: [B, N, F_S, 2048]
        x = x.reshape(B, N, F_S, -1)

        # Put the time dimension before the player dimension.
        # Shape: [B, F_S, N, 2048]
        x = x.transpose(1, 2)

        # Expand the mask so it can be applied to every frame and feature.
        mask = player_mask.to(device=x.device, dtype=torch.bool)
        mask = mask[:, None, :, None]

        # Exclude padded player slots from max pooling.
        x = x.masked_fill(~mask, float("-inf"))

        # Aggregate player features independently at each time step.
        # Shape: [B, F_S, 2048]
        x = torch.max(x, dim=2).values

        # Keep the original architecture's fallback for non-finite values.
        # Every sample should still contain at least one valid player.
        x = torch.where(
            torch.isfinite(x),
            x,
            torch.zeros_like(x),
        )

        # Model temporal relationships between aggregated team features.
        # Shape: [B, F_S, hidden_size1]
        x, _ = self.image_lstm(x)

        # Classify using the final temporal output.
        # Shape: [B, hidden_size1]
        x = x[:, -1, :]
        x = self.fc(x)

        # Output: [B, num_classes] (raw logits).
        return x

