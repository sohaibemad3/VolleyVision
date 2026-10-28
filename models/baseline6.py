
"""Baseline 6: Per-player temporal modeling with an LSTM."""

import torch
import torch.nn as nn
from torchvision import models


class Baseline6(nn.Module):
    def __init__(self, hidden_size1=2048, num_classes=8):
        super().__init__()

        # CNN backbone for extracting features from each player crop.
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )
        self.model = nn.Sequential(*list(resnet.children())[:-1])

        # Process the frame sequence for each player independently.
        self.player_lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size1,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.5)

        # Predict the group activity after player aggregation.
        self.fc = nn.Linear(hidden_size1, num_classes)

    def forward(self, x, player_mask):
        # x: [B, N, F_S, C, H, W]
        # player_mask: [B, N]
        B, N, F_S, C, H, W = x.shape

        # Extract a feature vector from every player in every frame.
        # Shape: [B * N * F_S, C, H, W]
        x = x.reshape(B * N * F_S, C, H, W)
        x = self.model(x)

        # Shape: [B * N * F_S, 2048]
        x = x.flatten(1)
        x = self.dropout(x)

        # Treat each player as an independent sequence.
        # Shape: [B * N, F_S, 2048]
        x = x.reshape(B * N, F_S, -1)

        # Learn temporal information for each player's sequence.
        # Shape: [B * N, F_S, hidden_size1]
        x, _ = self.player_lstm(x)

        # Keep the final time step for each player.
        # Shape: [B * N, hidden_size1]
        x = x[:, -1, :]

        # Restore the player dimension.
        # Shape: [B, N, hidden_size1]
        x = x.reshape(B, N, -1)

        # Convert the player mask to boolean on the same device.
        mask = player_mask.to(device=x.device, dtype=torch.bool)

        # Exclude padded players from max pooling.
        x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        # Aggregate the player representations into a team representation.
        # Shape: [B, hidden_size1]
        x = torch.max(x, dim=1).values

        # Preserve the original fallback for non-finite values.
        # Every sample should contain at least one valid player.
        x = torch.where(
            torch.isfinite(x),
            x,
            torch.zeros_like(x),
        )

        # Output: [B, num_classes] (raw logits).
        x = self.fc(x)

        return x

