
"""Baseline 7: Hierarchical player and team temporal modeling."""

import torch
import torch.nn as nn
from torchvision import models


class Baseline7(nn.Module):
    def __init__(
        self,
        hidden_size1=1024,
        hidden_size2=512,
        num_classes=8,
    ):
        super().__init__()

        # Extract a 2048-dimensional feature vector from each player crop.
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )
        self.model = nn.Sequential(*list(resnet.children())[:-1])

        # First temporal level: model each player's frame sequence.
        self.player_lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size1,
            num_layers=1,
            batch_first=True,
        )

        # Second temporal level: model the sequence of team representations.
        self.image_lstm = nn.LSTM(
            input_size=hidden_size1,
            hidden_size=hidden_size2,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.5)

        # Map the final team representation to group activity classes.
        self.fc = nn.Linear(hidden_size2, num_classes)

    def forward(self, x, player_mask):
        # x: [B, N, F_S, C, H, W]
        # player_mask: [B, N]
        B, N, F_S, C, H, W = x.shape

        # Extract CNN features from all player crops.
        # Shape: [B * N * F_S, C, H, W]
        x = x.reshape(B * N * F_S, C, H, W)
        x = self.model(x)

        # Flatten the CNN output into feature vectors.
        # Shape: [B * N * F_S, 2048]
        x = self.dropout(x)
        x = x.flatten(1)

        # Build one temporal sequence per player.
        # Shape: [B * N, F_S, 2048]
        x = x.reshape(B * N, F_S, -1)

        # Learn temporal features for each player.
        # Shape: [B * N, F_S, hidden_size1]
        x, _ = self.player_lstm(x)

        # Restore batch, player, and time dimensions.
        # Shape: [B, N, F_S, hidden_size1]
        x = x.reshape(B, N, F_S, -1)

        # Arrange dimensions for aggregation across players.
        # Shape: [B, F_S, N, hidden_size1]
        x = x.transpose(1, 2)

        # Exclude padded players before aggregating their features.
        mask = player_mask.to(device=x.device, dtype=torch.bool)
        x = x.masked_fill(
            ~mask[:, None, :, None],
            float("-inf"),
        )

        # Create one team representation for each time step.
        # Shape: [B, F_S, hidden_size1]
        x = torch.max(x, dim=2).values

        # Model how the team representation changes over time.
        # Shape: [B, F_S, hidden_size2]
        x, _ = self.image_lstm(x)

        # Use the final team-level temporal representation.
        # Shape: [B, hidden_size2]
        x = x[:, -1, :]

        # Output: [B, num_classes] (raw logits).
        x = self.fc(x)

        return x

