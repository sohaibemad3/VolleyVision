
"""Baseline 8: Hierarchical temporal modeling with two team representations."""

import torch
import torch.nn as nn
from torchvision import models


class Baseline8(nn.Module):
    def __init__(
        self,
        hidden_size1=1024,
        hidden_size2=512,
        num_classes=8,
    ):
        super().__init__()

        # CNN backbone for extracting player-level image features.
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )
        self.model = nn.Sequential(*list(resnet.children())[:-1])

        # Temporal modeling for each player's frame sequence.
        self.player_lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size1,
            num_layers=1,
            batch_first=True,
        )

        # The input contains two concatenated team representations.
        self.image_lstm = nn.LSTM(
            input_size=hidden_size1 * 2,
            hidden_size=hidden_size2,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, player_mask):
        # x: [B, N, F_S, C, H, W]
        # player_mask: [B, N]
        B, N, F_S, C, H, W = x.shape

        # Extract features from every player crop in every frame.
        # Shape: [B * N * F_S, C, H, W]
        x = x.reshape(B * N * F_S, C, H, W)
        x = self.model(x)

        # Flatten CNN outputs into feature vectors.
        # Shape: [B * N * F_S, 2048]
        x = self.dropout(x)
        x = x.flatten(1)

        # Create a temporal sequence for each player.
        # Shape: [B * N, F_S, 2048]
        x = x.reshape(B * N, F_S, -1)

        # Learn temporal features for each player.
        # Shape: [B * N, F_S, hidden_size1]
        x, _ = self.player_lstm(x)

        # Restore the batch, player, and time dimensions.
        # Shape: [B, N, F_S, hidden_size1]
        x = x.reshape(B, N, F_S, -1)

        # Arrange the tensor for frame-by-frame team aggregation.
        # Shape: [B, F_S, N, hidden_size1]
        x = x.transpose(1, 2)

        # Split players into two groups using their positions in the
        # player dimension: slots [0:6] and [6:N].
        team1 = x[:, :, :6, :]
        team2 = x[:, :, 6:, :]

        # Split the validity mask using the same player boundaries.
        team1_mask = player_mask[:, :6].to(
            device=x.device,
            dtype=torch.bool,
        )
        team2_mask = player_mask[:, 6:].to(
            device=x.device,
            dtype=torch.bool,
        )

        # Ignore padded player slots within each team.
        team1 = team1.masked_fill(
            ~team1_mask[:, None, :, None],
            float("-inf"),
        )
        team2 = team2.masked_fill(
            ~team2_mask[:, None, :, None],
            float("-inf"),
        )

        # Aggregate the players separately for each team.
        # Each result has shape [B, F_S, hidden_size1].
        team1 = torch.max(team1, dim=2).values
        team2 = torch.max(team2, dim=2).values

        # Concatenate both team representations along the feature axis.
        # Shape: [B, F_S, hidden_size1 * 2]
        x = torch.cat((team1, team2), dim=-1)

        # Model the temporal relationship between the two team features.
        # Shape: [B, F_S, hidden_size2]
        x, _ = self.image_lstm(x)

        # Use the final temporal output for group activity classification.
        # Shape: [B, hidden_size2]
        x = x[:, -1, :]

        # Output: [B, num_classes] (raw logits).
        x = self.fc(x)

        return x

