
import torch
import torch.nn as nn
from torchvision import models


class Baseline4(nn.Module):
    def __init__(self, hidden_size, num_classes=8):
        super(Baseline4, self).__init__()

        # Load ResNet50 pretrained on ImageNet.
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Remove the final classification layer.
        self.model = nn.Sequential(*list(resnet.children())[:-1])

        # LSTM processes the sequence of frame features.
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Predict the group activity class.
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Input shape: [B, F_S, C, H, W]
        B, F_S, C, H, W = x.shape

        # Process every frame independently through ResNet50.
        x = x.reshape(B * F_S, C, H, W)
        x = self.model(x)

        # Reshape features into a temporal sequence.
        # Shape: [B, F_S, 2048]
        x = x.reshape(B, F_S, -1)

        # Learn temporal relationships between frames.
        x, _ = self.lstm(x)

        # Use the output of the final frame.
        x = x[:, -1, :]

        # Return raw class logits.
        x = self.fc(x)

        return x
