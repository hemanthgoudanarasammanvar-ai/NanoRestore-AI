"""
NanoRestore-AI
Lightweight image restoration model for semiconductor inspection.

Input:
    Grayscale degraded image: [B, 1, H, W]

Output:
    Restored grayscale image: [B, 1, H, W]

The model performs denoising and detail restoration.
A separate upsampling stage can be added when the official KLA
dataset's exact scale configuration is confirmed.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Lightweight residual feature-refinement block."""

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class DetailBlock(nn.Module):
    """Refines high-frequency image details."""

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class NanoRestoreAI(nn.Module):
    """
    Lightweight restoration network.

    Architecture:
        Input
          ↓
        Feature extraction
          ↓
        Residual refinement
          ↓
        Detail refinement
          ↓
        Reconstruction
          ↓
        Residual output
    """

    def __init__(self, in_channels=1, out_channels=1, features=48):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        self.residual_layers = nn.Sequential(
            ResidualBlock(features),
            ResidualBlock(features),
            ResidualBlock(features),
            ResidualBlock(features)
        )

        self.detail = DetailBlock(features)

        self.tail = nn.Conv2d(
            features,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        features = self.head(x)

        refined = self.residual_layers(features)

        detailed = self.detail(refined)

        residual = self.tail(detailed)

        # Global residual connection.
        # The network learns the correction instead of
        # reconstructing the complete image from scratch.
        output = x + residual

        return output


def create_model():
    """Create a NanoRestore-AI model."""

    return NanoRestoreAI(
        in_channels=1,
        out_channels=1,
        features=48
    )


if __name__ == "__main__":
    # Simple architecture sanity check.
    model = create_model()

    dummy_input = torch.randn(1, 1, 256, 256)

    with torch.no_grad():
        output = model(dummy_input)

    print("NanoRestore-AI")
    print("-" * 40)
    print(f"Input shape : {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(
        f"Parameters  : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
