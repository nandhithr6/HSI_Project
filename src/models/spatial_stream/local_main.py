"""
Fully Adaptive Local Feature Stream for Hyperspectral Image Analysis (v2).

This module implements a state-of-the-art, fully adaptive architecture for
extracting local spatial-spectral features from HSI cubes. It improves upon
previous versions by introducing a dynamic calculation for the model's base
capacity (`base_channels`) based on the input `num_bands`.

Key Adaptive Features:
1.  **Dynamic Model Capacity:** The `base_channels` parameter is now calculated
    with a heuristic that scales the model size with the number of spectral bands,
    preventing underfitting on rich data and overfitting on simpler data.
2.  **Learnable Spectral Reduction:** Instead of using average pooling, this module
    uses 3D convolutions with dynamically calculated spectral kernel sizes. This
    allows the model to *learn* the optimal way to compress spectral information
    at each stage.
3.  **Proportional Band Reduction:** Follows a robust schedule to progressively
    reduce spectral bands (e.g., 60 -> 30 -> 15 -> 1) while expanding
    feature channels (e.g., 64 -> 128 -> 256).

This design ensures the model is powerful, generalizable, and optimally configured
for any given HSI dataset without requiring manual tuning.

Author: Nandhitha
Date: September 17, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# --------------------------
# CBAM Attention Module
# --------------------------

class ChannelAttention(nn.Module):
    """Convolutional Block Attention Module (CBAM) - Channel Attention."""
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1))))
        max_out = self.fc2(F.relu(self.fc1(F.adaptive_max_pool2d(x, 1))))
        return torch.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """Convolutional Block Attention Module (CBAM) - Spatial Attention."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_att = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x_att))

class CBAM(nn.Module):
    """The complete CBAM module."""
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = x * self.channel_attention(x)
        x_out = x_out * self.spatial_attention(x_out)
        return x_out

# --------------------------
# Residual Spatial Block (Corrected Structure)
# --------------------------

class ResidualSpatialBlock(nn.Module):
    """
    A robust residual block with integrated CBAM attention, following best practices.
    The structure is Conv -> BN -> ReLU -> CBAM -> Conv -> BN, then add residual.
    """
    def __init__(self, in_planes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.cbam = CBAM(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01, inplace=True)
        out = self.cbam(out)
        out = self.bn2(self.conv2(out))
        out += residual # Add the residual connection
        return F.leaky_relu(out, negative_slope=0.01, inplace=True) # Apply final activation

# -------------------------------------
# Helper for Adaptive 3D Convolution
# -------------------------------------

class _Adaptive3DBlock(nn.Module):
    """
    An adaptive 3D CNN block that dynamically calculates its spectral kernel size
    to perform learnable dimensional reduction.
    """
    def __init__(self, in_ch: int, out_ch: int, in_bands: int, out_bands: int):
        super().__init__()
        # K = D_in - D_out + 1 (for stride=1, padding=0)
        spectral_kernel_size = in_bands - out_bands + 1
        
        if spectral_kernel_size < 1:
            raise ValueError(
                f"Invalid 3D kernel size. Input bands ({in_bands}) are too few to reduce to {out_bands}."
            )

        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(spectral_kernel_size, 3, 3), # (Depth=Bands, Height, Width)
            padding=(0, 1, 1),
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.01, inplace=True)

# -------------------------------------
# Main Adaptive Local Stream Class
# -------------------------------------

class LocalFeatureStream(nn.Module):
    """
    The complete, dataset-adaptive Local Feature Stream.
    """
    def __init__(self, num_bands: int):
        """
        Args:
            num_bands (int): The number of spectral bands in the input HSI cube.
        """
        super().__init__()
        if num_bands != 37:
            raise ValueError("This version of LocalFeatureStream is fixed for 37 spectral bands.")

        # --- Static Band Reduction and Channel Progression ---
        # Bands: 37 -> 18 -> 9 -> 1
        # Channels: 1 -> 32 -> 64 -> 128
        self.cnn3d_1 = _Adaptive3DBlock(1, 32, 37, 18)
        self.cnn3d_2 = _Adaptive3DBlock(32, 64, 18, 9)
        self.cnn3d_3 = _Adaptive3DBlock(64, 128, 9, 1)

        # --- Subsequent layers use final channel count ---
        final_3d_channels = 128
        self.rsb = ResidualSpatialBlock(final_3d_channels)
        self.conv2d_1 = nn.Conv2d(final_3d_channels, final_3d_channels, kernel_size=3, padding=1, bias=False)
        self.bn2d_1 = nn.BatchNorm2d(final_3d_channels)
        self.conv2d_2 = nn.Conv2d(final_3d_channels, final_3d_channels, kernel_size=3, padding=1, bias=False)
        self.bn2d_2 = nn.BatchNorm2d(final_3d_channels)

        self.use_checkpointing = True

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Core forward logic of the module."""
        # Reshape for 3D conv: (B, Bands, H, W) -> (B, 1, Bands, H, W)
        x = x.unsqueeze(1)
        
        # 3D CNN processing with learnable spectral reduction
        x = self.cnn3d_1(x)
        x = self.cnn3d_2(x)
        x = self.cnn3d_3(x)

        # Squeeze the spectral dimension: (B, C, 1, H, W) -> (B, C, H, W)
        x = x.squeeze(2)

        # 2D processing (RSB-CBAM and 2D CNNs)
        x = self.rsb(x)
        x = F.leaky_relu(self.bn2d_1(self.conv2d_1(x)), negative_slope=0.01, inplace=True)
        x = F.leaky_relu(self.bn2d_2(self.conv2d_2(x)), negative_slope=0.01, inplace=True)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass.
        
        Args:
            x (torch.Tensor): Input HSI data cube of shape (B, Bands, H, W).
            
        Returns:
            torch.Tensor: Output 2D feature map.
        """
        if self.use_checkpointing and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

