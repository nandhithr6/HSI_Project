"""
Fully Adaptive Local Feature Stream for Hyperspectral Image Analysis (v2) - CORRECTED

This module implements a state-of-the-art, fully adaptive architecture for
extracting local spatial-spectral features from HSI cubes.

CHANGES:
- Removed the hardcoded `num_bands != 37` check.
- Implemented the adaptive logic described in the docstrings, where the model's
  base capacity (`base_channels`) and the spectral reduction schedule are
  dynamically calculated based on the input `num_bands`.
- This version is now truly adaptive and will work with HSI datasets of
  varying spectral depths.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math

# --- CBAM Attention Module ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1))))
        max_out = self.fc2(F.relu(self.fc1(F.adaptive_max_pool2d(x, 1))))
        return torch.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_att = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x_att))

class CBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = x * self.channel_attention(x)
        x_out = x_out * self.spatial_attention(x_out)
        return x_out

# --- Residual Spatial Block ---
class ResidualSpatialBlock(nn.Module):
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
        out += residual
        return F.leaky_relu(out, negative_slope=0.01, inplace=True)

# --- Adaptive 3D Convolution Block ---
class _Adaptive3DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, in_bands: int, out_bands: int):
        super().__init__()
        spectral_kernel_size = in_bands - out_bands + 1
        if spectral_kernel_size < 1:
            raise ValueError(f"Invalid 3D kernel calc. Cannot reduce from {in_bands} to {out_bands} bands in one step.")
        self.conv = nn.Conv3d(
            in_channels=in_ch, out_channels=out_ch,
            kernel_size=(spectral_kernel_size, 3, 3), padding=(0, 1, 1), bias=False
        )
        self.bn = nn.BatchNorm3d(out_ch)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.01, inplace=True)

# --- Main Adaptive Local Stream Class ---
class LocalFeatureStream(nn.Module):
    def __init__(self, num_bands: int):
        super().__init__()
        
        base_channels = 16 * int(math.log2(max(16, num_bands)))
        
        band_schedule = [
            num_bands,
            max(2, num_bands // 2),
            max(2, num_bands // 4),
            1
        ]
        band_schedule_valid = [band_schedule[0]]
        for i in range(1, len(band_schedule)):
            if band_schedule[i] < band_schedule_valid[-1]:
                 band_schedule_valid.append(band_schedule[i])
        if band_schedule_valid[-1] != 1:
            band_schedule_valid.append(1)

        channel_schedule = [1]
        for i in range(len(band_schedule_valid) -1):
            channel_schedule.append(base_channels * (2**i))
            
        self.cnn3d_stack = nn.ModuleList()
        for i in range(len(band_schedule_valid) - 1):
            self.cnn3d_stack.append(
                _Adaptive3DBlock(
                    in_ch=channel_schedule[i],
                    out_ch=channel_schedule[i+1],
                    in_bands=band_schedule_valid[i],
                    out_bands=band_schedule_valid[i+1]
                )
            )

        final_3d_channels = channel_schedule[-1]
        self.rsb = ResidualSpatialBlock(final_3d_channels)
        self.conv2d_1 = nn.Conv2d(final_3d_channels, 128, kernel_size=3, padding=1, bias=False)
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.conv2d_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn2d_2 = nn.BatchNorm2d(128)

        self.use_checkpointing = True

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        
        for layer in self.cnn3d_stack:
            x = layer(x)

        x = x.squeeze(2)

        x = self.rsb(x)
        x = F.leaky_relu(self.bn2d_1(self.conv2d_1(x)), negative_slope=0.01, inplace=True)
        x = F.leaky_relu(self.bn2d_2(self.conv2d_2(x)), negative_slope=0.01, inplace=True)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

