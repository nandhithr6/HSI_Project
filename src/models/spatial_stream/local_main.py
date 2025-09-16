"""
Local Feature Stream with 3D CNN, Residual Spatial Block (RSB) with CBAM Attention Module and 2D CNN.  
Author: Nandhitha
Date: September 16, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------
# CBAM Attention Module
# ---------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1))))
        max_out = self.fc2(F.relu(self.fc1(F.adaptive_max_pool2d(x, 1))))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(out))


class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


# ---------------------------
# Residual Spatial Block (RSB)
# ---------------------------
class ResidualSpatialBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.cbam = CBAM(channels)

    def forward(self, x):
        out = self.bn(self.conv(x))
        out = self.cbam(out)
        out = F.leaky_relu(x + out, negative_slope=0.01, inplace=True)
        return out


# ---------------------------
# Local Feature Stream
# ---------------------------
class LocalStream(nn.Module):
    def __init__(self, in_channels=1, base_channels=128, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # 3D CNN stack
        self.conv3d_1 = nn.Conv3d(in_channels, base_channels // 2, kernel_size=(3, 3, 3), padding=1, bias=False)
        self.bn3d_1 = nn.BatchNorm3d(base_channels // 2)

        self.conv3d_2 = nn.Conv3d(base_channels // 2, int(base_channels * 0.75),
                                  kernel_size=(3, 3, 5), padding=(1, 1, 2), bias=False)
        self.bn3d_2 = nn.BatchNorm3d(int(base_channels * 0.75))

        self.conv3d_3 = nn.Conv3d(int(base_channels * 0.75), base_channels,
                                  kernel_size=(3, 3, 7), padding=(1, 1, 3), bias=False)
        self.bn3d_3 = nn.BatchNorm3d(base_channels)

        self.pool3d = nn.AdaptiveAvgPool3d((None, None, 1))  # squeeze bands

        # RSB + CBAM
        self.rsb = ResidualSpatialBlock(base_channels)

        # 2D CNN stack
        self.conv2d_1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False)
        self.bn2d_1 = nn.BatchNorm2d(base_channels)
        self.conv2d_2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False)
        self.bn2d_2 = nn.BatchNorm2d(base_channels)

    def _forward_3d(self, x):
        x = F.leaky_relu(self.bn3d_1(self.conv3d_1(x)), negative_slope=0.01, inplace=True)
        x = F.leaky_relu(self.bn3d_2(self.conv3d_2(x)), negative_slope=0.01, inplace=True)
        x = F.leaky_relu(self.bn3d_3(self.conv3d_3(x)), negative_slope=0.01, inplace=True)
        x = self.pool3d(x)  # (B, C, H, W, 1)
        return torch.squeeze(x, dim=-1)  # (B, C, H, W)

    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint(self._forward_3d, x, use_reentrant=False)
        else:
            x = self._forward_3d(x)

        x = x.contiguous(memory_format=torch.channels_last)
        x = self.rsb(x)

        x = F.leaky_relu(self.bn2d_1(self.conv2d_1(x)), negative_slope=0.01, inplace=True)
        x = F.leaky_relu(self.bn2d_2(self.conv2d_2(x)), negative_slope=0.01, inplace=True)
        return x
