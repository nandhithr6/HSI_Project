"""
Spatial Fusion Module
Combines local and global spatial features using dynamic weighting, pixel-wise cross-attention, and le
Author: Nandhitha
Date: September 16, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialFusion(nn.Module):
    def __init__(self, channels, reduction=8, leak_thresh=0.1, leak_factor=0.2):
        super().__init__()
        self.channels = channels
        self.leak_thresh = leak_thresh
        self.leak_factor = leak_factor

        # Dynamic weighting
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2 * channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, 2, bias=False)

        # Cross-attention projections
        self.Wq = nn.Linear(channels, channels, bias=False)
        self.Wk = nn.Linear(channels, channels, bias=False)
        self.Wv = nn.Linear(channels, channels, bias=False)

    def forward(self, Floc, Fglob):
        B, C, H, W = Floc.shape

        # ---------------- Dynamic weighting ----------------
        g = self.gap(torch.cat([Floc, Fglob], dim=1)).view(B, -1)
        weights = torch.softmax(self.fc2(F.relu(self.fc1(g))), dim=1)
        alpha, beta = weights[:, 0].view(B, 1, 1, 1), weights[:, 1].view(B, 1, 1, 1)
        Fdw = alpha * Floc + beta * Fglob

        # ---------------- Pixel-wise cross-attention ----------------
        q = self.Wq(Fdw.permute(0, 2, 3, 1)).view(B, H * W, C)
        k = self.Wk(Fglob.permute(0, 2, 3, 1)).view(B, H * W, C)
        v = self.Wv(Fglob.permute(0, 2, 3, 1)).view(B, H * W, C)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # FlashAttention

        attn = attn.view(B, H, W, C).permute(0, 3, 1, 2)

        # ---------------- Leaky gating ----------------
        gamma = torch.mean(attn, dim=1, keepdim=True)
        mask = (gamma > self.leak_thresh).float() + self.leak_factor * (gamma <= self.leak_thresh).float()
        Ffused = mask * Fdw + (1 - mask) * attn

        return Ffused
