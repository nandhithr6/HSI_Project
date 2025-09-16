"""
Spatial Tokenizer Module with 2D Sinusoidal Positional Encoding -- ready for TCME input.
Author: Nandhitha
Date: September 16, 2025
"""

import torch
import torch.nn as nn
import math


def get_1d_sin_cos(pos, dim):
    """1D sin-cos encoding for positions."""
    omega = torch.arange(dim, dtype=torch.float32) / dim
    omega = 1.0 / (10000 ** omega)
    out = pos.flatten()[..., None] * omega[None]
    sin, cos = torch.sin(out), torch.cos(out)
    return torch.cat([sin, cos], dim=1)  # (num_positions, dim*2)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """2D sinusoidal positional encoding with output dim == embed_dim."""
    assert embed_dim % 2 == 0, "Embed dim must be even"
    half_dim = embed_dim // 2  # half for H, half for W

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")

    emb_h = get_1d_sin_cos(grid[0], half_dim // 2)  # (grid^2, half_dim)
    emb_w = get_1d_sin_cos(grid[1], half_dim // 2)  # (grid^2, half_dim)

    return torch.cat([emb_h, emb_w], dim=1)  # (grid^2, embed_dim)


class SpatialTokenizer(nn.Module):
    def __init__(self, in_channels, embed_dim=256, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.unfold(x).transpose(1, 2)  # (B, N_patches, P^2*C)
        tokens = self.proj(patches)  # (B, N_patches, D)

        # Positional encoding
        grid_size = int(math.sqrt(tokens.size(1)))
        pe = get_2d_sincos_pos_embed(self.embed_dim, grid_size).to(x.device)  # (N_patches, D)
        tokens = tokens + pe.unsqueeze(0)  # (B, N_patches, D)

        return tokens
