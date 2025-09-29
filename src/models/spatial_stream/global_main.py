"""
Inherently Adaptive Global Feature Stream for HSI Analysis.

This module uses a 2D Vision Mamba (2DVMamba) architecture, which is
naturally suited for HSI data with varying numbers of spectral bands.

How it adapts:
1.  **Convolutional Stem:** The first layer is a 2D convolution that treats
    the input spectral bands as input channels (`in_channels=num_bands`).
    It learns a projection to map the entire spectral vector at each pixel
    into a single feature vector of `embed_dim`.
2.  **2D Processing:** After the stem, all subsequent operations (the Mamba
    blocks) are purely 2D, operating on the patch-embedded feature map.

This design is elegant and efficient, requiring no complex manual calculations
for different datasets.

Author: Nandhitha
Date: September 17, 2025
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba

# --------------------------
# 2D Mamba Block
# --------------------------

class Mamba2DBlock(nn.Module):
    """
    A 2D Mamba block that applies bidirectional scanning (row-wise and column-wise).
    """
    def __init__(self, embed_dim: int, mamba_d_state: int = 16, mamba_d_conv: int = 4, mamba_expand: int = 2):
        """
        Args:
            embed_dim (int): The embedding dimension of the input tokens.
            mamba_d_state (int): The state space dimension for the Mamba layer.
            mamba_d_conv (int): The convolution kernel size for the Mamba layer.
            mamba_expand (int): The expansion factor for the Mamba layer.
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mamba = Mamba(
            d_model=embed_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, C).
        
        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        B, H, W, C = x.shape
        residual = x
        x = self.norm(x)

        # Row-wise scan
        x_row = x.view(B * H, W, C)
        x_row = self.mamba(x_row)
        x_row = x_row.view(B, H, W, C)

        # Column-wise scan
        x_col = x.permute(0, 2, 1, 3).contiguous().view(B * W, H, C)
        x_col = self.mamba(x_col)
        x_col = x_col.view(B, W, H, C).permute(0, 2, 1, 3).contiguous()

        # Add both scans to the residual
        return residual + x_row + x_col

# --------------------------
# Main Global Stream Class
# --------------------------

class GlobalFeatureStream(nn.Module):
    """
    The complete, dataset-adaptive Global Feature Stream.
    """
    def __init__(self, num_bands: int, embed_dim: int = 256, patch_size: int = 4, num_layers: int = 4):
        """
        Args:
            num_bands (int): The number of spectral bands in the input HSI cube.
                             This is used as `in_channels` for the stem.
            embed_dim (int): The dimension of the feature map after patching.
            patch_size (int): The size of the patches to extract.
            num_layers (int): The number of Mamba2DBlocks to stack.
        """
        super().__init__()
        self.patch_size = patch_size

        # --- The Key to Adaptability ---
        # The Conv2D stem treats bands as channels, learning to project them.
        self.stem = nn.Conv2d(
            in_channels=num_bands, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size, 
            bias=False
        )
        self.norm_in = nn.BatchNorm2d(embed_dim)

        self.layers = nn.ModuleList([
            Mamba2DBlock(embed_dim=embed_dim) for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(embed_dim)
        # Allow trainer to toggle gradient checkpointing on this module
        self.use_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the GlobalFeatureStream.
        
        Args:
            x (torch.Tensor): Input HSI data cube of shape (B, Bands, H, W).
        
        Returns:
            torch.Tensor: Output 2D feature map of shape (B, embed_dim, H/patch, W/patch).
        """
        # 1. Patch Embedding (B, Bands, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.stem(x)
        x = self.norm_in(x)
        B, C, H, W = x.shape

        # 2. Reshape for Mamba blocks (B, embed_dim, H/P, W/P) -> (B, H/P, W/P, embed_dim)
        x = x.permute(0, 2, 3, 1).contiguous()

        # 3. Process through Mamba layers
        if self.use_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for layer in self.layers:
                x = checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x)

        # 4. Final normalization and reshape back to image format
        x = self.norm_out(x)
        # (B, H/P, W/P, embed_dim) -> (B, embed_dim, H/P, W/P)
        output_map = x.permute(0, 3, 1, 2).contiguous()
        
        return output_map

