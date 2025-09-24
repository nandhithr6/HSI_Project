"""
Unified Spectral Stream Processing Framework for HSI Segmentation.

This module implements the Multi-Scale Windowed Mamba architecture described in
the research paper "Multi-Scale Windowed Mamba for Hyperspectral Image
Segmentation." It processes each pixel's spectral signature independently to
extract a rich feature representation.

The pipeline is as follows:
1.  For each pixel, the full spectral vector (Bands) is processed by Mamba
    blocks operating on multiple window sizes.
2.  The outputs from all Mamba blocks are concatenated to form a feature vector.
3.  This feature vector is treated as a "token" and is projected to a final
    embedding dimension.
4.  A **2D Positional Encoding** is added to these tokens to inject spatial
    awareness, correctly representing the height and width relationships.
5.  The final output is a spatially and spectrally aware feature map.

Author: 
Date: September 17, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import math

# --------------------------
# Helper Functions
# --------------------------

def sliding_windows_gpu(spectral_flat: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    """
    Creates sliding windows over the spectral dimension in a GPU-friendly way.
    """
    num_pixels, bands = spectral_flat.shape
    if bands < window_size:
        padding = window_size - bands
        spectral_flat = F.pad(spectral_flat, (0, padding), "constant", 0.0)
    windows = spectral_flat.unfold(dimension=1, size=window_size, step=stride)
    return windows.contiguous()

class PositionalEncoding2D(nn.Module):
    """
    Adds 2D positional encoding to the input feature map.

    This module generates separate sinusoidal positional encodings for the height
    and width dimensions and adds them to the input tensor, providing the model
    with explicit information about the spatial location of each token.
    """
    def __init__(self, d_model: int, max_h: int = 512, max_w: int = 512):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"Cannot create 2D positional encoding with odd "
                             f"d_model ({d_model}). Must be an even number.")
        
        pe = torch.zeros(d_model, max_h, max_w)
        d_model_half = d_model // 2
        
        div_term = torch.exp(torch.arange(0., d_model_half, 2) * -(math.log(10000.0) / d_model_half))
        
        pos_w = torch.arange(0., max_w).unsqueeze(1)
        pos_h = torch.arange(0., max_h).unsqueeze(1)
        
        pe[0:d_model_half:2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        pe[1:d_model_half:2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_w)
        
        pe[d_model_half::2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)
        pe[d_model_half+1::2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_h, 1)

        self.register_buffer('pe', pe) # (d_model, max_h, max_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, D, H, W).
        Returns:
            torch.Tensor: Tensor with added positional encoding.
        """
        B, D, H, W = x.shape
        return x + self.pe[:, :H, :W].unsqueeze(0)

# --------------------------
# Core Spectral Stream Module
# --------------------------

class SpectralStream(nn.Module):
    """
    The complete Spectral Stream module.
    """
    def __init__(self, num_bands: int, embed_dim: int = 128, window_sizes: list = [8, 16, 32], stride: int = 4, pixels_per_chunk: int = 8192):
        super().__init__()
        self.num_bands = num_bands
        self.window_sizes = window_sizes
        self.stride = stride
        self.pixels_per_chunk = pixels_per_chunk

        self.mamba_blocks = nn.ModuleDict()
        mamba_output_dim = 0
        for ws in window_sizes:
            self.mamba_blocks[f'mamba_ws{ws}'] = Mamba(d_model=ws, d_state=16, d_conv=4, expand=2)
            effective_bands = max(ws, num_bands)
            num_windows = 1 + (effective_bands - ws) // stride
            mamba_output_dim += num_windows * ws

        self.tokenizer_projection = nn.Linear(mamba_output_dim, embed_dim)
        # --- CORRECTED: Using 2D Positional Encoding ---
        self.pos_encoder_2d = PositionalEncoding2D(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SpectralStream.
        Args:
            x (torch.Tensor): Input HSI cube of shape (B, Bands, H, W).
        Returns:
            torch.Tensor: A spectral feature map of shape (B, embed_dim, H, W).
        """
        B, Bands, H, W = x.shape
        if Bands != self.num_bands:
            raise ValueError(f"Input tensor has {Bands} bands, but model was initialized with {self.num_bands}.")

        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, self.num_bands)
        N = x_flat.size(0)
        ppc = max(1024, int(self.pixels_per_chunk))

        tokens_chunks = []
        for start in range(0, N, ppc):
            end = min(start + ppc, N)
            x_chunk = x_flat[start:end]

            mamba_outputs = []
            for ws in self.window_sizes:
                windows = sliding_windows_gpu(x_chunk, window_size=ws, stride=self.stride)
                mamba_out = self.mamba_blocks[f'mamba_ws{ws}'](windows)
                mamba_outputs.append(mamba_out.flatten(start_dim=1))

            concatenated_features = torch.cat(mamba_outputs, dim=1)
            tokens_chunk = self.tokenizer_projection(concatenated_features)
            tokens_chunks.append(tokens_chunk)

        tokens = torch.cat(tokens_chunks, dim=0)

        # --- Reshape for 2D PE application ---
        # (N, D) -> (B, H, W, D)
        tokens_2d = tokens.view(B, H, W, -1)
        
        # Apply normalization
        tokens_2d = self.norm(tokens_2d)

        # Permute to (B, D, H, W) for convolutions and PE
        tokens_permuted = tokens_2d.permute(0, 3, 1, 2).contiguous()

        # Add 2D Positional Encoding
        output_map = self.pos_encoder_2d(tokens_permuted)

        return output_map

