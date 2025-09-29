"""
Unified Spectral Stream - V2 (Patch-Based Tokenization)

This module has been completely redesigned to solve the memory bottleneck.
Instead of pixel-level tokenization, it now uses PATCH-BASED tokenization,
creating a manageable number of tokens and aligning its output with the spatial stream.

The pipeline is as follows:
1.  The HSI cube is flattened pixel-wise and processed in chunks by multi-scale
    windowed Mamba blocks to extract a rich spectral feature vector for each pixel.
2.  These pixel features are reassembled into a 2D feature map of shape (B, D, H, W).
3.  A SpatialTokenizer (the same kind used in the spatial stream) is then applied
    to this feature map to create patch-based tokens.
4.  The final output is a sequence of tokens (e.g., [B, 1024, D]), not a feature map.

This change fundamentally solves the CUDA memory errors and simplifies the TCME design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from src.models.spatial_stream.spatial_tokenizer import SpatialTokenizer

# --- Helper for GPU-friendly sliding windows ---
def sliding_windows_gpu(spectral_flat: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    num_pixels, bands = spectral_flat.shape
    if bands < window_size:
        padding = window_size - bands
        spectral_flat = F.pad(spectral_flat, (0, padding), "constant", 0.0)
    windows = spectral_flat.unfold(dimension=1, size=window_size, step=stride)
    return windows.contiguous()

# --- Core Spectral Stream Module (V2) ---
class SpectralStream(nn.Module):
    def __init__(self,
                 num_bands: int,
                 embed_dim: int = 128,
                 patch_size: int = 16,
                 window_sizes: list = [8, 16, 32],
                 stride: int = 4,
                 pixels_per_chunk: int = 2048):
        super().__init__()
        self.num_bands = num_bands
        self.window_sizes = window_sizes
        self.stride = stride
        self.pixels_per_chunk = pixels_per_chunk

        # Mamba blocks to process raw spectral vectors
        self.mamba_blocks = nn.ModuleDict()
        mamba_output_dim = 0
        for ws in window_sizes:
            self.mamba_blocks[f'mamba_ws{ws}'] = Mamba(d_model=ws, d_state=16, d_conv=4, expand=2)
            effective_bands = max(ws, num_bands)
            num_windows = 1 + (effective_bands - ws) // stride
            mamba_output_dim += num_windows * ws
        
        # Projection to create the intermediate feature map
        self.feature_projection = nn.Linear(mamba_output_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # --- NEW: Tokenizer for patch-based output ---
        self.tokenizer = SpatialTokenizer(
            in_channels=embed_dim,
            embed_dim=embed_dim, # Output token dim is same as feature map dim
            patch_size=patch_size
        )
        # Toggle for gradient checkpointing (set by trainer)
        self.use_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the redesigned SpectralStream.
        Args:
            x (torch.Tensor): Input HSI cube of shape (B, Bands, H, W).
        Returns:
            torch.Tensor: A sequence of patch-based spectral tokens (B, N_patches, D).
        """
        B, Bands, H, W = x.shape
        if Bands != self.num_bands:
            raise ValueError(f"Input tensor has {Bands} bands, but model was initialized with {self.num_bands}.")

        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, self.num_bands)
        N = x_flat.size(0)
        ppc = max(1024, int(self.pixels_per_chunk))

        # 1. Process spectral vectors with Mamba blocks (chunked for memory efficiency)
        pixel_features_chunks = []
        for start in range(0, N, ppc):
            end = min(start + ppc, N)
            x_chunk = x_flat[start:end]

            def _chunk_forward(inp: torch.Tensor):
                mamba_outputs = []
                for ws in self.window_sizes:
                    windows = sliding_windows_gpu(inp, window_size=ws, stride=self.stride)
                    mamba_out = self.mamba_blocks[f'mamba_ws{ws}'](windows)
                    mamba_outputs.append(mamba_out.flatten(start_dim=1))
                concatenated_features = torch.cat(mamba_outputs, dim=1)
                return self.feature_projection(concatenated_features)

            if self.use_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                projected_chunk = checkpoint(_chunk_forward, x_chunk, use_reentrant=False)
            else:
                projected_chunk = _chunk_forward(x_chunk)
            pixel_features_chunks.append(projected_chunk)

        pixel_features = torch.cat(pixel_features_chunks, dim=0)
        pixel_features = self.norm(pixel_features)

        # 2. Reassemble into a 2D feature map
        # (B*H*W, D) -> (B, H, W, D) -> (B, D, H, W)
        feature_map = pixel_features.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # 3. Apply the tokenizer to create patch-based tokens
        # (B, D, H, W) -> (B, N_patches, D)
        spectral_tokens = self.tokenizer(feature_map)

        return spectral_tokens

