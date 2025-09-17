"""
HSI Full Model: Combines Spatial and Spectral Streams up to Tokenization
Author:
Date: September 17, 2025
"""

import torch
import torch.nn as nn
from src.models.spatial_stream import LocalFeatureStream, SpatialTokenizer
from src.models.spectral_stream import SpectralStream

class HSIModel(nn.Module):
    def __init__(self, num_bands=37, spatial_patch_size=16, spatial_embed_dim=128, spectral_embed_dim=128):
        super().__init__()
        # Spatial stream
        self.local_stream = LocalFeatureStream(num_bands)
        self.spatial_tokenizer = SpatialTokenizer(
            in_channels=128,  # final channels from local_stream
            embed_dim=spatial_embed_dim,
            patch_size=spatial_patch_size
        )
        # Spectral stream
        self.spectral_stream = SpectralStream(num_bands, embed_dim=spectral_embed_dim)

    def forward(self, x):
        # x: (B, Bands, H, W)
        spatial_feat = self.local_stream(x)  # (B, 128, H, W)
        spatial_tokens = self.spatial_tokenizer(spatial_feat)  # (B, N_patches, spatial_embed_dim)

        spectral_feat = self.spectral_stream(x)  # (B, spectral_embed_dim, H, W)
        # You can add spectral tokenization here if needed

        return spatial_tokens, spectral_feat
