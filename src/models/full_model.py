"""
Full HSI Model (Spatial + Spectral Streams up to Tokenization)

This module integrates:
- LocalFeatureStream (adaptive 3D CNNs + RSB + CBAM + 2D CNNs)
- GlobalFeatureStream (2D Mamba-based global context extractor)
- SpatialFusionModule (dynamic weighting + cross-attention + gating)
- SpatialTokenizer (patch embedding + 2D PE)
- SpectralStream (multi-scale windowed Mamba + 2D PE)

Outputs:
- Spatial tokens: (B, N_patches, D_spatial)
- Spectral tokens: (B, H*W, D_spectral)

These are ready to be passed into the Token Cross-Modal Enhancer (TCME).
"""


import torch
import torch.nn as nn
from src.models.spatial_stream import (
    LocalFeatureStream,
    GlobalFeatureStream,
    SpatialFusionModule,
    SpatialTokenizer
)
from src.models.spectral_stream import SpectralStream
from src.models.tcme import TokenCrossModalEnhancer

from src.models.hcmff import HierarchicalCrossModalityFrequencyFusion
from src.models.decoder.main import MSTDHSHDecoder


class HSIModel(nn.Module):
    def __init__(self, num_bands=37, spatial_embed_dim=256, spectral_embed_dim=128, patch_size=16):
        super().__init__()
        # --- Spatial Branch ---
        self.local_stream = LocalFeatureStream(num_bands)
        self.global_stream = GlobalFeatureStream(num_bands, embed_dim=spatial_embed_dim, patch_size=4)
        self.fusion = SpatialFusionModule(channels=128)  # local final channels fixed at 128
        self.spatial_tokenizer = SpatialTokenizer(
            in_channels=128,
            embed_dim=spatial_embed_dim,
            patch_size=patch_size
        )
        # --- Spectral Branch ---
        self.spectral_stream = SpectralStream(num_bands, embed_dim=spectral_embed_dim)
        # --- TCME ---
        self.tcme = TokenCrossModalEnhancer(
            dim=spatial_embed_dim,  # assumes spatial_embed_dim == spectral_embed_dim
            num_heads=8,
            N_pairs=5000,
            K_spatial=800,
            K_spectral=2000
        )
        # --- Fusion Block (HCMFF) ---
        self.hcmff = HierarchicalCrossModalityFrequencyFusion(feature_dim=spatial_embed_dim)

        # --- Decoder ---
        self.decoder = MSTDHSHDecoder(input_token_dim=spatial_embed_dim, num_classes=5)


    def forward(self, x):
        # x: (B, Bands, H, W)
        B, Bands, H, W = x.shape

        # --- Spatial: Local and Global run in parallel ---
        f_local = self.local_stream(x)  # (B, 128, H, W)
        f_global = self.global_stream(x)  # (B, spatial_embed_dim, H/4, W/4)

        # --- Upsample global to match local, fuse ---
        f_fused = self.fusion(f_local, f_global)  # (B, 128, H, W)

        # --- Tokenize fused spatial features ---
        spatial_tokens = self.spatial_tokenizer(f_fused)  # (B, N_patches, spatial_embed_dim)

        # --- Spectral stream (runs in parallel to spatial) ---
        spectral_map = self.spectral_stream(x)  # (B, spectral_embed_dim, H, W)
        spectral_tokens = spectral_map.permute(0, 2, 3, 1).contiguous().view(B, H * W, -1)

        # --- TCME ---
        T_spatial_sel, T_spectral_sel = self.tcme(spatial_tokens, spectral_tokens, H, W)

        # --- Fusion Block (HCMFF) ---
        fused_features = self.hcmff(T_spatial_sel, T_spectral_sel)

        # --- Decoder ---
        seg_outputs = self.decoder(fused_features)
        return seg_outputs
