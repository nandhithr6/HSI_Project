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
import math
from typing import Dict
from src.models.spatial_stream import (
    LocalFeatureStream,
    GlobalFeatureStream,
    SpatialFusionModule,
    SpatialTokenizer
)
from src.models.spectral_stream import SpectralStream
from src.models.TCME import TokenCrossModalEnhancer

from src.models.HCMFF import HierarchicalCrossModalityFrequencyFusion
from src.models.decoder.main import MSTDHSHDecoder


class HSIModel(nn.Module):
    def __init__(self,
                 num_bands: int = 37,
                 spatial_embed_dim: int = 256,
                 spectral_embed_dim: int = 128,
                 patch_size: int = 16,
                 global_patch_size: int = 4,
                 spectral_window_sizes: list = [8, 16, 32],
                 spectral_stride: int = 4,
                 spectral_pixels_per_chunk: int = 8192,
                 num_classes: int = 5,
                 verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        # --- Spatial Branch ---
        self.local_stream = LocalFeatureStream(num_bands)
        self.global_stream = GlobalFeatureStream(num_bands, embed_dim=spatial_embed_dim, patch_size=global_patch_size)
        self.spatial_fusion = SpatialFusionModule(channels=128)
        self.spatial_fusion_align = nn.Conv2d(spatial_embed_dim, 128, kernel_size=1, bias=False)
        self.spatial_tokenizer = SpatialTokenizer(
            in_channels=128,
            embed_dim=spatial_embed_dim,
            patch_size=patch_size
        )
        # --- Spectral Branch ---
        self.spectral_stream = SpectralStream(
            num_bands,
            embed_dim=spectral_embed_dim,
            window_sizes=spectral_window_sizes,
            stride=spectral_stride,
            pixels_per_chunk=spectral_pixels_per_chunk
        )
        # --- TCME ---
        self.tcme = TokenCrossModalEnhancer(
            dim=spatial_embed_dim,  # assumes spatial_embed_dim == spectral_embed_dim
            num_heads=8,
            N_pairs=5000,
            K_spatial=800,
            K_spectral=2000
        )
        # Align spectral token dimension to spatial if they differ
        self._needs_spec_proj = (spectral_embed_dim != spatial_embed_dim)
        if self._needs_spec_proj:
            self.spectral_to_spatial = nn.Linear(spectral_embed_dim, spatial_embed_dim)
        # --- Fusion Block (HCMFF) ---
        self.hcmff = HierarchicalCrossModalityFrequencyFusion(feature_dim=spatial_embed_dim, verbose=verbose)

        # --- Decoder ---
        self.decoder = MSTDHSHDecoder(input_token_dim=spatial_embed_dim, num_classes=num_classes, verbose=verbose)

        if self.verbose:
            print("[Init] Spatial branch initialized: Local + Global will run in parallel.")
            print(f"[Init] GlobalFeatureStream ✓ (embed_dim={spatial_embed_dim}, patch={global_patch_size})")
            print("[Init] SpatialFusionModule ✓ (channels=128)")
            print(f"[Init] SpatialTokenizer ✓ (patch_size={patch_size}, embed_dim={spatial_embed_dim})")
            print(f"[Init] Spectral branch initialized ✓ (embed_dim={spectral_embed_dim}, windows={spectral_window_sizes}, stride={spectral_stride})")
            print("[Init] TCME ✓ (joint pairing + token compression)")
            print("[Init] HCMFF ✓ (frequency-domain fusion)")
            print(f"[Init] Decoder ✓ (num_classes={num_classes})")

    def _should_log(self, x: torch.Tensor) -> bool:
        if not self.verbose:
            return False
        dev = x.device
        # Log on CPU or only on GPU 0 to avoid duplicate prints under DataParallel
        return (dev.type == 'cpu') or (getattr(dev, 'index', None) in (None, 0))


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the complete HSI model.

        Args:
            x (torch.Tensor): Input HSI data cube of shape (B, Bands, H, W).

        Returns:
            Dict[str, torch.Tensor]: Dictionary of output tensors, including logits.
        """
        # Ensure contiguous memory before entering each major module to prevent misaligned address errors
        # under DataParallel and AMP.
        x_contig = x.contiguous()

        # 1. Parallel Spatial and Spectral Processing
        if self.verbose: print("[Flow] Starting parallel spatial (local+global) and spectral processing...")
        if self.verbose: print(f"[FWD] Input: {tuple(x_contig.shape)}")

        # --- Spatial Branches ---
        with torch.cuda.amp.autocast(enabled=False):
            f_local = self.local_stream(x_contig.float())
        f_global = self.global_stream(x_contig)
        if self.verbose: print(f"[FWD] Local -> {tuple(f_local.shape)} | Global -> {tuple(f_global.shape)}")

        # --- Spectral Branch ---
        # The spectral stream now returns a map, not tokens. We flatten it for TCME.
        spectral_map = self.spectral_stream(x_contig)
        B, D, H, W = spectral_map.shape
        T_spectral = spectral_map.permute(0, 2, 3, 1).contiguous().view(B, H * W, D)
        if self.verbose: print(f"[Tokens] Spectral: N={T_spectral.shape[1]} D={T_spectral.shape[2]} | map={tuple(spectral_map.shape)}")

        # 2. Spatial Fusion and Tokenization
        if self.verbose: print("[Flow] Global branch aligned to local channels; proceeding to spatial fusion.")
        f_global_aligned = self.spatial_fusion_align(f_global.contiguous())
        if self.verbose: print(f"[FWD] Global reduced -> {tuple(f_global_aligned.shape)}")

        f_fused = self.spatial_fusion(f_local.contiguous(), f_global_aligned)
        if self.verbose: print("[Flow] Spatial fusion executed; fused spatial features ready.")
        if self.verbose: print(f"[FWD] Fused spatial -> {tuple(f_fused.shape)}")

        if self.verbose: print("[Flow] Preparing tokenization: spatial (patch-level) and spectral (pixel-level).")
        # Force float32 for the tokenizer to prevent CUBLAS errors with mixed precision.
        with torch.cuda.amp.autocast(enabled=False):
            T_spatial = self.spatial_tokenizer(f_fused.contiguous().float())
        if self.verbose: print(f"[Tokens] Spatial: N={T_spatial.shape[1]} D={T_spatial.shape[2]} | grid={self.spatial_tokenizer.grid_size} | patch={self.spatial_tokenizer.patch_size}")

        # Align spectral token dimension to spatial before TCME
        if self._needs_spec_proj:
            T_spectral = self.spectral_to_spatial(T_spectral.contiguous())
            if self.verbose: print(f"[Flow] Spectral tokens projected to match spatial dim -> {tuple(T_spectral.shape)}")

        # 3. Cross-Modal Token Enhancement (TCME)
        if self.verbose: print("[Flow] TCME: joint pairing in progress; compressing tokens to balance spatial and spectral.")
        T_spatial_sel, T_spectral_sel = self.tcme(T_spatial.contiguous(), T_spectral.contiguous(), H, W)
        if self.verbose: print(f"[FWD] TCME output -> spatial {tuple(T_spatial_sel.shape)}, spectral {tuple(T_spectral_sel.shape)}")

        # 4. Hierarchical Cross-Modal Frequency Fusion (HCMFF)
        if self.verbose: print("[Flow] Passing tokens to HCMFF for frequency-domain fusion.")
        # Ensure inputs to FFT are float32 and contiguous
        fused_features = self.hcmff(T_spatial_sel.contiguous().float(), T_spectral_sel.contiguous().float())
        if self.verbose: print(f"[FWD] HCMFF fused features -> {tuple(fused_features.shape)}")

        # 5. Decoder
        if self.verbose: print("[Flow] Decoding fused representation to segmentation logits.")
        # The decoder path is sensitive; ensure input is contiguous and disable cuDNN for the final head
        with torch.backends.cudnn.flags(enabled=False):
            outputs = self.decoder(fused_features.contiguous())
        if self.verbose: print(f"[FWD] Decoder outputs -> keys: {list(outputs.keys())}")

        return outputs
