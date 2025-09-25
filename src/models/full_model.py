"""
Full HSI Model - V2 (Patch-Based Architecture)

This module integrates the new patch-based streams and the simplified TCME.

CHANGES:
- The data flow is updated for the new patch-based SpectralStream and simplified TCME.
- The `spectral_stream` now directly outputs tokens.
- The `tcme` module takes the two token sets and outputs a single fused token tensor.
- `num_input_tokens` for the decoder is now calculated automatically based on the
  expected number of tokens from the concatenated output of TCME.
- The buggy HCMFF module remains bypassed.
"""

import torch
import torch.nn as nn
import math
from .spatial_stream import (
    LocalFeatureStream,
    GlobalFeatureStream,
    SpatialFusionModule,
    SpatialTokenizer
)
from .spectral_stream.main import SpectralStream
from .TCME.main import TokenCrossModalEnhancer
from .HCMFF.main import HierarchicalCrossModalityFrequencyFusion
from .decoder.main import MSTDHSHDecoder


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
                 num_classes: int = 6,
                 verbose: bool = False,
                 # Compatibility with training script flags (currently bypassed inside the model)
                 use_hcmff: bool = False,
                 hcmff_tokens: int = 256):
        super().__init__()
        self.verbose = verbose
        self.decoder_input_dim = spatial_embed_dim
        # Store flags for compatibility; current V2 pipeline does not route through HCMFF
        self.use_hcmff = use_hcmff
        self.hcmff_tokens = hcmff_tokens

        # --- Spatial Branch ---
        self.local_stream = LocalFeatureStream(num_bands)
        self.global_stream = GlobalFeatureStream(num_bands, embed_dim=128, patch_size=global_patch_size)
        self.fusion = SpatialFusionModule(channels=128)
        self.spatial_tokenizer = SpatialTokenizer(
            in_channels=128,
            embed_dim=spatial_embed_dim,
            patch_size=patch_size
        )
        
        # --- Spectral Branch (V2: Patch-Based) ---
        self.spectral_stream = SpectralStream(
            num_bands,
            embed_dim=spectral_embed_dim,
            patch_size=patch_size,
            window_sizes=spectral_window_sizes,
            stride=spectral_stride,
            pixels_per_chunk=spectral_pixels_per_chunk
        )
        
        # --- TCME (V2: Simplified) ---
        # We need to align embedding dimensions before TCME
        self._align_dims = (spatial_embed_dim != spectral_embed_dim)
        if self._align_dims:
            self.spec_proj = nn.Linear(spectral_embed_dim, spatial_embed_dim)
        self.tcme = TokenCrossModalEnhancer(dim=spatial_embed_dim, num_heads=8)
        
        self.decoder = None
        self.num_classes = num_classes
        self.patch_size = patch_size
        # --- Optional HCMFF path ---
        self.hcmff = None
        self.hcmff_token_proj = None  # lazy, projects sequence length N->K for HCMFF compute control
        if self.use_hcmff:
            self.hcmff = HierarchicalCrossModalityFrequencyFusion(feature_dim=spatial_embed_dim, verbose=verbose)

        if self.verbose:
            print("[Init] Model V2: Patch-Based Architecture")
            print(f"[Init] SpatialTokenizer ✓ (patch_size={patch_size})")
            print(f"[Init] SpectralStream V2 ✓ (patch_size={patch_size})")
            print("[Init] TCME V2 ✓ (Simplified Fusion)")
            print(f"[Init] Decoder ✓ (num_classes={num_classes}, will be initialized dynamically)")
            if self.use_hcmff:
                print("[Init] Note: use_hcmff flag provided but HCMFF path is bypassed in this V2 model.")


    def _should_log(self, x: torch.Tensor) -> bool:
        return self.verbose and ((x.device.type == 'cpu') or (getattr(x.device, 'index', 0) == 0))

    def forward(self, x):
        B, Bands, H, W = x.shape

        if self._should_log(x):
            print("\n" + "="*50)
            print(f"[FWD] Input: {tuple(x.shape)}")

        # --- 1. Spatial Stream -> Spatial Tokens ---
        f_local = self.local_stream(x)
        f_global = self.global_stream(x)
        f_fused_spatial = self.fusion(f_local, f_global)
        spatial_tokens = self.spatial_tokenizer(f_fused_spatial)
        
        # --- 2. Spectral Stream -> Spectral Tokens ---
        spectral_tokens = self.spectral_stream(x)

        if self._should_log(x):
            print("[Flow] Parallel streams computed and tokenized.")
            print(f"[FWD] Spatial Tokens -> {tuple(spatial_tokens.shape)}")
            print(f"[FWD] Spectral Tokens -> {tuple(spectral_tokens.shape)}")

        # --- 3. Fusion (HCMFF if enabled, otherwise TCME) ---
        if self._align_dims:
            spectral_tokens = self.spec_proj(spectral_tokens)
            if self._should_log(x):
                print(f"[Flow] Aligned spectral tokens to dim {self.decoder_input_dim}")

        if self.use_hcmff and (self.hcmff is not None):
            # Optional token compression to control HCMFF compute
            target_tokens = int(self.hcmff_tokens) if getattr(self, 'hcmff_tokens', None) else None
            def _compress_seq(tokens: torch.Tensor, K: int) -> torch.Tensor:
                # tokens: (B, N, D) -> (B, D, N) -> Linear(N->K) -> (B, K, D)
                if K is None or tokens.size(1) == K:
                    return tokens
                N = tokens.size(1)
                if (self.hcmff_token_proj is None) or (self.hcmff_token_proj.in_features != N) or (self.hcmff_token_proj.out_features != K):
                    self.hcmff_token_proj = nn.Linear(N, K).to(tokens.device)
                t = tokens.transpose(1, 2)
                t = self.hcmff_token_proj(t)
                return t.transpose(1, 2)

            if target_tokens is not None:
                spatial_for_fusion = _compress_seq(spatial_tokens, target_tokens)
                spectral_for_fusion = _compress_seq(spectral_tokens, target_tokens)
            else:
                spatial_for_fusion = spatial_tokens
                spectral_for_fusion = spectral_tokens

            if self._should_log(x):
                print(f"[FWD] HCMFF in Spatial -> {tuple(spatial_for_fusion.shape)} | Spectral -> {tuple(spectral_for_fusion.shape)}")
            fused_tokens = self.hcmff(spatial_for_fusion, spectral_for_fusion)
            if self._should_log(x):
                print("[Flow] HCMFF fusion complete.")
        else:
            fused_tokens = self.tcme(spatial_tokens, spectral_tokens)
            if self._should_log(x):
                print("[Flow] TCME fusion complete.")
        
        if self._should_log(x):
            print(f"[FWD] Fused Tokens for Decoder -> {tuple(fused_tokens.shape)}")
        
        # --- 4. Decoder (with dynamic initialization) ---
        if self.decoder is None:
            # First forward pass: inspect token shape and initialize the decoder
            num_fused_tokens = fused_tokens.shape[1]
            token_dim = fused_tokens.shape[2]
            
            self.decoder = MSTDHSHDecoder(
                input_token_dim=token_dim,
                num_classes=self.num_classes,
                num_input_tokens=num_fused_tokens
            ).to(fused_tokens.device)
            
            if self._should_log(x):
                print("[Flow] Dynamically initialized decoder.")
                print(f"       - Decoder expects tokens: {num_fused_tokens}")
                print(f"       - Token Dim: {token_dim}")

        seg_outputs = self.decoder(fused_tokens)
        
        if self._should_log(x):
            print("[Flow] Decoding complete.")
            print(f"[FWD] Decoder Final Logits -> {tuple(seg_outputs['final_logits'].shape)}")
            print("="*50 + "\n")

        return seg_outputs
