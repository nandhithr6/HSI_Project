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
        self.fusion = SpatialFusionModule(channels=128)  # local final channels fixed at 128
        # Align global stream channels (spatial_embed_dim) to local channels (128) for fusion
        self.global_reduce = nn.Conv2d(spatial_embed_dim, 128, kernel_size=1, bias=False)
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
        self.decoder = MSTDHSHDecoder(input_token_dim=spatial_embed_dim, num_classes=num_classes)

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


    def forward(self, x):
        # x: (B, Bands, H, W)
        B, Bands, H, W = x.shape

        # --- Spatial: Local and Global run in parallel ---
        if self._should_log(x):
            print("[Flow] Starting parallel spatial (local+global) and spectral processing...")
            print(f"[FWD] Input: {tuple(x.shape)}")
        f_local = self.local_stream(x)  # (B, 128, H, W)
        f_global = self.global_stream(x)  # (B, spatial_embed_dim, H/patch, W/patch)
        if self._should_log(x):
            print("[Flow] Spatial branch parallel paths done: local and global features computed.")
            print(f"[FWD] Local -> {tuple(f_local.shape)} | Global -> {tuple(f_global.shape)}")
        f_global = self.global_reduce(f_global)  # (B, 128, H/4, W/4)
        if self._should_log(x):
            print("[Flow] Global branch aligned to local channels; proceeding to spatial fusion.")
            print(f"[FWD] Global reduced -> {tuple(f_global.shape)}")

        # --- Upsample global to match local, fuse ---
        f_fused = self.fusion(f_local, f_global)  # (B, 128, H, W)
        if self._should_log(x):
            print("[Flow] Spatial fusion executed; fused spatial features ready.")
            print(f"[FWD] Fused spatial -> {tuple(f_fused.shape)}")

        # --- Tokenize fused spatial features ---
        if self._should_log(x):
            print("[Flow] Preparing tokenization: spatial (patch-level) and spectral (pixel-level).")
        spatial_tokens = self.spatial_tokenizer(f_fused)  # (B, N_patches, spatial_embed_dim)
        if self._should_log(x):
            print(f"[FWD] Spatial patch tokens -> {tuple(spatial_tokens.shape)}")

        # --- Spectral stream (runs in parallel to spatial) ---
        spectral_map = self.spectral_stream(x)  # (B, spectral_embed_dim, H, W)
        spectral_tokens = spectral_map.permute(0, 2, 3, 1).contiguous().view(B, H * W, -1)
        if self._needs_spec_proj:
            spectral_tokens = self.spectral_to_spatial(spectral_tokens)
        if self._should_log(x):
            print("[Flow] Spectral pixel-level tokenization complete in parallel with spatial.")
            print(f"[FWD] Spectral map -> {tuple(spectral_map.shape)} | Spectral tokens -> {tuple(spectral_tokens.shape)}")

        # --- TCME ---
        if self._should_log(x):
            try:
                K_sp = self.tcme.compressor.K_spatial
                K_spec = self.tcme.compressor.K_spectral
            except Exception:
                K_sp, K_spec = None, None
            if K_sp is not None and K_spec is not None and K_sp > 0 and K_spec > 0:
                g = math.gcd(int(K_sp), int(K_spec))
                simple = (K_sp//g, K_spec//g) if g > 0 else (K_sp, K_spec)
                ratio = (K_sp / float(K_spec))
                print("[Flow] TCME: joint pairing in progress; compressing tokens to balance spatial and spectral.")
                print(f"[Flow] Sweet point ratio K_spatial:K_spectral = {K_sp}:{K_spec} (~{ratio:.2f}:1), simplified {simple[0]}:{simple[1]}")
            else:
                print("[Flow] TCME: joint pairing + token compression to balance modalities.")
        T_spatial_sel, T_spectral_sel = self.tcme(spatial_tokens, spectral_tokens, H, W)
        if self._should_log(x):
            print(f"[FWD] TCME output -> spatial {tuple(T_spatial_sel.shape)}, spectral {tuple(T_spectral_sel.shape)}")

        # --- Fusion Block (HCMFF) ---
        if self._should_log(x):
            print("[Flow] Passing tokens to HCMFF for frequency-domain fusion.")
        fused_features = self.hcmff(T_spatial_sel, T_spectral_sel)
        if self._should_log(x):
            print(f"[FWD] HCMFF fused features -> {tuple(fused_features.shape)}")

        # --- Decoder ---
        seg_outputs = self.decoder(fused_features)
        if self._should_log(x):
            print("[Flow] Decoding fused representation to segmentation logits.")
            print(f"[FWD] Decoder outputs -> keys: {list(seg_outputs.keys())}")
        return seg_outputs
