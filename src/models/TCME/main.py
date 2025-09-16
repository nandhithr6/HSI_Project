"""
TCME: Token Cross-Modal Enhancer 
A multi-scale transformer block for spatial–spectral token fusion and compression
Author: Nandhitha  
Date: September 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# -------------------------------------------------
# Multi-Scale Token Division
# -------------------------------------------------
class MultiScaleDivider(nn.Module):
    """
    Generate multi-resolution token sets (scales = {1, 2, 4}) by average pooling.
    - Input: tokens (B, N, D), reshaped into (H, W) grid
    - Output: dict of scale -> downsampled tokens
    """
    def __init__(self, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales

    def forward(self, tokens, H, W):
        """
        Args:
            tokens: (B, N, D) where N = H*W (spectral) or N_patches (spatial)
            H, W: spatial dimensions of the grid
        Returns:
            dict {scale: (B, N_scale, D)}
        """
        B, N, D = tokens.shape
        out = {}

        # reshape tokens back to image grid (B, D, H, W)
        grid = tokens.view(B, H, W, D).permute(0, 3, 1, 2)

        for s in self.scales:
            if s == 1:
                out[s] = tokens  # keep original
            else:
                # average pooling by factor s
                pooled = F.avg_pool2d(grid, kernel_size=s, stride=s)
                Hs, Ws = pooled.shape[2], pooled.shape[3]
                # flatten back into sequence of tokens
                out[s] = pooled.flatten(2).transpose(1, 2)  # (B, N//s^2, D)

        return out


# -------------------------------------------------
# Attention Block (Self + Cross Modal)
# -------------------------------------------------
class CrossModalAttentionBlock(nn.Module):
    """
    Performs both:
    - Self-attention: within spatial tokens and spectral tokens
    - Cross-attention: spatial->spectral and spectral->spatial
    Uses FlashAttention kernels (via scaled_dot_product_attention).
    """
    def __init__(self, dim, num_heads=8, use_checkpoint=False):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # projection layers for self-attention
        self.qkv_spatial = nn.Linear(dim, dim * 3, bias=False)
        self.qkv_spectral = nn.Linear(dim, dim * 3, bias=False)

        # projection layers for cross-attention
        self.q_spatial = nn.Linear(dim, dim, bias=False)
        self.kv_spectral = nn.Linear(dim, dim * 2, bias=False)

        self.q_spectral = nn.Linear(dim, dim, bias=False)
        self.kv_spatial = nn.Linear(dim, dim * 2, bias=False)

        self.out_proj = nn.Linear(dim, dim)

    def _flash_attention(self, q, k, v):
        """
        Wrapper for PyTorch 2.5+ scaled_dot_product_attention,
        which automatically uses FlashAttention kernels on GPU.
        """
        return F.scaled_dot_product_attention(q, k, v)

    def forward_fn(self, Tspatial, Tspectral):
        """
        Args:
            Tspatial: (B, N1, D)  spatial tokens
            Tspectral: (B, N2, D) spectral tokens
        Returns:
            SA_spatial, SA_spectral, CA_s2sp, CA_sp2s
        """
        B, N1, D = Tspatial.shape
        _, N2, _ = Tspectral.shape

        # --- Self-Attention (spatial)
        qkv = self.qkv_spatial(Tspatial).reshape(B, N1, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.unbind(2)
        SA_spatial = self._flash_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        SA_spatial = SA_spatial.transpose(1, 2).reshape(B, N1, D)

        # --- Self-Attention (spectral)
        qkv = self.qkv_spectral(Tspectral).reshape(B, N2, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.unbind(2)
        SA_spectral = self._flash_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        SA_spectral = SA_spectral.transpose(1, 2).reshape(B, N2, D)

        # --- Cross Attention Spatial -> Spectral
        q = self.q_spatial(Tspatial).reshape(B, N1, self.num_heads, D // self.num_heads).transpose(1, 2)
        kv = self.kv_spectral(Tspectral).reshape(B, N2, 2, self.num_heads, D // self.num_heads)
        k, v = kv[:, :, 0], kv[:, :, 1]
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        CA_s2sp = self._flash_attention(q, k, v).transpose(1, 2).reshape(B, N1, D)

        # --- Cross Attention Spectral -> Spatial
        q = self.q_spectral(Tspectral).reshape(B, N2, self.num_heads, D // self.num_heads).transpose(1, 2)
        kv = self.kv_spatial(Tspatial).reshape(B, N1, 2, self.num_heads, D // self.num_heads)
        k, v = kv[:, :, 0], kv[:, :, 1]
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        CA_sp2s = self._flash_attention(q, k, v).transpose(1, 2).reshape(B, N2, D)

        return SA_spatial, SA_spectral, CA_s2sp, CA_sp2s

    def forward(self, Tspatial, Tspectral):
        if self.use_checkpoint:
            return checkpoint(self.forward_fn, Tspatial, Tspectral, use_reentrant=False)
        else:
            return self.forward_fn(Tspatial, Tspectral)


# -------------------------------------------------
# Token Scoring (multi-criteria)
# -------------------------------------------------
class TokenScorer(nn.Module):
    """
    Compute importance scores per token based on:
    - Self-attention weights
    - Cross-attention weights
    - Value vector norms
    Combined multiplicatively with learnable temperature parameters.
    """
    def __init__(self, dim):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(5))  # τk params

    def forward(self, SA_spatial, SA_spectral, CA_s2sp, CA_sp2s,
                values_spatial, values_spectral):
        """
        Args:
            SA_spatial: (B, N1, D)
            SA_spectral: (B, N2, D)
            CA_s2sp: (B, N1, D)
            CA_sp2s: (B, N2, D)
            values_spatial: original spatial tokens (B, N1, D)
            values_spectral: original spectral tokens (B, N2, D)
        Returns:
            Score_spatial: (B, N1)
            Score_spectral: (B, N2)
        """
        # attention aggregation
        S_spatial = SA_spatial.abs().sum(dim=-1)
        S_spectral = SA_spectral.abs().sum(dim=-1)
        S_s2sp = CA_s2sp.abs().sum(dim=-1)
        S_sp2s = CA_sp2s.abs().sum(dim=-1)

        # value vector norm
        Vnorm_spatial = values_spatial.norm(dim=-1)
        Vnorm_spectral = values_spectral.norm(dim=-1)

        τ = torch.clamp(self.temperatures, 0.1, 10.0)

        Score_spatial = (S_spatial ** τ[0]) * (S_s2sp ** τ[2]) * (Vnorm_spatial ** τ[4])
        Score_spectral = (S_spectral ** τ[1]) * (S_sp2s ** τ[3]) * (Vnorm_spectral ** τ[4])

        return Score_spatial, Score_spectral


# -------------------------------------------------
# Token Compression
# -------------------------------------------------
class TokenCompressor(nn.Module):
    """
    Two-stage compression strategy simplified into Top-K selection:
    - Stage 1 (pair-constrained selection) can be added later
    - Stage 2 (Top-K within subsets) implemented directly here
    """
    def __init__(self, N_pairs=5000, K_spatial=800, K_spectral=2000):
        super().__init__()
        self.N_pairs = N_pairs
        self.K_spatial = K_spatial
        self.K_spectral = K_spectral

    def forward(self, Score_spatial, Score_spectral, Tspatial, Tspectral):
        # --- Top-K spatial tokens
        topk_sp = torch.topk(Score_spatial, k=min(self.K_spatial, Score_spatial.size(1)), dim=1)
        idx_sp = topk_sp.indices
        T_spatial_sel = torch.gather(Tspatial, 1, idx_sp.unsqueeze(-1).expand(-1, -1, Tspatial.size(-1)))

        # --- Top-K spectral tokens
        topk_spec = torch.topk(Score_spectral, k=min(self.K_spectral, Score_spectral.size(1)), dim=1)
        idx_spec = topk_spec.indices
        T_spectral_sel = torch.gather(Tspectral, 1, idx_spec.unsqueeze(-1).expand(-1, -1, Tspectral.size(-1)))

        return T_spatial_sel, T_spectral_sel


# -------------------------------------------------
# Full TCME Module
# -------------------------------------------------
class TokenCrossModalEnhancer(nn.Module):
    """
    End-to-end Token Cross-Modal Enhancer:
    1. Multi-scale division of tokens
    2. Self- and cross-attention
    3. Token scoring
    4. Token compression (Top-K)
    """
    def __init__(self, dim=256, num_heads=8,
                 N_pairs=5000, K_spatial=800, K_spectral=2000,
                 use_checkpoint=False):
        super().__init__()
        self.divider = MultiScaleDivider()
        self.attn = CrossModalAttentionBlock(dim, num_heads, use_checkpoint)
        self.scorer = TokenScorer(dim)
        self.compressor = TokenCompressor(N_pairs, K_spatial, K_spectral)

    def forward(self, Tspatial, Tspectral, H, W):
        """
        Args:
            Tspatial: (B, Np, D) patch tokens
            Tspectral: (B, H*W, D) pixel tokens
            H, W: spatial dimensions (pixels)
        Returns:
            Compressed tokens:
              - Spatial: (B, K_spatial, D)
              - Spectral: (B, K_spectral, D)
        """
        # Multi-scale division (only scale=1 used for now)
        # infer grid size from number of spatial tokens
        B, Np, D = Tspatial.shape
        H_patch = int((Np) ** 0.5)   # assume square patch grid
        W_patch = H_patch

        spatial_scales = self.divider(Tspatial, H_patch, W_patch)
        spectral_scales = self.divider(Tspectral, H, W)

        # Cross-modal attention at fine scale
        SA_spatial, SA_spectral, CA_s2sp, CA_sp2s = self.attn(spatial_scales[1], spectral_scales[1])

        # Token importance scores
        Score_spatial, Score_spectral = self.scorer(
            SA_spatial, SA_spectral, CA_s2sp, CA_sp2s,
            spatial_scales[1], spectral_scales[1]
        )

        # Token compression
        T_spatial_sel, T_spectral_sel = self.compressor(
            Score_spatial, Score_spectral,
            spatial_scales[1], spectral_scales[1]
        )

        return T_spatial_sel, T_spectral_sel