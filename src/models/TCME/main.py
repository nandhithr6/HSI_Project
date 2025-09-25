"""
TCME: Token Cross-Modal Enhancer - V2 (Simplified)

This module has been completely redesigned. With both spatial and spectral streams
now producing an equal number of patch-based tokens, the original TCME's complex
logic for handling token imbalance is no longer necessary.

This new version is much simpler and more direct:
1.  It receives spatial and spectral tokens of the same sequence length.
2.  It concatenates them along the sequence dimension.
3.  It applies a standard Transformer block (self-attention + MLP) to the
    concatenated sequence. This allows for powerful, all-to-all fusion
    between every spatial and spectral token.
4.  No token compression is performed, as the total number of tokens is manageable.

This design is more efficient and directly fuses the information from both modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    """A standard Transformer block with Pre-Normalization."""
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalization
        x = x + self.attn(*[self.norm1(x)]*3)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TokenCrossModalEnhancer(nn.Module):
    """
    End-to-end Token Cross-Modal Enhancer (V2).
    Fuses two token sequences via concatenation and a Transformer block.
    """
    def __init__(self, dim=256, num_heads=8, depth=1):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(depth)
        ])
        
    def forward(self, Tspatial: torch.Tensor, Tspectral: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Tspatial: (B, N, D) patch tokens from the spatial stream.
            Tspectral: (B, N, D) patch tokens from the spectral stream.
                       (Note: N must be the same for both).
        Returns:
            Fused tokens: (B, 2*N, D)
        """
        if Tspatial.shape[1] != Tspectral.shape[1]:
            raise ValueError(
                f"Spatial and Spectral token counts must match for this TCME version. "
                f"Got {Tspatial.shape[1]} and {Tspectral.shape[1]}."
            )

        # 1. Concatenate tokens along the sequence dimension
        fused_tokens = torch.cat([Tspatial, Tspectral], dim=1)

        # 2. Apply Transformer blocks for fusion
        for block in self.transformer_blocks:
            fused_tokens = block(fused_tokens)

        return fused_tokens