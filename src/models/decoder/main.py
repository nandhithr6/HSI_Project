"""
Multi-Scale Transformer Decoder (MSTD) + Hierarchical Segmentation Head (HSH) - CORRECTED
Complete implementation in single file

CHANGES:
- The `TokenToFeatureConverter` has been redesigned to correctly handle the concatenated
  fused tokens (e.g., 2800 tokens) from the upstream module.
- It no longer discards tokens. It now uses a linear projection to map the variable
  number of input tokens to the fixed number of tokens (1024) required to form the
  initial 32x32 feature grid for the decoder.
- Added `num_input_tokens` to the `__init__` to make this explicit.
- Removed optional `freq_features` argument as it's not used in the corrected pipeline.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Tuple, List, Optional, Dict
class _NoopCtx:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False

def _sdpa_ctx(enable_flash=True, enable_math=False, enable_mem_efficient=True):
    try:
        from torch.nn.attention import sdpa_kernel as _sdpa
        return _sdpa(enable_flash=enable_flash, enable_math=enable_math, enable_mem_efficient=enable_mem_efficient)
    except Exception:
        try:
            from torch.backends.cuda import sdp_kernel as _legacy
            return _legacy(enable_flash=enable_flash, enable_math=enable_math, enable_mem_efficient=enable_mem_efficient)
        except Exception:
            return _NoopCtx()

# =================== HELPER MODULES ===================

class TokenToFeatureConverter(nn.Module):
    """
    CORRECTED: Converts a sequence of fused tokens into a 2D feature map.
    It projects the variable-length input token sequence to a fixed length
    (spatial_size * spatial_size) and then reshapes it into a grid.
    """
    def __init__(self, input_dim: int, output_dim: int, num_input_tokens: int, spatial_size: int = 8):
        super().__init__()
        self.spatial_size = spatial_size
        self.target_num_tokens = spatial_size * spatial_size
        self.num_input_tokens = num_input_tokens

        # Project from variable input tokens to target number of tokens
        self.token_proj = nn.Linear(num_input_tokens, self.target_num_tokens)

        # Project feature dimension
        self.feature_proj = nn.Linear(input_dim, output_dim)
        
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.target_num_tokens, output_dim))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_tokens: [B, N_input, D_in] e.g., [B, 2800, 256]
        Returns:
            features: [B, 32, 32, D_out]
        """
        # 1. Project to target number of tokens.
        # (B, N_in, D) -> transpose -> (B, D, N_in)
        tokens_t = fused_tokens.transpose(1, 2)
        # (B, D, N_in) -> project -> (B, D, N_target)
        projected_tokens_t = self.token_proj(tokens_t)
        # (B, D, N_target) -> transpose -> (B, N_target, D)
        projected_tokens = projected_tokens_t.transpose(1, 2)

        # 2. Project feature dimension
        projected_features = self.feature_proj(projected_tokens)
        
        # 3. Add positional encoding
        with_pos = projected_features + self.pos_encoding
        normalized = self.norm(with_pos)
        
        # 4. Reshape to spatial grid
        B, _, D = normalized.shape
        features_2d = normalized.view(B, self.spatial_size, self.spatial_size, D)
        
        return features_2d.contiguous()


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Dropout(0.1)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_4d = x.dim() == 4
        if is_4d:
            B, H, W, C = x.shape
            x_flat = x.flatten(1, 2)
        else:
            x_flat = x
            
        with _sdpa_ctx(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            attn_out, _ = self.attention(x_flat, x_flat, x_flat, need_weights=False)
        x_flat = self.norm1(x_flat + attn_out)
        
        ff_out = self.feed_forward(x_flat)
        x_flat = self.norm2(x_flat + ff_out)
        
        if is_4d:
            return x_flat.view(B, H, W, C)
        return x_flat

class CrossScaleAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, prev_dim: Optional[int] = None):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.prev_proj = None
        if prev_dim is not None and prev_dim != dim:
            self.prev_proj = nn.Linear(prev_dim, dim)
        
    def forward(self, current_scale: torch.Tensor, prev_scale: torch.Tensor) -> torch.Tensor:
        B, H, W, C = current_scale.shape
        current_flat = current_scale.flatten(1, 2)
        
        prev_upsampled = F.interpolate(
            prev_scale.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=False
        ).permute(0, 2, 3, 1)
        prev_flat = prev_upsampled.flatten(1, 2)
        if self.prev_proj is not None:
            prev_flat = self.prev_proj(prev_flat)

        with _sdpa_ctx(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            attn_out, _ = self.attention(current_flat, prev_flat, prev_flat, need_weights=False)
        output = self.norm(current_flat + attn_out)
        
        return output.view(B, H, W, C)

class MultiScaleFusion(nn.Module):
    def __init__(self, num_scales: int = 3, feature_dim: int = 64):
        super().__init__()
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(feature_dim, 64),
            nn.ReLU(), nn.Linear(64, num_scales), nn.Softmax(dim=1)
        )
        
    def forward(self, scale_features: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        B = scale_features[0].shape[0]
        target_h, target_w = target_size
        
        resized_features = []
        for feat in scale_features:
            feat_permuted = feat.permute(0, 3, 1, 2)
            if feat_permuted.shape[2:] != (target_h, target_w):
                 resized = F.interpolate(feat_permuted, size=(target_h, target_w), mode='bilinear', align_corners=False)
            else:
                 resized = feat_permuted
            resized_features.append(resized)

        weights = self.weight_generator(resized_features[-1]).view(B, len(resized_features), 1, 1, 1)
        
        stacked_features = torch.stack(resized_features, dim=1)
        fused = torch.sum(weights * stacked_features, dim=1)
        return fused.permute(0, 2, 3, 1)

class HierarchicalSegmentationHead(nn.Module):
    def __init__(self, feature_dim: int = 32, num_classes: int = 6):
        super().__init__()
        self.main_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1), nn.BatchNorm2d(feature_dim),
            nn.GELU(), nn.Conv2d(feature_dim, num_classes, 1)
        )
        self.aux_head_256 = nn.Conv2d(64, num_classes, 1)
        self.aux_head_128 = nn.Conv2d(128, num_classes, 1)
        
    def forward(self, main_features: torch.Tensor, aux_features_256: torch.Tensor, aux_features_128: torch.Tensor) -> dict:
        main_logits = self.main_head(main_features)
        aux_logits_256 = self.aux_head_256(aux_features_256)
        aux_logits_128 = self.aux_head_128(aux_features_128)
        
        return {
            'main_logits': main_logits,
            'aux_logits_256': aux_logits_256,
            'aux_logits_128': aux_logits_128,
            'final_logits': main_logits,
        }

# =================== MAIN DECODER CLASS ===================

class MSTDHSHDecoder(nn.Module):
    def __init__(self, input_token_dim: int = 256, num_classes: int = 6, num_input_tokens: int = 2048, verbose: bool = False):
        super().__init__()
        
        self.token_converter = TokenToFeatureConverter(
            input_dim=input_token_dim, 
            output_dim=512, 
            num_input_tokens=num_input_tokens, 
            spatial_size=32
        )
        
        self.upsample1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.transformer1 = TransformerBlock(256, num_heads=8)
        
        self.upsample2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.cross_attn2 = CrossScaleAttention(128, num_heads=2, prev_dim=256)
        self.transformer2 = TransformerBlock(128, num_heads=8)
        
        self.upsample3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.multi_fusion3 = MultiScaleFusion(num_scales=3, feature_dim=64)
        self.transformer3 = TransformerBlock(64, num_heads=8)
        self.align_256_to_64 = nn.Conv2d(256, 64, 1)
        self.align_128_to_64 = nn.Conv2d(128, 64, 1)
        
        self.upsample4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.final_refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.GELU(), nn.Conv2d(32, 32, 1)
        )
        
        self.seg_head = HierarchicalSegmentationHead(feature_dim=32, num_classes=num_classes)
    
    def forward(self, fused_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        F0 = self.token_converter(fused_tokens).permute(0, 3, 1, 2)
        
        F1 = self.upsample1(F0)
        F1_refined = self.transformer1(F1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        F2 = self.upsample2(F1_refined)
        F2_cross = self.cross_attn2(F2.permute(0, 2, 3, 1), F1_refined.permute(0, 2, 3, 1))
        F2_final = self.transformer2(F2_cross).permute(0, 3, 1, 2)
        
        F3 = self.upsample3(F2_final)
        F1_aligned = self.align_256_to_64(F1_refined).permute(0, 2, 3, 1)
        F2_aligned = self.align_128_to_64(F2_final).permute(0, 2, 3, 1)
        scale_features_for_fusion = [
            F1_aligned,
            F2_aligned,
            F3.permute(0, 2, 3, 1)
        ]
        F3_fused = self.multi_fusion3(scale_features_for_fusion, target_size=(F3.shape[2], F3.shape[3]))
        F3_final = self.transformer3(F3_fused).permute(0, 3, 1, 2)
        
        F4 = self.upsample4(F3_final)
        F_output = self.final_refine(F4)
        
        return self.seg_head(
            main_features=F_output, aux_features_256=F3_final, aux_features_128=F2_final
        )
