"""
Spatial Fusion Module for Hyperspectral Image Analysis.

This module intelligently fuses the feature maps from the Local and Global
spatial streams. The fusion process consists of three main stages:
1.  **Dynamic Weighting:** A Squeeze-and-Excitation style mechanism that
    learns to weigh the importance of local vs. global features for the
    entire image.
2.  **Pixel-wise Cross-Attention:** The dynamically weighted features are used
    as a query to attend to the global feature map, allowing the model to
    incorporate fine-grained global context into the local details.
3.  **Gated Fusion:** A final gating mechanism adaptively blends the
    dynamically weighted features and the attention-enhanced features to
    produce the final fused output.

A critical step is the initial upsampling of the global features to match the
spatial resolution of the local features before fusion.

Author: Nandhitha
Date: September 17, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialFusionModule(nn.Module):
    """
    Fuses local and global spatial features using dynamic weighting,
    cross-attention, and a final gating mechanism.
    """
    def __init__(self, channels: int, reduction: int = 8):
        """
        Args:
            channels (int): The number of channels in the input feature maps.
                            Both local and global streams must have the same channel count.
            reduction (int): The reduction ratio for the dynamic weighting block's
                             fully connected layers.
        """
        super().__init__()
        self.channels = channels

        # --- Dynamic Weighting Layers ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        # We concatenate two feature maps, so the input is 2 * channels
        self.fc1 = nn.Linear(2 * channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, 2, bias=False) # Outputs 2 weights

        # --- Cross-Attention Projection Layers ---
        # Note: Using Conv2d is often more efficient than Linear for image data
        self.wq = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.wk = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.wv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.attention_norm = nn.InstanceNorm2d(channels)

        # --- Gated Fusion Layers ---
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.final_norm = nn.BatchNorm2d(channels)

    def forward(self, f_local: torch.Tensor, f_global: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the SpatialFusionModule.
        Args:
            f_local (torch.Tensor): Feature map from the LocalFeatureStream.
                                    Shape: (B, C, H, W).
            f_global (torch.Tensor): Feature map from the GlobalFeatureStream.
                                     Shape: (B, C, H', W'), where H' < H.
        Returns:
            torch.Tensor: The final fused feature map of shape (B, C, H, W).
        """
        B, C, H, W = f_local.shape

        # 1. Upsample global features to match local feature dimensions
        f_global_upsampled = F.interpolate(f_global, size=(H, W), mode='bilinear', align_corners=False)

        # 2. Dynamic Weighting
        g = self.gap(torch.cat([f_local, f_global_upsampled], dim=1)).view(B, -1)
        weights = torch.softmax(self.fc2(F.relu(self.fc1(g))), dim=1)
        alpha, beta = weights[:, 0].view(B, 1, 1, 1), weights[:, 1].view(B, 1, 1, 1)
        f_dw = (alpha * f_local) + (beta * f_global_upsampled)

        # 3. Pixel-wise Cross-Attention
        # The dynamically weighted features (f_dw) act as the query.
        # The upsampled global features act as the key and value.
        q = self.wq(f_dw).view(B, C, -1)         # (B, C, H*W)
        k = self.wk(f_global_upsampled).view(B, C, -1)  # (B, C, H*W)
        v = self.wv(f_global_upsampled).view(B, C, -1)  # (B, C, H*W)

        # (B, C, H*W) @ (B, H*W, C) -> (B, C, C)
        attention_map = F.softmax(torch.bmm(q, k.transpose(1, 2)) * (C ** -0.5), dim=-1)
        # (B, C, C) @ (B, C, H*W) -> (B, C, H*W)
        f_attn = torch.bmm(attention_map, v).view(B, C, H, W)
        f_attn = self.attention_norm(f_attn) # Normalize attention output

        # 4. Gated Fusion Mechanism
        # The gate decides how much of f_attn to blend with f_dw
        gate = torch.sigmoid(self.gate_conv(f_attn))
        f_fused = (1 - gate) * f_dw + gate * f_attn
        f_fused = self.final_norm(f_fused)

        return f_fused
