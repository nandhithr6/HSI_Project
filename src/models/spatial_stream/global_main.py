"""
2DVMamba for Global Context Extraction of HSI Data
Author: Nandhitha
Date: September 16, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# 2DVMamba Block
# ---------------------------
class Mamba2D(nn.Module):
    def __init__(self, channels, state_dim=16):
        super().__init__()
        self.channels = channels
        self.state_dim = state_dim

        self.A = nn.Parameter(torch.randn(state_dim, state_dim))
        self.B = nn.Linear(channels, state_dim, bias=False)
        self.C = nn.Linear(state_dim, channels, bias=False)

    def forward_scan(self, x):
        # Row-wise then column-wise selective scan
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

        # Flatten for easier GPU block operations
        x = x.view(B * H, W, C)

        h = torch.zeros(B * H, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(W):
            h = torch.matmul(h, self.A.T) + self.B(x[:, t, :])
            y = self.C(h)
            outputs.append(y)
        y = torch.stack(outputs, dim=1)  # (B*H, W, C)

        return y.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        out = self.forward_scan(x)  # row-wise scan
        return out  # already (B, C, H, W)



# ---------------------------
# Global Stream
# ---------------------------
class GlobalStream2DVMamba(nn.Module):
    def __init__(self, in_channels, out_channels=256, state_dim=16):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mamba = Mamba2D(out_channels, state_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn(self.conv1x1(x)), negative_slope=0.01, inplace=True)
        x = x.contiguous(memory_format=torch.channels_last)  # enforce
        return self.mamba(x)
