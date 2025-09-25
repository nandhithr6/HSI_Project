"""
HCMFF (stabilized): Frequency-domain fusion with fp32 FFT, amplitude normalization,
and nn.MultiheadAttention for cross-modal phase alignment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


def _gate_print(t: torch.Tensor) -> bool:
    return (t.device.type == 'cpu') or (getattr(t.device, 'index', 0) == 0)


class FrequencyDomainTransforms(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_hw = 16  # 16x16 -> 256 tokens

    @torch.no_grad()
    def _norm_amp(self, amp: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Normalize amplitude per-sample across tokens to improve stability
        m = amp.amax(dim=(1, 2), keepdim=True)
        m = torch.where(m == 0, torch.ones_like(m), m)
        return amp / (m + eps)

    def spatial_to_frequency(self, spatial_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = spatial_tokens.shape
        if N == self.grid_hw * self.grid_hw:
            x = spatial_tokens.view(B, self.grid_hw, self.grid_hw, D).to(torch.float32)
            f = torch.fft.fft2(x, dim=(1, 2))
            amp = self._norm_amp(f.abs())
            ph = f.angle()
            return amp, ph
        else:
            x = spatial_tokens.to(torch.float32)
            f = torch.fft.fft(x, dim=1)
            amp = self._norm_amp(f.abs())
            ph = f.angle()
            return amp, ph

    def spectral_to_frequency(self, spectral_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = spectral_tokens.to(torch.float32)
        f = torch.fft.fft(x, dim=1)
        amp = self._norm_amp(f.abs())
        ph = f.angle()
        return amp, ph

    def frequency_to_spatial_reconstruction(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        comp = amplitude.to(torch.float32) * torch.exp(1j * phase.to(torch.float32))
        B, N, D = comp.shape
        if N == self.grid_hw * self.grid_hw:
            c2d = comp.view(B, self.grid_hw, self.grid_hw, D)
            x = torch.fft.ifft2(c2d, dim=(1, 2)).real
            return x.view(B, N, D).to(amplitude.dtype)
        else:
            # If not a square grid, interpolate sequence back to 256 tokens
            seq = torch.fft.ifft(comp, dim=1).real
            if seq.size(1) != self.grid_hw * self.grid_hw:
                seq = F.interpolate(seq.transpose(1, 2), size=self.grid_hw * self.grid_hw, mode='linear', align_corners=False).transpose(1, 2)
            return seq.to(amplitude.dtype)


class CrossModalFrequencyFusionCore(nn.Module):
    def __init__(self, feature_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_w = nn.Parameter(torch.tensor(1.0))
        self.spectral_w = nn.Parameter(torch.tensor(1.0))
        self.mha = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.ln_phase = nn.LayerNorm(feature_dim)
        self.amp_ln = nn.LayerNorm(feature_dim)
        self.low = nn.Sequential(nn.Conv1d(feature_dim, feature_dim, 7, padding=3), nn.GELU())
        self.high = nn.Sequential(nn.Conv1d(feature_dim, feature_dim, 3, padding=1), nn.GELU())

    def fuse_amplitude(self, a_sp: torch.Tensor, a_spec: torch.Tensor) -> torch.Tensor:
        if a_sp.dim() == 4:
            a_sp = a_sp.flatten(1, 2)
        if a_sp.size(1) != a_spec.size(1):
            a_sp = F.interpolate(a_sp.transpose(1, 2), size=a_spec.size(1), mode='linear', align_corners=False).transpose(1, 2)
        fused = self.spatial_w * a_sp + self.spectral_w * a_spec
        return self.amp_ln(fused)

    def align_phase(self, p_sp: torch.Tensor, p_spec: torch.Tensor) -> torch.Tensor:
        if p_sp.dim() == 4:
            p_sp = p_sp.flatten(1, 2)
        if p_sp.size(1) != p_spec.size(1):
            p_sp = F.interpolate(p_sp.transpose(1, 2), size=p_spec.size(1), mode='linear', align_corners=False).transpose(1, 2)
        with torch.cuda.amp.autocast(enabled=False):
            q = p_sp.to(torch.float32)
            k = p_spec.to(torch.float32)
            v = p_spec.to(torch.float32)
            out, _ = self.mha(q, k, v, need_weights=False)
        return self.ln_phase(out.to(p_sp.dtype))

    def split_freq(self, fused_amp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = fused_amp.transpose(1, 2)
        with torch.cuda.amp.autocast(enabled=False):
            lo = self.low(x.float()).transpose(1, 2)
            hi = self.high(x.float()).transpose(1, 2)
        return lo.to(fused_amp.dtype), hi.to(fused_amp.dtype)

    def forward(self, a_sp: torch.Tensor, p_sp: torch.Tensor, a_spec: torch.Tensor, p_spec: torch.Tensor):
        fused_amp = self.fuse_amplitude(a_sp, a_spec)
        aligned_phase = self.align_phase(p_sp, p_spec)
        low, high = self.split_freq(fused_amp)
        return fused_amp, aligned_phase, low, high


class HierarchicalMultiScaleProcessor(nn.Module):
    def __init__(self, feature_dim: int = 128, scales: List[int] = [1, 2, 4, 8], verbose: bool = False):
        super().__init__()
        self.scales = scales
        self.verbose = verbose
        self.blocks = nn.ModuleDict({
            str(s): nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim, 3, padding=1), nn.GELU(), nn.Conv1d(feature_dim, feature_dim, 1)
            ) for s in scales
        })
        self.scale_w = nn.Parameter(torch.ones(len(scales)))

    def _proc(self, s: int, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        if s == 1:
            t = lo + hi
        elif s == 2:
            t = F.avg_pool1d(lo.transpose(1, 2), 2, stride=1, padding=1).transpose(1, 2) + 0.5 * F.avg_pool1d(hi.transpose(1, 2), 2, stride=1, padding=1).transpose(1, 2)
        elif s == 4:
            t = F.avg_pool1d(lo.transpose(1, 2), 4, stride=1, padding=2).transpose(1, 2) + 0.25 * F.avg_pool1d(hi.transpose(1, 2), 4, stride=1, padding=2).transpose(1, 2)
        else:
            t = F.avg_pool1d(lo.transpose(1, 2), 8, stride=1, padding=4).transpose(1, 2)
        y = self.blocks[str(s)](t.transpose(1, 2)).transpose(1, 2)
        return y

    def forward(self, fused_amp: torch.Tensor, aligned_phase: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        feats = [self._proc(s, lo, hi) for s in self.scales]
        for i in range(len(feats)):
            if feats[i].size(1) != fused_amp.size(1):
                feats[i] = F.interpolate(feats[i].transpose(1, 2), size=fused_amp.size(1), mode='linear', align_corners=False).transpose(1, 2)
        w = F.softmax(self.scale_w, dim=0).view(-1, 1, 1, 1)
        stacked = torch.stack(feats, dim=0)
        out = (w * stacked).sum(dim=0)
        if self.verbose and _gate_print(out):
            print(f"[HCMFF] HMS out: {tuple(out.shape)}")
        return out


class HierarchicalCrossModalityFrequencyFusion(nn.Module):
    def __init__(self, feature_dim: int = 128, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.fft = FrequencyDomainTransforms(feature_dim)
        self.core = CrossModalFrequencyFusionCore(feature_dim)
        self.hms = HierarchicalMultiScaleProcessor(feature_dim, verbose=verbose)

    def forward(self, spatial_tokens: torch.Tensor, spectral_tokens: torch.Tensor) -> torch.Tensor:
        if self.verbose and _gate_print(spatial_tokens):
            print(f"[HCMFF] in spatial={tuple(spatial_tokens.shape)} spectral={tuple(spectral_tokens.shape)}")
        a_sp, p_sp = self.fft.spatial_to_frequency(spatial_tokens)
        a_spec, p_spec = self.fft.spectral_to_frequency(spectral_tokens)
        fused_amp, aligned_phase, lo, hi = self.core(a_sp, p_sp, a_spec, p_spec)
        fused = self.hms(fused_amp, aligned_phase, lo, hi)
        out = self.fft.frequency_to_spatial_reconstruction(fused, aligned_phase)
        return out