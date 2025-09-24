"""
Hierarchical Cross-Modality Frequency Fusion (HCMFF) - NO ENHANCEMENT VERSION
Author: Abiram
Date: September 2025

HCMFF takes compressed tokens from TokenCrossModalEnhancer and does direct FFT fusion.
NO ENHANCEMENT - Direct frequency processing only!
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Global one-shot logging registry to prevent repeated prints across DP replicas
# and re-entrant forwards (e.g., from gradient checkpointing or retries).
_HCMFF_GLOBAL_PRINT = {
    'hcmff_header_printed': False,
    'hms_header_printed': False,
    'hms_body_printed': False,
}


def _should_log_tensor(t: torch.Tensor) -> bool:
    """Gate prints based on the tensor's device: CPU or CUDA:0 only.
    This avoids duplicate logs across DataParallel replicas.
    """
    try:
        dev = t.device
        if dev.type != 'cuda':
            return True
        return getattr(dev, 'index', None) in (None, 0)
    except Exception:
        return True


# ================================================================================================
# FREQUENCY DOMAIN TRANSFORMS - DIRECT FFT ONLY
# ================================================================================================
class FrequencyDomainTransforms(nn.Module):
    """Direct FFT operations with NO enhancement - just conversion."""
    
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_h = self.spatial_w = 16  # default 16x16 grid when applicable (N==256)
        
    def spatial_to_frequency(self, spatial_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Direct spatial FFT conversion - NO ENHANCEMENT.
        
        Args:
            spatial_tokens: [B, 256, D] compressed spatial tokens from TCME
        Returns:
            spatial_amplitude: [B, 16, 16, D] frequency amplitude
            spatial_phase: [B, 16, 16, D] frequency phase
        """
        B, N, D = spatial_tokens.shape
        # If tokens count matches the default grid (16x16), use 2D FFT. Otherwise, fall back to 1D FFT over sequence.
        if N == self.spatial_h * self.spatial_w:
            spatial_2d = spatial_tokens.view(B, self.spatial_h, self.spatial_w, D)
            freq_spatial = torch.fft.fft2(spatial_2d, dim=(1, 2))
            spatial_amplitude = freq_spatial.abs()
            spatial_phase = freq_spatial.angle()
            return spatial_amplitude, spatial_phase  # shapes [B, 16, 16, D]
        else:
            # 1D FFT along token axis when no square grid is available
            freq_seq = torch.fft.fft(spatial_tokens, dim=1)
            return freq_seq.abs(), freq_seq.angle()  # shapes [B, N, D]
    
    def spectral_to_frequency(self, spectral_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Direct spectral FFT conversion - NO ENHANCEMENT.
        Handles cuFFT power-of-two limitation for float16 by casting to float32 if needed.
        Args:
            spectral_tokens: [B, N, D] compressed spectral tokens from TCME
        Returns:
            spectral_amplitude: [B, N, D] spectral frequency amplitude
            spectral_phase: [B, N, D] spectral frequency phase
        """
        N = spectral_tokens.shape[1]
        # If float16 and not power of two, cast to float32 to avoid cuFFT error
        if spectral_tokens.dtype == torch.float16 and (N & (N - 1)) != 0:
            spectral_tokens = spectral_tokens.float()
        freq_spectral = torch.fft.fft(spectral_tokens, dim=1)
        return freq_spectral.abs(), freq_spectral.angle()
    
    def frequency_to_spatial_reconstruction(self, final_amplitude: torch.Tensor, 
                                          aligned_phase: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct spatial features using Inverse FFT.
        
        Args:
            final_amplitude: [B, N, D] final fused amplitude
            aligned_phase: [B, N, D] aligned phase
        Returns:
            reconstructed_features: [B, 256, D] reconstructed spatial features
        """
        # Create complex representation
        complex_freq = final_amplitude * torch.exp(1j * aligned_phase)
        B, N, D = final_amplitude.shape

        if N == 256:
            # Reshape and apply inverse FFT
            complex_2d = complex_freq.view(B, 16, 16, D)
            reconstructed = torch.fft.ifft2(complex_2d, dim=(1, 2))
            return reconstructed.real.view(B, 256, D)
        else:
            # Handle other dimensions
            real_part = complex_freq.real
            imag_part = complex_freq.imag
            
            target_size = 256
            if real_part.size(1) != target_size:
                real_part = F.interpolate(
                    real_part.transpose(1,2), size=target_size,
                    mode='linear', align_corners=False
                ).transpose(1,2)
            
            return real_part


# ================================================================================================
# CROSS-MODAL FREQUENCY FUSION CORE - 4 COMPONENT PREPARATION
# ================================================================================================
class CrossModalFrequencyFusionCore(nn.Module):
    """Cross-modal fusion that prepares 4 components for hierarchical processing."""
    
    def __init__(self, feature_dim: int = 128, num_heads: int = 8):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Learnable fusion weights for Component 1: Fused Amplitude
        self.spatial_weight = nn.Parameter(torch.ones(1))
        self.spectral_weight = nn.Parameter(torch.ones(1))
        
        # Component 2: Phase attention for alignment
        self.phase_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, 
            batch_first=True, dropout=0.0
        )
        self.phase_proj = nn.Linear(2 * feature_dim, feature_dim)  # For phase concatenation

        # Components 3 & 4: Low and high frequency filters
        self.low_freq_filter = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(feature_dim),
            nn.GELU()
        )
        self.high_freq_filter = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.GELU()
        )

        # Normalization layers
        self.amp_norm = nn.LayerNorm(feature_dim)
        self.phase_norm = nn.LayerNorm(feature_dim)
        
    def create_fused_amplitude(self, spatial_amplitude: torch.Tensor, 
                             spectral_amplitude: torch.Tensor) -> torch.Tensor:
        """Component 1: Create fused amplitude with learnable weights."""
        # Align dimensions to spectral format
        if spatial_amplitude.dim() == 4:
            B, H, W, D = spatial_amplitude.shape
            spatial_amplitude = spatial_amplitude.flatten(1, 2)
        
        target_seq_len = spectral_amplitude.size(1)
        if spatial_amplitude.size(1) != target_seq_len:
            spatial_amplitude = F.interpolate(
                spatial_amplitude.transpose(1, 2), size=target_seq_len,
                mode='linear', align_corners=False
            ).transpose(1, 2)
        
        # Weighted fusion
        fused_amplitude = self.spatial_weight * spatial_amplitude + self.spectral_weight * spectral_amplitude
        return self.amp_norm(fused_amplitude)
    
    def align_phase_information(self, spatial_phase: torch.Tensor, 
                              spectral_phase: torch.Tensor) -> torch.Tensor:
        """Component 2: Align phase using cross-modal attention."""
        # Align spatial phase dimensions
        if spatial_phase.dim() == 4:
            B, H, W, D = spatial_phase.shape
            spatial_phase = spatial_phase.flatten(1, 2)
        
        target_seq_len = spectral_phase.size(1)
        if spatial_phase.size(1) != target_seq_len:
            spatial_phase = F.interpolate(
                spatial_phase.transpose(1, 2), size=target_seq_len,
                mode='linear', align_corners=False
            ).transpose(1, 2)
        
        # Phase alignment via attention
        phase_concat = torch.cat([spatial_phase, spectral_phase], dim=-1)
        phase_proj = self.phase_proj(phase_concat)
        
        # --- Manual Multi-Head Attention for Stability ---
        # Forcing float32 and using the 'math' SDP kernel to prevent misaligned address errors.
        qkv = phase_proj.contiguous().float()
        B, N, D_total = qkv.shape
        num_heads = self.phase_attention.num_heads

        # Project and reshape for multi-head attention
        q, k, v = qkv.chunk(3, dim=-1)
        
        # head_dim is calculated from the chunked tensor's dimension
        head_dim = q.shape[-1] // num_heads

        q = q.reshape(B, N, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(B, N, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(B, N, num_heads, head_dim).transpose(1, 2)

        # Use safe math kernel for scaled_dot_product_attention
        try:
            from torch.backends.cuda import sdp_kernel
            ctx = sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False)
        except (ImportError, AttributeError):
            class _NoopCtx:
                def __enter__(self): pass
                def __exit__(self, *args): pass
            ctx = _NoopCtx()

        with ctx:
            attn_output = F.scaled_dot_product_attention(q, k, v)

        # Reshape and project back
        aligned_phase = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        aligned_phase = self.phase_attention.out_proj(aligned_phase)
        
        return self.phase_norm(aligned_phase)
    
    def separate_frequency_components(self, fused_amplitude: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Components 3 & 4: Separate into low and high frequency parts."""
        # Reshape for 1D convolution
        fused_conv = fused_amplitude.transpose(1, 2)  # [B, D, N]
        # Conv1d can be fragile under AMP on some CUDA backends. Force fp32 with autocast disabled.
        orig_dtype = fused_conv.dtype
        fused_conv_fp32 = fused_conv.float()
        try:
            with torch.cuda.amp.autocast(enabled=False):
                low_freq_conv = self.low_freq_filter(fused_conv_fp32)
                high_freq_conv = self.high_freq_filter(fused_conv_fp32)
        except RuntimeError as e:
            # As a last resort, retry without autocast context (already disabled) but re-cast inputs
            # This branch is unlikely to hit, but keeps training from crashing.
            low_freq_conv = self.low_freq_filter(fused_conv_fp32)
            high_freq_conv = self.high_freq_filter(fused_conv_fp32)
        # Cast outputs back to original dtype if needed (e.g., to bf16/half under AMP)
        if orig_dtype != torch.float32:
            low_freq_conv = low_freq_conv.to(orig_dtype)
            high_freq_conv = high_freq_conv.to(orig_dtype)
        
        # Back to token format
        low_freq = low_freq_conv.transpose(1, 2)   # [B, N, D]
        high_freq = high_freq_conv.transpose(1, 2) # [B, N, D]
        
        return low_freq, high_freq
    
    def forward(self, spatial_amplitude: torch.Tensor, spatial_phase: torch.Tensor,
                spectral_amplitude: torch.Tensor, spectral_phase: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create 4 components for hierarchical processing.
        
        Returns:
            fused_amplitude: Component 1 - Combined amplitude
            aligned_phase: Component 2 - Cross-modal aligned phase  
            low_freq: Component 3 - Low-frequency global structures
            high_freq: Component 4 - High-frequency fine details
        """
        # Component 1: Fused Amplitude
        fused_amplitude = self.create_fused_amplitude(spatial_amplitude, spectral_amplitude)
        
        # Component 2: Aligned Phase
        aligned_phase = self.align_phase_information(spatial_phase, spectral_phase)
        
        # Components 3 & 4: Frequency Separation
        low_freq, high_freq = self.separate_frequency_components(fused_amplitude)
        
        return fused_amplitude, aligned_phase, low_freq, high_freq


# ================================================================================================
# HIERARCHICAL MULTI-SCALE PROCESSOR - THE CORE FUSION
# ================================================================================================
class HierarchicalMultiScaleProcessor(nn.Module):
    """The CORE INNOVATION: Hierarchical Multi-Scale Processing with octave scales [1,2,4,8]."""
    
    def __init__(self, feature_dim: int = 128, scales: List[int] = [1, 2, 4, 8], verbose: bool = False):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.scales = scales
        self.verbose = verbose
        self._printed_once = False
        
        # Scale-specific processors
        self.scale_processors = nn.ModuleDict({
            str(scale): nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(feature_dim),
                nn.GELU(),
                nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
            ) for scale in scales
        })
        
        # Learnable scale attention weights for FUSED WEIGHTED SUM
        self.scale_attention_weights = nn.Parameter(torch.ones(len(scales)))
        
        # High-frequency attenuation factors for each scale
        self.high_freq_weights = {1: 1.0, 2: 0.5, 4: 0.25, 8: 0.0}
        
    def process_scale(self, scale: int, low_freq: torch.Tensor, high_freq: torch.Tensor) -> torch.Tensor:
        """Process individual scale with proper octave-scale logic."""
        if scale == 1:
            # Scale 1 (Full): Combine all frequencies
            return low_freq + high_freq
        elif scale == 2:
            # Scale 2 (Half): Reduce high freq by 0.5
            down_low = F.avg_pool1d(low_freq.transpose(1, 2), kernel_size=2, stride=1, padding=1).transpose(1, 2)
            down_high = F.avg_pool1d(high_freq.transpose(1, 2), kernel_size=2, stride=1, padding=1).transpose(1, 2)
            return down_low + self.high_freq_weights[2] * down_high
        elif scale == 4:
            # Scale 4 (Quarter): Reduce high freq by 0.25
            down_low = F.avg_pool1d(low_freq.transpose(1, 2), kernel_size=4, stride=1, padding=2).transpose(1, 2)
            down_high = F.avg_pool1d(high_freq.transpose(1, 2), kernel_size=4, stride=1, padding=2).transpose(1, 2)
            return down_low + self.high_freq_weights[4] * down_high
        else:  # scale == 8
            # Scale 8 (Eighth): Only global structure
            return F.avg_pool1d(low_freq.transpose(1, 2), kernel_size=8, stride=1, padding=4).transpose(1, 2)
    
    def fused_weighted_sum_amplitude_fusion(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """THE FUSED WEIGHTED SUM AMPLITUDE FUSION - The missing part!"""
        # Compute learnable scale weights
        scale_weights = F.softmax(self.scale_attention_weights, dim=0)
        
        # Stack features and apply weighted sum
        stacked_features = torch.stack(multi_scale_features, dim=0)  # [num_scales, B, N, D]
        final_fused_amplitude = torch.sum(
            scale_weights.view(-1, 1, 1, 1) * stacked_features, dim=0
        )
        
        return final_fused_amplitude
    
    def forward(self, fused_amplitude: torch.Tensor, aligned_phase: torch.Tensor,
                low_freq: torch.Tensor, high_freq: torch.Tensor) -> torch.Tensor:
        """
        THE ACTUAL FUSION: Hierarchical Multi-Scale Processing with Fused Weighted Sum.
        """
        B, N, D = fused_amplitude.shape
        multi_scale_features = []
        
        # Print header only once globally and as early as possible to avoid duplicates on re-entrant forwards
        if self.verbose and _should_log_tensor(fused_amplitude) and not _HCMFF_GLOBAL_PRINT['hms_header_printed']:
            print(f"🔥 HIERARCHICAL MULTI-SCALE PROCESSING")
            print(f"   Processing scales: {self.scales}")
            _HCMFF_GLOBAL_PRINT['hms_header_printed'] = True
        
        # Process each octave scale
        for i, scale in enumerate(self.scales):
            # Scale-specific component integration
            combined_amplitude = self.process_scale(scale, low_freq, high_freq)
            
            # Scale-specific learnable processing
            processed = self.scale_processors[str(scale)](combined_amplitude.transpose(1, 2)).transpose(1, 2)
            
            # Ensure correct output size
            if processed.size(1) != N:
                processed = F.interpolate(
                    processed.transpose(1, 2), size=N, mode='linear', align_corners=False
                ).transpose(1, 2)
            
            multi_scale_features.append(processed)
            if self.verbose and _should_log_tensor(processed) and not _HCMFF_GLOBAL_PRINT['hms_body_printed']:
                print(f"   ✓ Scale {scale}: {processed.shape}")
        
        # THE FUSED WEIGHTED SUM AMPLITUDE FUSION
        final_fused_amplitude = self.fused_weighted_sum_amplitude_fusion(multi_scale_features)
        
        scale_weights = F.softmax(self.scale_attention_weights, dim=0)
        if self.verbose and _should_log_tensor(final_fused_amplitude) and not _HCMFF_GLOBAL_PRINT['hms_body_printed']:
            print(f"   🎯 FUSED WEIGHTED SUM weights: {scale_weights.detach().cpu().numpy()}")
            print(f"   ✅ Final fused amplitude: {final_fused_amplitude.shape}")
            # Mark both the instance and global body as printed to prevent duplicates
            self._printed_once = True
            _HCMFF_GLOBAL_PRINT['hms_body_printed'] = True
        
        return final_fused_amplitude


# ================================================================================================
# MAIN HCMFF ARCHITECTURE
# ================================================================================================
class HierarchicalCrossModalityFrequencyFusion(nn.Module):
    """
    HCMFF: Hierarchical Cross-Modality Frequency Fusion - NO ENHANCEMENT VERSION
    
    Takes compressed tokens from TCME and performs direct frequency fusion.
    """
    
    def __init__(self, feature_dim: int = 128, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self._printed_once = False
        # Direct frequency processing - no enhancement
        self.freq_transforms = FrequencyDomainTransforms(feature_dim)
        self.fusion_core = CrossModalFrequencyFusionCore(feature_dim)
        self.hierarchical_processor = HierarchicalMultiScaleProcessor(feature_dim, verbose=verbose)
    
    def forward(self, spatial_tokens: torch.Tensor, spectral_tokens: torch.Tensor) -> torch.Tensor:
        """
        HCMFF Forward Pass - Direct from TCME outputs to frequency fusion.
        
        Args:
            spatial_tokens: [B, 256, D] compressed spatial tokens from TCME
            spectral_tokens: [B, N, D] compressed spectral tokens from TCME
        Returns:
            final_fused_features: [B, 256, D] fused features for downstream processing
        """
        # Print HCMFF header once globally (device-gated) to avoid repeats across DP replicas/retries
        if self.verbose and _should_log_tensor(spatial_tokens) and not _HCMFF_GLOBAL_PRINT['hcmff_header_printed']:
            print(f"\n🔥 HCMFF Forward Pass - Direct Frequency Fusion")
            print(f"   📥 TCME Outputs - Spatial: {spatial_tokens.shape}, Spectral: {spectral_tokens.shape}")
            _HCMFF_GLOBAL_PRINT['hcmff_header_printed'] = True
        
        # Stage 1: Direct Frequency Domain Transformation (NO ENHANCEMENT)
        if self.verbose and _should_log_tensor(spatial_tokens) and not self._printed_once:
            print(f"\n🌊 Direct FFT Conversion")
        spatial_amp, spatial_phase = self.freq_transforms.spatial_to_frequency(spatial_tokens)
        spectral_amp, spectral_phase = self.freq_transforms.spectral_to_frequency(spectral_tokens)
        if self.verbose and _should_log_tensor(spatial_tokens) and not self._printed_once:
            print(f"   ✓ Spatial FFT: amp={spatial_amp.shape}, phase={spatial_phase.shape}")
            print(f"   ✓ Spectral FFT: amp={spectral_amp.shape}, phase={spectral_phase.shape}")
        
        # Stage 2: Cross-Modal Frequency Fusion (4-component preparation)
        if self.verbose and _should_log_tensor(spatial_tokens) and not self._printed_once:
            print(f"\n🔄 Cross-Modal Frequency Fusion (4-Component Prep)")
        fused_amplitude, aligned_phase, low_freq, high_freq = self.fusion_core(
            spatial_amp, spatial_phase, spectral_amp, spectral_phase
        )
        if self.verbose and _should_log_tensor(spatial_tokens) and not self._printed_once:
            print(f"   ✓ Component 1 - Fused amplitude: {fused_amplitude.shape}")
            print(f"   ✓ Component 2 - Aligned phase: {aligned_phase.shape}")
            print(f"   ✓ Component 3 - Low frequency: {low_freq.shape}")
            print(f"   ✓ Component 4 - High frequency: {high_freq.shape}")
        
        # Stage 3: Hierarchical Multi-Scale Processing (THE ACTUAL FUSION)
        if self.verbose and _should_log_tensor(spatial_tokens) and not self._printed_once:
            print(f"\n⚡ Hierarchical Multi-Scale Processing (ACTUAL FUSION)")
        hierarchical_features = self.hierarchical_processor(
            fused_amplitude, aligned_phase, low_freq, high_freq
        )
        
        # Stage 4: Spatial Reconstruction
        if self.verbose and _should_log_tensor(spatial_tokens) and not self._printed_once:
            print(f"\n🔄 Spatial Domain Reconstruction")
        final_features = self.freq_transforms.frequency_to_spatial_reconstruction(
            hierarchical_features, aligned_phase
        )
        if self.verbose and _should_log_tensor(spatial_tokens) and not self._printed_once:
            print(f"   ✅ Final output: {final_features.shape}")
            print(f"\n🎉 HCMFF Complete!")
            self._printed_once = True
        return final_features


# ================================================================================================
# INDIVIDUAL FUNCTION ACCESS - FOR FULL_MODEL.PY USAGE
# ================================================================================================

# You can access individual functions like this in your full_model.py:

def get_spatial_to_frequency():
    """Returns the spatial_to_frequency function for individual use."""
    freq_transforms = FrequencyDomainTransforms()
    return freq_transforms.spatial_to_frequency

def get_spectral_to_frequency():
    """Returns the spectral_to_frequency function for individual use."""
    freq_transforms = FrequencyDomainTransforms()
    return freq_transforms.spectral_to_frequency

def get_cross_modal_fusion():
    """Returns the cross-modal fusion function for individual use."""
    fusion_core = CrossModalFrequencyFusionCore()
    return fusion_core.forward

def get_hierarchical_processor():
    """Returns the hierarchical processor function for individual use."""
    hierarchical_processor = HierarchicalMultiScaleProcessor()
    return hierarchical_processor.forward


# ================================================================================================
# TEST FUNCTION - COMMENT OUT FOR PRODUCTION
# ================================================================================================
def test_hcmff_no_enhancement():
    """Test HCMFF with simulated TCME outputs."""
    print("="*80)
    print("🧪 TESTING HCMFF (NO ENHANCEMENT) - Simulated TCME Outputs")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    try:
        # Simulate TCME outputs (compressed tokens)
        batch_size = 1
        spatial_tokens = torch.randn(batch_size, 256, 128, device=device)  # TCME spatial output
        spectral_tokens = torch.randn(batch_size, 224, 128, device=device) # TCME spectral output
        
        print(f"\n📥 Simulated TCME Outputs:")
        print(f"   Spatial tokens: {spatial_tokens.shape}")
        print(f"   Spectral tokens: {spectral_tokens.shape}")
        
        # Initialize HCMFF (no enhancement)
        model = HierarchicalCrossModalityFrequencyFusion(feature_dim=128).to(device)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(spatial_tokens, spectral_tokens)
        
        # Validate
        expected_shape = (batch_size, 256, 128)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
        assert torch.isfinite(output).all(), "Output contains NaN/Inf!"
        
        print(f"\n✅ TEST PASSED!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
        
        return model, output
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        raise e


# ================================================================================================
# MAIN EXECUTION
# ================================================================================================
if __name__ == "__main__":
    print("🏥 HCMFF: Hierarchical Cross-Modality Frequency Fusion - NO ENHANCEMENT")
    print("👨‍💻 Author: Abiram | KLH University & IIIT-H iHUB DATA")
    print("📅 September 2025")
    print()
    
    # COMMENT OUT THE LINE BELOW WHEN PUSHING TO TEAMMATES
    # model, output = test_hcmff_no_enhancement()
    
    print(f"\n🔥 HCMFF ready for integration!")
    print(f"📝 Usage in full_model.py:")
    print(f"   from models.hf_net import HierarchicalCrossModalityFrequencyFusion")
    print(f"   from models.hf_net import get_spatial_to_frequency, get_hierarchical_processor")
    print(f"   hcmff = HierarchicalCrossModalityFrequencyFusion(feature_dim=128)")
    print(f"   output = hcmff(tcme_spatial_outputs, tcme_spectral_outputs)")