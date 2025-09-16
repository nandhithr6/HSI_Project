"""
Multi-Scale Transformer Decoder (MSTD) + Hierarchical Segmentation Head (HSH)
Complete implementation in single file
Author: Abiram
Date: September 16, 2025
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

def get_device():
    """Get optimal device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def print_memory_usage(device):
    """Print memory usage"""
    if device.type == 'cuda':
        print(f"CUDA Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    elif device.type == 'mps':
        print(f"MPS Memory: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")
    else:
        print("CPU mode - no GPU memory tracking")

# =================== CORE COMPONENTS ===================

class TokenToFeatureConverter(nn.Module):
    """Stage 1: Token-to-Feature Conversion with Cross-Modal Integration"""
    
    def __init__(self, input_dim: int = 256, output_dim: int = 512, 
                 spatial_size: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spatial_size = spatial_size
        
        # Adaptive Token Embedding
        self.token_linear = nn.Linear(input_dim, output_dim)
        
        # 2D Positional Encoding
        self.register_buffer('pos_encoding', 
                           self._generate_2d_pos_encoding(spatial_size, output_dim))
        
        # Cross-Modal Feature Integration
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def _generate_2d_pos_encoding(self, size: int, dim: int) -> torch.Tensor:
        """Generate 2D sinusoidal positional encoding"""
        pos_enc = torch.zeros(size, size, dim)
        
        for i in range(size):
            for j in range(size):
                for k in range(0, dim//4):
                    div_term = 10000 ** (4*k / dim)
                    pos_enc[i, j, 4*k] = math.sin(i / div_term)
                    pos_enc[i, j, 4*k+1] = math.cos(i / div_term) 
                    pos_enc[i, j, 4*k+2] = math.sin(j / div_term)
                    pos_enc[i, j, 4*k+3] = math.cos(j / div_term)
                    
        return pos_enc.flatten(0, 1)  # [size*size, dim]
    
    def forward(self, fused_tokens: torch.Tensor, 
                freq_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            fused_tokens: [B, N, D] where N=2800, D=256
            freq_features: [B, 256, 128] from HF-Net (optional)
        Returns:
            features: [B, 32, 32, 512]
        """
        B, N, D = fused_tokens.shape
        
        # Adaptive Token Embedding
        F0 = self.token_linear(fused_tokens)  # [B, N, 512]
        
        # Take first 1024 tokens and reshape to 32x32 grid
        F0_reshaped = F0[:, :1024, :].contiguous()  # [B, 1024, 512]
        F0_spatial = F0_reshaped.view(B, 32, 32, -1)  # [B, 32, 32, 512]
        
        # Add positional encoding
        F_pos = F0_spatial + self.pos_encoding.view(32, 32, -1).unsqueeze(0)
        
        # Cross-modal integration if freq_features provided
        if freq_features is not None:
            # Reshape for attention
            F_flat = F_pos.flatten(1, 2)  # [B, 1024, 512]
            freq_proj = F.adaptive_avg_pool1d(
                freq_features.transpose(1, 2), 1024
            ).transpose(1, 2)  # [B, 1024, 128]
            
            # Pad freq features to match dimension
            freq_padded = F.pad(freq_proj, (0, 512-128))  # [B, 1024, 512]
            
            # Cross attention
            F_integrated, _ = self.cross_attention(F_flat, freq_padded, freq_padded)
            F_integrated = self.layer_norm(F_integrated + F_flat)
            
            # Reshape back
            F_pos = F_integrated.view(B, 32, 32, -1)
        
        return F_pos.contiguous()

class TransformerBlock(nn.Module):
    """Transformer block with spatial awareness"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, C] or [B, H*W, C]
        """
        if x.dim() == 4:
            B, H, W, C = x.shape
            x_flat = x.flatten(1, 2)  # [B, H*W, C]
        else:
            x_flat = x
            
        # Self-attention
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + attn_out)
        
        # Feed forward
        ff_out = self.feed_forward(x_flat)
        x_flat = self.norm2(x_flat + ff_out)
        
        if x.dim() == 4:
            return x_flat.view(B, H, W, C)
        return x_flat

class CrossScaleAttention(nn.Module):
    """Cross-Scale Attention Mechanism"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, current_scale: torch.Tensor, 
                prev_scale: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current_scale: [B, H, W, C]
            prev_scale: [B, H_prev, W_prev, C]
        """
        B, H, W, C = current_scale.shape
        B_prev, H_prev, W_prev, C_prev = prev_scale.shape
        
        # Flatten current scale
        current_flat = current_scale.flatten(1, 2)  # [B, H*W, C]
        
        # Upsample previous scale to match current
        prev_upsampled = F.interpolate(
            prev_scale.permute(0, 3, 1, 2), 
            size=(H, W), mode='bilinear', align_corners=False
        ).permute(0, 2, 3, 1)  # [B, H, W, C]
        
        prev_flat = prev_upsampled.flatten(1, 2)  # [B, H*W, C]
        
        # Cross-scale attention
        attn_out, _ = self.attention(current_flat, prev_flat, prev_flat)
        output = self.norm(current_flat + attn_out)
        
        return output.view(B, H, W, C)

class MultiScaleFusion(nn.Module):
    """Multi-Scale Fusion Strategy with Adaptive Weights"""
    
    def __init__(self, num_scales: int = 3, feature_dim: int = 64):
        super().__init__()
        self.num_scales = num_scales
        
        # Adaptive fusion weight generation
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_scales),
            nn.Softmax(dim=1)
        )
        
    def forward(self, scale_features: List[torch.Tensor], 
                target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            scale_features: List of [B, H_i, W_i, C] tensors
            target_size: (H_target, W_target)
        """
        B = scale_features[0].shape[0]
        target_h, target_w = target_size
        
        # Resize all features to target size
        resized_features = []
        for feat in scale_features:
            if feat.shape[1:3] != (target_h, target_w):
                feat_resized = F.interpolate(
                    feat.permute(0, 3, 1, 2),
                    size=(target_h, target_w), 
                    mode='bilinear', align_corners=False
                ).permute(0, 2, 3, 1)
            else:
                feat_resized = feat
            resized_features.append(feat_resized)
        
        # Generate adaptive weights using the last feature
        weights = self.weight_generator(scale_features[-1].permute(0, 3, 1, 2))
        weights = weights.view(B, len(scale_features), 1, 1, 1)
        
        # Weighted fusion
        stacked_features = torch.stack(resized_features, dim=1)  # [B, num_scales, H, W, C]
        fused = torch.sum(weights * stacked_features, dim=1)
        
        return fused

class HierarchicalSegmentationHead(nn.Module):
    """Hierarchical Segmentation Head with Multi-Scale Prediction"""
    
    def __init__(self, feature_dim: int = 32, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        
        # Primary prediction head (512x512)
        self.main_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, num_classes, 1)
        )
        
        # Auxiliary prediction heads
        self.aux_head_256 = nn.Conv2d(64, num_classes, 1)   # 256x256
        self.aux_head_128 = nn.Conv2d(128, num_classes, 1)  # 128x128
        
        # Boundary enhancement
        self.boundary_detector = nn.Sequential(
            nn.Conv2d(feature_dim, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Class-specific refinement heads
        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim//2, 3, padding=1),
                nn.BatchNorm2d(feature_dim//2),
                nn.GELU(),
                nn.Conv2d(feature_dim//2, 1, 1)
            ) for _ in range(num_classes)
        ])
        
    def forward(self, main_features: torch.Tensor,
                aux_features_256: torch.Tensor,
                aux_features_128: torch.Tensor) -> dict:
        """
        Args:
            main_features: [B, 32, 512, 512]
            aux_features_256: [B, 64, 256, 256] 
            aux_features_128: [B, 128, 128, 128]
        Returns:
            dict with logits and predictions
        """
        # Main prediction
        main_logits = self.main_head(main_features)  # [B, num_classes, 512, 512]
        main_probs = F.softmax(main_logits, dim=1)
        
        # Auxiliary predictions
        aux_logits_256 = self.aux_head_256(aux_features_256)  # [B, num_classes, 256, 256]
        aux_logits_128 = self.aux_head_128(aux_features_128)  # [B, num_classes, 128, 128]
        
        # Boundary enhancement
        boundary_mask = self.boundary_detector(main_features)  # [B, 1, 512, 512]
        enhanced_logits = main_logits * (1 + 0.1 * boundary_mask)
        
        # Class-specific refinement
        class_outputs = []
        for class_head in self.class_heads:
            class_out = class_head(main_features)  # [B, 1, 512, 512]
            class_outputs.append(class_out)
        
        class_logits = torch.cat(class_outputs, dim=1)  # [B, num_classes, 512, 512]
        
        # Ensemble final prediction
        final_logits = (enhanced_logits + class_logits) / 2
        final_probs = F.softmax(final_logits, dim=1)
        
        return {
            'main_logits': main_logits,
            'main_probs': main_probs,
            'aux_logits_256': aux_logits_256,
            'aux_logits_128': aux_logits_128,
            'boundary_mask': boundary_mask,
            'final_logits': final_logits,
            'final_probs': final_probs
        }

# =================== MAIN DECODER CLASS ===================

class MSTDHSHDecoder(nn.Module):
    """
    Multi-Scale Transformer Decoder + Hierarchical Segmentation Head
    
    Input: Fused tokens from HCMFF [B, 2800, 256] + freq features [B, 256, 128]
    Output: Segmentation predictions at multiple scales
    """
    
    def __init__(self, 
                 input_token_dim: int = 256,
                 num_classes: int = 5,
                 scales: list = [64, 128, 256, 512]):
        super().__init__()
        
        self.scales = scales
        self.num_classes = num_classes
        
        print("🚀 Initializing MSTD+HSH Decoder...")
        
        # Stage 1: Token-to-Feature Conversion
        self.token_converter = TokenToFeatureConverter(
            input_dim=input_token_dim, output_dim=512, spatial_size=32
        )
        print("   ✓ Token-to-Feature Converter")
        
        # Stage 2: Hierarchical Upsampling with Cross-Scale Attention
        # Scale 1: 32x32 -> 64x64 (512 -> 256 channels)
        self.upsample1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.transformer1 = TransformerBlock(256, num_heads=8)
        self.norm1 = nn.LayerNorm(256)
        
        # Scale 2: 64x64 -> 128x128 (256 -> 128 channels)  
        self.upsample2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.cross_attn2 = CrossScaleAttention(128, num_heads=8)
        self.transformer2 = TransformerBlock(128, num_heads=8)
        
        # Scale 3: 128x128 -> 256x256 (128 -> 64 channels)
        self.upsample3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.multi_fusion3 = MultiScaleFusion(num_scales=3, feature_dim=64)
        self.transformer3 = TransformerBlock(64, num_heads=8)
        
        # Scale 4: 256x256 -> 512x512 (64 -> 32 channels)
        self.upsample4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.final_refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 1)
        )
        
        # Hierarchical Segmentation Head
        self.seg_head = HierarchicalSegmentationHead(
            feature_dim=32, num_classes=num_classes
        )
        print("   ✓ Hierarchical Segmentation Head")
        
        print("🎯 MSTD+HSH Decoder initialization complete!")
    
    def forward(self, fused_tokens: torch.Tensor, 
                freq_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MSTD+HSH decoder
        
        Args:
            fused_tokens: [B, 2800, 256] from HCMFF
            freq_features: [B, 256, 128] from HF-Net (optional)
        
        Returns:
            Dictionary with predictions at multiple scales
        """
        # Use mixed precision for efficiency
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            
            # Stage 1: Token-to-Feature Conversion
            F0 = self.token_converter(fused_tokens, freq_features)  # [B, 32, 32, 512]
            F0 = F0.permute(0, 3, 1, 2).contiguous()  # [B, 512, 32, 32]
            
            # Scale 1: 32x32 -> 64x64
            F1 = self.upsample1(F0)  # [B, 256, 64, 64]
            F1_refined = self.transformer1(F1.permute(0, 2, 3, 1))  # [B, 64, 64, 256]
            F1_final = self.norm1(F1_refined + F1.permute(0, 2, 3, 1))
            F1_final = F1_final.permute(0, 3, 1, 2).contiguous()  # [B, 256, 64, 64]
            
            # Scale 2: 64x64 -> 128x128  
            F2 = self.upsample2(F1_final)  # [B, 128, 128, 128]
            F2_cross = self.cross_attn2(
                F2.permute(0, 2, 3, 1), 
                F1_final.permute(0, 2, 3, 1)
            )  # [B, 128, 128, 128]
            F2_final = self.transformer2(F2_cross)
            F2_final = F2_final.permute(0, 3, 1, 2).contiguous()  # [B, 128, 128, 128]
            
            # Scale 3: 128x128 -> 256x256
            F3 = self.upsample3(F2_final)  # [B, 64, 256, 256]
            # Multi-scale fusion
            scale_features = [
                F1_final.permute(0, 2, 3, 1),  # [B, 64, 64, 256]
                F2_final.permute(0, 2, 3, 1),  # [B, 128, 128, 128] 
                F3.permute(0, 2, 3, 1)         # [B, 256, 256, 64]
            ]
            F3_fused = self.multi_fusion3(scale_features, target_size=(256, 256))
            F3_final = self.transformer3(F3_fused)
            F3_final = F3_final.permute(0, 3, 1, 2).contiguous()  # [B, 64, 256, 256]
            
            # Scale 4: 256x256 -> 512x512
            F4 = self.upsample4(F3_final)  # [B, 32, 512, 512]
            F_output = self.final_refine(F4)  # [B, 32, 512, 512]
            
            # Hierarchical Segmentation Head
            seg_outputs = self.seg_head(
                main_features=F_output,
                aux_features_256=F3_final,
                aux_features_128=F2_final
            )
            
        return seg_outputs

    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical loss function
        
        Args:
            outputs: Dictionary from forward pass
            targets: [B, H, W] ground truth labels
        
        Returns:
            Dictionary with loss components
        """
        # Main segmentation loss
        main_loss = F.cross_entropy(outputs['main_logits'], targets)
        
        # Auxiliary losses
        targets_256 = F.interpolate(
            targets.unsqueeze(1).float(), size=(256, 256), mode='nearest'
        ).squeeze(1).long()
        targets_128 = F.interpolate(
            targets.unsqueeze(1).float(), size=(128, 128), mode='nearest'
        ).squeeze(1).long()
        
        aux_loss_256 = F.cross_entropy(outputs['aux_logits_256'], targets_256)
        aux_loss_128 = F.cross_entropy(outputs['aux_logits_128'], targets_128)
        
        # Total hierarchical loss
        seg_loss = main_loss + 0.5 * aux_loss_256 + 0.25 * aux_loss_128
        final_loss = F.cross_entropy(outputs['final_logits'], targets)
        
        total_loss = seg_loss + final_loss
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'aux_loss_256': aux_loss_256,
            'aux_loss_128': aux_loss_128,
            'seg_loss': seg_loss,
            'final_loss': final_loss
        }

# =================== TEST FUNCTION ===================

def test_mstd_hsh_decoder():
    """Test MSTD+HSH decoder with simulated inputs"""
    print("="*80)
    print("🧪 TESTING MSTD+HSH DECODER")
    print("="*80)
    
    device = get_device()
    print(f"🔧 Device: {device}")
    
    # Initialize model
    model = MSTDHSHDecoder(
        input_token_dim=256,
        num_classes=5,
        scales=[64, 128, 256, 512]
    ).to(device)
    
    print_memory_usage(device)
    
    # Simulate inputs from HCMFF
    batch_size = 2
    fused_tokens = torch.randn(batch_size, 2800, 256, device=device)  # From HCMFF
    freq_features = torch.randn(batch_size, 256, 128, device=device)  # From HF-Net
    targets = torch.randint(0, 5, (batch_size, 512, 512), device=device)  # Ground truth
    
    print(f"\n📥 Input Shapes:")
    print(f"   Fused tokens: {fused_tokens.shape}")
    print(f"   Freq features: {freq_features.shape}")
    print(f"   Targets: {targets.shape}")
    
    # Test forward pass
    model.eval()
    print(f"\n🔥 Forward Pass...")
    t0 = time.time()
    
    with torch.no_grad():
        outputs = model(fused_tokens, freq_features)
    
    t1 = time.time()
    
    # Print outputs
    print(f"\n📤 Output Shapes:")
    for key, value in outputs.items():
        print(f"   {key}: {value.shape}")
    
    # Test loss computation
    print(f"\n💥 Loss Computation...")
    model.train()
    outputs_train = model(fused_tokens, freq_features)
    losses = model.compute_loss(outputs_train, targets)
    
    print(f"\n📊 Loss Values:")
    for key, value in losses.items():
        print(f"   {key}: {value.item():.4f}")
    
    print_memory_usage(device)
    print(f"\n⏱️  Inference Time: {t1-t0:.3f}s")
    print(f"🎯 Final logits stats: mean={outputs['final_logits'].mean().item():.4f}, "
          f"std={outputs['final_logits'].std().item():.4f}")
    
    print(f"\n✅ MSTD+HSH DECODER TEST PASSED!")
    
    return model, outputs, losses

# =================== MAIN EXECUTION ===================

if __name__ == "__main__":
    print("🏥 MSTD+HSH: Multi-Scale Transformer Decoder + Hierarchical Segmentation Head")
    print("👨‍💻 Author: Abiram | KLH University & IIIT-H iHUB DATA")
    print("📅 September 16, 2025")
    print()
    
    # COMMENT OUT THE LINE BELOW FOR PRODUCTION
    # model, outputs, losses = test_mstd_hsh_decoder()
    
    print(f"\n🔥 MSTD+HSH Decoder ready for integration!")
    print(f"📝 Usage:")
    print(f"   from main import MSTDHSHDecoder")
    print(f"   decoder = MSTDHSHDecoder(input_token_dim=256, num_classes=5)")
    print(f"   outputs = decoder(fused_tokens, freq_features)")
