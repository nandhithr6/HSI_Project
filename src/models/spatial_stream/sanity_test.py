"""
Sanity test for the Spatial Stream components: LocalStream, GlobalStream2DVMamba, SpatialFusion, and SpatialTokenizer.
This script verifies the forward and backward passes, AMP compatibility, and memory format handling.    
Author: Nandhitha
Date: September 16, 2025
"""

import torch
from local_main import LocalStream
from global_main import GlobalStream2DVMamba
from fusion_main import SpatialFusion
from spatial_tokenizer import SpatialTokenizer

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_sanity():
    print("🚀 Running Spatial Stream Sanity Test")

    # Dummy input (like Placenta dataset: 38 bands)
    x = torch.randn(2, 1, 64, 64, 38, device=device)

    # ---------------- Local ----------------
    local = LocalStream(in_channels=1, base_channels=128).to(device)
    Floc = local(x)
    print("LocalStream out:", Floc.shape)  # (B, C, H, W)

    # ---------------- Global ----------------
    glob = GlobalStream2DVMamba(in_channels=Floc.size(1), out_channels=128).to(device)
    Fglob = glob(Floc)
    print("GlobalStream out:", Fglob.shape)

    # ---------------- Fusion ----------------
    fusion = SpatialFusion(channels=128).to(device)
    Ffused = fusion(Floc, Fglob)
    print("Fusion out:", Ffused.shape)

    # ---------------- Tokenizer ----------------
    tokenizer = SpatialTokenizer(in_channels=128, embed_dim=64, patch_size=16).to(device)
    tokens = tokenizer(Ffused)
    print("Tokenizer out:", tokens.shape)  # (B, N_patches, D)

    # ---------------- AMP Test ----------------
    print("\n⚡ Testing AMP compatibility...")
    scaler = torch.amp.GradScaler('cuda')
    with torch.amp.autocast('cuda'):
        out = tokenizer(fusion(Floc, Fglob))
        loss = out.mean()
    scaler.scale(loss).backward()
    print("✅ AMP backward pass successful")

    # ---------------- Channels-last Test ----------------
    print("\n⚡ Checking memory format...")
    print("Floc channels_last:", Floc.is_contiguous(memory_format=torch.channels_last))
    print("Fglob channels_last:", Fglob.is_contiguous(memory_format=torch.channels_last))
    print("Ffused channels_last:", Ffused.is_contiguous(memory_format=torch.channels_last))

    print("\n🎉 Spatial Stream sanity test passed!")


if __name__ == "__main__":
    run_sanity()
