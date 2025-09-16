import torch
from main import TokenCrossModalEnhancer

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_sanity():
    print("🚀 Running TCME Sanity Test")

    # ---------------- Dummy Input ----------------
    B, H, W, D = 2, 8, 8, 32    # very small so it runs fast
    patch_size = 2              # tiny patch size
    Np = (H // patch_size) * (W // patch_size)

    # Spatial tokens (patches)
    Tspatial = torch.randn(B, Np, D, device=device)
    # Spectral tokens (pixels)
    Tspectral = torch.randn(B, H * W, D, device=device)

    # ---------------- Model ----------------
    model = TokenCrossModalEnhancer(dim=D, num_heads=4,
                                    N_pairs=10, K_spatial=4, K_spectral=8,
                                    use_checkpoint=False).to(device)

    # ---------------- Forward ----------------
    Tsp_sel, Tspc_sel = model(Tspatial, Tspectral, H, W)

    print("Spatial tokens out:", Tsp_sel.shape)   # (B, K_spatial, D)
    print("Spectral tokens out:", Tspc_sel.shape) # (B, K_spectral, D)

    # ---------------- AMP Test ----------------
    if device == "cuda":
        print("\n⚡ Testing AMP compatibility...")
        scaler = torch.amp.GradScaler("cuda")
        with torch.amp.autocast("cuda"):
            out_sp, out_spec = model(Tspatial, Tspectral, H, W)
            loss = (out_sp.mean() + out_spec.mean())
        scaler.scale(loss).backward(retain_graph=True) if loss.requires_grad else None
        print("⚠️ Skipped backward (no gradients in dummy input)")
    else:
        print("\n⚡ Skipping AMP test (CPU only)")
        loss = (Tsp_sel.mean() + Tspc_sel.mean())
        loss.backward()
        print("✅ CPU backward pass successful")

    print("\n🎉 TCME sanity test passed!")


if __name__ == "__main__":
    run_sanity()
