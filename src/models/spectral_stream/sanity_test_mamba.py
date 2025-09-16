"""
sanity_test_mamba.py

Quick sanity test for SpectralStreamMamba using a dummy HSI cube.
"""

import torch
import numpy as np
from main import SpectralStreamExact as SpectralStreamMamba, process_tiles  # import from your main.py
#
def main():
    # ----------------------
    # 1. Create dummy HSI cube
    # ----------------------
    H, W, B = 256, 256, 38
    cube = np.random.rand(H, W, B).astype(np.float32)  # random spectral cube

    # ----------------------
    # 2. Initialize model
    # ----------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralStreamMamba(
        band_count=B,
        window_sizes=[8, 16, 32],  # valid since B >= 32
        model_dim=32,              # keep small for sanity test
        token_dim=64,
        out_dim=64
    ).to(device)

    # ----------------------
    # 3. Process tiles (simulate full pipeline)
    # ----------------------
    F_spectral, T_tokens = process_tiles(
        cube, model, device,
        tile=32,     # small tile size for quick check
        overlap=8
    )

    # ----------------------
    # 4. Print sanity check outputs
    # ----------------------
    print("Input cube shape:", cube.shape)
    print("F_spectral shape:", F_spectral.shape)
    print("T_tokens shape:", T_tokens.shape)

    # Expected:
    # Input cube shape: (64, 64, 40)
    # F_spectral shape: (64, 64, out_dim)
    # T_tokens shape:   (64, 64, token_dim)

if __name__ == "__main__":
    main()
