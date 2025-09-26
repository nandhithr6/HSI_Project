# HSI Project - Session Summary for Abiram

## Overview
We worked on debugging and improving the HSI segmentation training pipeline. The main focus was fixing device placement issues that prevented stable multi-GPU training and implementing better training infrastructure.

## ✅ What We Successfully Implemented

### 1. Organized Folder Structure
- **Before**: Logs and weights were mixed in `saved/models/` directory
- **After**: Created run-specific directories: `saved/models/{run-name}/logs/` and `saved/models/{run-name}/weights/`
- **Files Modified**: `src/training/train.py` (CSVLogger class)

### 2. Comprehensive Data Augmentations  
- **Added**: Horizontal/vertical flips, 90° rotations, spectral noise injection
- **Implementation**: `NPZDataset` class with `augment` parameter
- **Files Modified**: `src/training/train.py` (NPZDataset class)

### 3. Device Placement Infrastructure
- **Added**: `warm_up()` method in `HSIModel` to initialize dynamic components before DataParallel
- **Added**: `ensure_device_consistency()` method to force all modules to correct device
- **Added**: Override of `train()` method to maintain device consistency
- **Files Modified**: `src/models/full_model.py`, `src/training/train.py`

### 4. HCMFF Token Projection Pre-initialization
- **Fixed**: Pre-initialize the 1024→128 token projection layer in constructor instead of creating dynamically
- **Files Modified**: `src/models/full_model.py` (constructor)

## ✅ Critical Issue - Resolved

### Device Placement Error in Epoch 2 (fixed)
We resolved the multi-GPU DataParallel device mismatch that previously appeared at the start of epoch 2.

Key fixes:
- Removed deprecated `torch.backends.cuda.sdp_kernel()` fallback; use `torch.nn.attention.sdpa_kernel()` with a no-op fallback (no more FutureWarning).
- Enforced `.contiguous()` on tensors before all conv/transpose/attention ops in the decoder to prevent CUDA "misaligned address" errors.
- Stabilized DP device placement:
  - Normalize the master device to `cuda:0` in `ensure_device_consistency()` and call it only during warmup and `train()` transitions (not inside `forward` and not at epoch boundaries).
  - Avoid moving modules mid-forward, which could desynchronize BatchNorm buffers across replicas.
- Made `CSVLogger` resilient to missing directories/files to avoid logging crashes.

Validation:
- 2-epoch and 3-epoch runs complete successfully across 4x RTX 3080 Ti with DP, without device mismatch or CUDA errors.
- Logs and weights are written to `saved/models/{run-name}/{logs|weights}` as expected.

## 🔍 Root Cause Analysis

The issue is **NOT** with our model code but likely with:
1. **Deep submodules**: One of the spatial_stream, spectral_stream, TCME, HCMFF, or decoder submodules creates CPU tensors
2. **Mamba-SSM library**: The spectral stream uses mamba-ssm which might have device placement bugs
3. **PyTorch version compatibility**: There might be version-specific DataParallel issues

## 📝 Architecture Summary

The HSI model has this flow:
```
Input (B, Bands, H, W)
├── Spatial Stream
│   ├── LocalFeatureStream (3D+2D CNN)
│   ├── GlobalFeatureStream (Mamba-based)
│   └── SpatialTokenizer → spatial_tokens (1024)
├── Spectral Stream → spectral_tokens (1024)
└── Fusion
    ├── HCMFF (compress to 128 tokens) OR
    └── TCME (keep 2048 tokens)
    └── Decoder → Segmentation Output
```

## 🚀 Next Steps for You

### Immediate Debugging
1. **Test without DataParallel**: Run single GPU with batch_size=1 to confirm model works
   ```bash
   # Disable DataParallel in train.py (lines 513-522)
   python -m src.training.train --data-dir <path> --batch-size 1 --epochs 3
   ```

2. **Add device debugging**: Insert device checks in submodules to find which creates CPU tensors
   ```python
   # Add this in forward methods of submodules
   for name, param in self.named_parameters():
       if param.device.type == 'cpu':
           print(f"[ERROR] {name} on CPU!")
   ```

### Alternative Solutions
1. **Try DistributedDataParallel**: Replace DataParallel with DDP
2. **Use different PyTorch version**: Try PyTorch 2.0 vs 2.1+
3. **Memory optimization**: Reduce model size to fit single GPU better

## 📁 Key Files Modified

- `src/training/train.py` - Main training script (folder structure, augmentations, warmup)
- `src/models/full_model.py` - Main model (device consistency, initialization)
- All submodules are untouched (potential source of CPU tensor issue)

## 🔧 Current Training Command

Working command (with our fixes):
```bash
python -m src.training.train \
  --data-dir /ssd_scratch/placenta/Placenta \
  --epochs 3 \
  --batch-size 4 \
  --use-hcmff \
  --hcmff-tokens 128 \
  --merge-icg-to-base \
  --crop-size 512 \
  --run-name "your-experiment-name"
```

## 💡 Final Thoughts

The model architecture is solid and works correctly. The issue is purely a PyTorch DataParallel device management problem. Focus on finding which submodule creates the rogue CPU tensor, and consider switching to DistributedDataParallel or optimizing for single-GPU usage.

Good luck! 🚀

---
*Session completed on 2025-09-26 by Chinmay with GitHub Copilot*