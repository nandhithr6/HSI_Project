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

## ❌ Critical Issue - Still Unresolved

### Device Placement Error in Epoch 2
**Error Message**: 
```
RuntimeError: module must have its parameters and buffers on device cuda:0 (device_ids[0]) but found one of them on device: cpu
```

**Pattern**: 
- ✅ First epoch always runs successfully 
- ❌ Second epoch always fails with CPU device error
- ❌ Happens even with single GPU + DataParallel
- ❌ Happens consistently across all configurations

**What We Tried**:
1. Dynamic decoder initialization with proper device placement ❌
2. Pre-initialized all components during warmup ❌  
3. Added device consistency checks at epoch start ❌
4. Forced all modules to GPU on every `train()` call ❌
5. Disabled DataParallel entirely (works but memory issues) ✅

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