"""
Production-ready HSI training script for ADA servers - V2.2

Features:
- NPZ dataset loading from a directory (one .npz per sample with image+mask)
- 5-fold cross-validation or train/val/test split discovery
- Joint spatial+spectral augmentations
- Unified Focal Loss (lambda-weighted focal CE + focal Dice)
- AdamW, gradient clipping, cosine warm restarts + warmup
- Mixed precision (AMP), early stopping, checkpointing
- Corrected modern torch.amp syntax to fix TypeError.

Notes:
- On ADA, datasets live under /ssd_scratch; pass --data-dir "/ssd_scratch/<user>/<dataset>"
- Checkpoints/weights saved to saved/models by default
"""

import argparse
import csv
import glob
import json
import os
import random

# Ensure CuBLAS deterministic workspace is set before importing torch (prevents Mamba determinism crash)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
try:
    from thop import profile as thop_profile
except Exception:
    thop_profile = None
try:
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import binary_erosion
except Exception:
    distance_transform_edt = None
    binary_erosion = None

# Local imports
from src.models.full_model import HSIModel

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

    # Preferred SDPA kernel: use torch.nn.attention.sdpa_kernel when available; otherwise no-op
class _NoopCtx:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False

def _sdpa_mem_eff_ctx():
    """Prefer modern SDPA kernel; if unavailable, use a no-op context to avoid deprecation warnings."""
    try:
        from torch.nn.attention import sdpa_kernel as _sdpa
        return _sdpa(enable_math=False, enable_flash=False, enable_mem_efficient=True)
    except Exception:
        return _NoopCtx()

###############################################
# Utilities and helpers
###############################################

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Avoid forcing full determinism because Mamba/cuBLAS may require special settings; rely on seeds + cudnn.determinism


def print_gpu_summary():
    if not torch.cuda.is_available():
        print("CUDA not available; running on CPU")
        return
    n = torch.cuda.device_count()
    print(f"CUDA available: {n} device(s)")
    for i in range(n):
        print(f"  - [{i}] {torch.cuda.get_device_name(i)}")


def _confusion_matrix_torch(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Fast confusion matrix for per-pixel predictions."""
    k = (target >= 0) & (target < num_classes)
    inds = num_classes * target[k].to(torch.int64) + pred[k].to(torch.int64)
    cm = torch.bincount(inds, minlength=num_classes**2).reshape(num_classes, num_classes).to(torch.int64)
    return cm


def _per_class_stats_from_cm(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        specificity = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
        dice = np.where(2 * tp + fp + fn > 0, (2 * tp) / (2 * tp + fp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'iou': iou,
        'dice': dice,
        'f1': f1,
    }


def _mcc_from_cm(cm: np.ndarray) -> float:
    """Multiclass MCC computed from confusion matrix (no need to flatten arrays)."""
    cm = cm.astype(np.float64)
    t_k = cm.sum(axis=1)
    p_k = cm.sum(axis=0)
    c = np.trace(cm)
    s = cm.sum()
    sum_pk_tk = (p_k * t_k).sum()
    sum_pk_sq = (p_k ** 2).sum()
    sum_tk_sq = (t_k ** 2).sum()
    denom = np.sqrt((s**2 - sum_pk_sq) * (s**2 - sum_tk_sq))
    if denom == 0:
        return float('nan')
    return float((c * s - sum_pk_tk) / denom)


def _hd95_per_image(gt: np.ndarray, pr: np.ndarray) -> Tuple[float, float]:
    """Compute Hausdorff distance (max) and HD95 between two binary masks.
    Returns (HD, HD95). If unavailable or empty masks, returns (nan, nan).
    """
    if distance_transform_edt is None or binary_erosion is None:
        return float('nan'), float('nan')
    gt_b = gt.astype(bool)
    pr_b = pr.astype(bool)
    if gt_b.sum() == 0 and pr_b.sum() == 0:
        return float('nan'), float('nan')
    if gt_b.sum() == 0 or pr_b.sum() == 0:
        return float('inf'), float('inf')
    gt_edge = gt_b ^ binary_erosion(gt_b)
    pr_edge = pr_b ^ binary_erosion(pr_b)
    dt_gt = distance_transform_edt(~gt_edge)
    dt_pr = distance_transform_edt(~pr_edge)
    d_gt_pr = dt_pr[gt_edge]
    d_pr_gt = dt_gt[pr_edge]
    if d_gt_pr.size == 0 or d_pr_gt.size == 0:
        return float('nan'), float('nan')
    hd = float(max(d_gt_pr.max(), d_pr_gt.max()))
    hd95 = float(max(np.percentile(d_gt_pr, 95), np.percentile(d_pr_gt, 95)))
    return hd, hd95


class CSVLogger:
    def __init__(self, log_dir: str, run_name: Optional[str], config: Dict):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = run_name or f"run-{ts}"
        self.run_dir = os.path.join(log_dir, name)
        ensure_dir(self.run_dir)
        # write config
        with open(os.path.join(self.run_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        # csv header
        self.csv_path = os.path.join(self.run_dir, 'epochs.csv')
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss", "lr"])

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        # Ensure directory and file exist; recreate header if missing
        run_dir = os.path.dirname(self.csv_path)
        ensure_dir(run_dir)
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["epoch", "train_loss", "val_loss", "lr"])
            w.writerow([epoch, train_loss, val_loss, lr])

    def close(self):
        pass


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.bad_count = 0

    def step(self, value: float) -> bool:
        if value + self.min_delta < self.best:
            self.best = value
            self.bad_count = 0
            return False
        self.bad_count += 1
        return self.bad_count >= self.patience


class UnifiedFocalLoss(nn.Module):
    """Lambda-weighted Focal CrossEntropy + Focal Dice.

    Args:
        num_classes: number of classes including background=0
        lambda_: weight between CE and Dice (0..1) for: loss = lambda*CE + (1-lambda)*Dice
        alpha, gamma: focal CE params
        delta: focal dice focusing parameter
        smooth: dice smoothing
    """
    def __init__(self, num_classes: int, lambda_: float = 0.5, alpha: float = 0.25, gamma: float = 2.0, delta: float = 0.5, smooth: float = 1e-8, dice_exclude_bg: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_ = lambda_
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.smooth = smooth
        self.dice_exclude_bg = dice_exclude_bg

    def forward(self, logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
        # logits: [B, C, H, W]; target: [B, H, W] (int)
        B, C, H, W = logits.shape
        # Focal CE
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        # Gather log-prob of the true class
        ce = F.nll_loss(logp, target.long(), reduction='none')  # [B,H,W]
        pt = torch.gather(p, dim=1, index=target.long().unsqueeze(1)).squeeze(1)  # [B,H,W]
        alpha_t = self.alpha
        if class_weights is not None:
            # Map per-class weights to each pixel via target
            wmap = class_weights.to(logits.device)[target.long()]
        else:
            wmap = 1.0
        focal_ce = (wmap * alpha_t * (1 - pt) ** self.gamma * ce).mean()

        # Focal Dice (soft dice over one-hot)
        target_1h = F.one_hot(target.long(), num_classes=C).permute(0, 3, 1, 2).float()  # [B,C,H,W]
        probs = F.softmax(logits, dim=1)
        # presence mask per sample per class (True if class has any pixel in target)
        present = (target_1h.sum(dim=(2, 3)) > 0).float()  # [B,C]
        if self.dice_exclude_bg and C > 1:
            present[:, 0] = 0.0
        # compute per-sample per-class dice
        intersection = (probs * target_1h).sum(dim=(2, 3))  # [B,C]
        cardinality = (probs.pow(self.delta) + target_1h.pow(self.delta)).sum(dim=(2, 3))  # [B,C]
        dice_bc = (2 * intersection + self.smooth) / (cardinality + self.smooth)  # [B,C]
        # average only over present classes for each sample
        valid_counts = present.sum(dim=1).clamp_min(1.0)  # [B]
        dice_per_sample = ((dice_bc * present).sum(dim=1) / valid_counts)  # [B]
        dice_loss = 1 - dice_per_sample.mean()

        return self.lambda_ * focal_ce + (1 - self.lambda_) * dice_loss


class NPZDataset(Dataset):
    def __init__(self, files: List[str], image_key: str = 'image', mask_key: Optional[str] = 'mask', mask_keys: Optional[List[str]] = None, crop_size: Optional[int] = None, return_path: bool = False, merge_aliases: Optional[Dict[str, List[str]]] = None, augment: bool = False):
        self.files = files
        self.image_key = image_key
        self.mask_key = mask_key
        self.mask_keys = mask_keys or []
        self.crop_size = crop_size
        self.return_path = return_path
        self.augment = augment
        # Optional class map if using multiple mask keys
        self.class_map = {k: i+1 for i, k in enumerate(self.mask_keys)} if self.mask_keys else {}
        # Optional merge aliases: base_key -> [other_keys that should map to base_key's class id]
        self.merge_aliases = merge_aliases or {}

    def __len__(self):
        return len(self.files)

    def _center_crop(self, img: np.ndarray, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        H, W = img.shape[-2], img.shape[-1]
        ch = min(size, H)
        cw = min(size, W)
        y0 = (H - ch) // 2
        x0 = (W - cw) // 2
        img_c = img[..., y0:y0+ch, x0:x0+cw]
        mask_c = mask[..., y0:y0+ch, x0:x0+cw]
        return img_c, mask_c

    def _apply_augmentations(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply spatial augmentations to HSI image and mask."""
        if not self.augment:
            return img, mask
            
        # Random horizontal flip (50% chance)
        if random.random() < 0.5:
            img = np.flip(img, axis=-1).copy()  # Flip width dimension
            mask = np.flip(mask, axis=-1).copy()
        
        # Random vertical flip (50% chance)
        if random.random() < 0.5:
            img = np.flip(img, axis=-2).copy()  # Flip height dimension
            mask = np.flip(mask, axis=-2).copy()
        
        # Random rotation (90, 180, 270 degrees with 25% chance each)
        if random.random() < 0.75:  # 75% chance for any rotation
            k = random.randint(1, 3)  # 1, 2, or 3 (90, 180, 270 degrees)
            img = np.rot90(img, k=k, axes=(-2, -1)).copy()
            mask = np.rot90(mask, k=k, axes=(-2, -1)).copy()
        
        # Small random spectral perturbation (only for training, preserve spectral structure)
        if random.random() < 0.3:  # 30% chance
            noise_scale = 0.02 * np.std(img)  # 2% of image std
            spectral_noise = np.random.normal(0, noise_scale, img.shape).astype(img.dtype)
            img = img + spectral_noise
            img = np.clip(img, 0, 1)  # Keep in [0,1] range
        
        return img, mask

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with np.load(path, allow_pickle=False) as npz:
            img = npz[self.image_key].astype(np.float32)  # [C,H,W] or [H,W,C]
            # Auto-detect and convert HWC -> CHW if needed (robust to non-square H/W)
            if img.ndim == 3:
                H, W, C_last = img.shape[0], img.shape[1], img.shape[2]
                # Heuristic: spectral bands are usually small (< 128), H/W are larger
                if C_last <= 128 and H >= 64 and W >= 64:
                    # Likely HWC
                    img = np.moveaxis(img, -1, 0)  # -> CHW
            if self.mask_keys:
                # Build multi-class mask with background=0
                H, W = img.shape[-2], img.shape[-1]
                mask_mc = np.zeros((H, W), dtype=np.int64)
                for k, cls_id in self.class_map.items():
                    # Try canonical key and alias with single/double underscore variation and comma-removed variant
                    keys_to_try = [k]
                    if 'mask__' in k:
                        keys_to_try.append(k.replace('mask__', 'mask_', 1))
                    elif 'mask_' in k:
                        keys_to_try.append(k.replace('mask_', 'mask__', 1))
                    if ',' in k:
                        keys_to_try.append(k.replace(',', ''))
                    # If canonical ends with _icg, also try alias with an inserted comma before _icg
                    if k.endswith('_icg') and ',_icg' not in k:
                        keys_to_try.append(k[:-4] + ',_icg')
                    # If there are merge aliases for this base key, also consider those keys (and their alias variants)
                    for extra in self.merge_aliases.get(k, []):
                        keys_to_try.append(extra)
                        if 'mask__' in extra:
                            keys_to_try.append(extra.replace('mask__', 'mask_', 1))
                        elif 'mask_' in extra:
                            keys_to_try.append(extra.replace('mask_', 'mask__', 1))
                        if ',' in extra:
                            keys_to_try.append(extra.replace(',', ''))
                        if extra.endswith('_icg') and ',_icg' not in extra:
                            keys_to_try.append(extra[:-4] + ',_icg')
                    for kk in keys_to_try:
                        if kk in npz:
                            m = npz[kk]
                            if m.ndim == 3 and m.shape[-1] == 1:
                                m = m[..., 0]
                            m = (m > 0).astype(np.int64)
                            mask_mc = np.where(m == 1, cls_id, mask_mc)
                            break
                mask = mask_mc
            else:
                mkey = self.mask_key
                mask = npz[mkey].astype(np.int64)
                if mask.ndim == 3 and mask.shape[-1] == 1:
                    mask = mask[..., 0]

        # Simple normalization (optional): scale image per-cube to [0,1]
        minv = img.min()
        maxv = img.max()
        if maxv > minv:
            img = (img - minv) / (maxv - minv)

        # Apply augmentations (before cropping to preserve augmentation effects)
        img, mask = self._apply_augmentations(img, mask)

        # Optional center crop
        if self.crop_size is not None and self.crop_size > 0:
            img, mask = self._center_crop(img, mask, self.crop_size)

        img_t = torch.from_numpy(img)  # [C,H,W]
        mask_t = torch.from_numpy(mask)  # [H,W]
        if self.return_path:
            return img_t, mask_t, path
        else:
            return img_t, mask_t


def _unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m


def ensure_chw_batch(x: torch.Tensor) -> torch.Tensor:
    """Ensure batch tensor is (B,C,H,W). If it looks like (B,H,W,C) with a small C,
    permute to channels-first. This guards against dataset inconsistencies.
    """
    if x.dim() == 4:
        B, A, B2, C = x.shape
        # If last dim is a small band count and the first two are typical H/W, fix it
        if C <= 256 and (A >= 64 and B2 >= 64) and (A > C and B2 > C):
            return x.permute(0, 3, 1, 2).contiguous()
    return x


def train_one_run(
    train_files: List[str],
    val_files: List[str],
    args: argparse.Namespace,
    logger: Optional[CSVLogger]
) -> Tuple[str, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # No TensorBoard writer (disabled per user request)

    # Datasets / Loaders
    want_paths = bool(getattr(args, 'very_verbose', False) or getattr(args, 'print_batch_files', False))
    # Build merge aliases if requested (merge *_icg into base classes)
    merge_aliases: Dict[str, List[str]] = {}
    if getattr(args, 'merge_icg_to_base', False) and args.mask_keys:
        if 'mask__stroma' in args.mask_keys:
            merge_aliases.setdefault('mask__stroma', []).extend(['mask__stroma_icg'])
        if 'mask__artery' in args.mask_keys:
            merge_aliases.setdefault('mask__artery', []).extend(['mask__artery_icg'])

    train_ds = NPZDataset(
        train_files,
        image_key=args.image_key,
        mask_key=args.mask_key,
        mask_keys=(args.mask_keys if len(args.mask_keys) > 0 else None),
        crop_size=args.crop_size,
        return_path=want_paths,
        merge_aliases=merge_aliases if merge_aliases else None,
        augment=True,  # Enable augmentations for training
    )
    val_ds = NPZDataset(
        val_files,
        image_key=args.image_key,
        mask_key=args.mask_key,
        mask_keys=(args.mask_keys if len(args.mask_keys) > 0 else None),
        crop_size=args.crop_size,
        return_path=want_paths,
        merge_aliases=merge_aliases if merge_aliases else None,
        augment=False,  # No augmentations for validation
    )

    def _collate_with_paths(batch):
        # batch: List of tuples (img, mask, path)
        xs, ys, ps = zip(*batch)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0), list(ps)

    collate_fn = _collate_with_paths if want_paths else None
    # Optimized loader kwargs for HPC performance
    optimal_workers = min(12, args.num_workers)  # Cap at 12 for 4-GPU setup with 40 CPUs
    dl_common = {
        'num_workers': optimal_workers,
        'pin_memory': True,
        'collate_fn': collate_fn,
        'prefetch_factor': 4,
        'persistent_workers': optimal_workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_common)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size//2), shuffle=False, **dl_common)

    # Determine number of classes
    if len(train_ds.class_map) > 0:
        num_classes_model = 1 + len(train_ds.class_map)
    else:
        num_classes_model = args.num_classes

    # Peek one sample to get num bands (dataset may return (img, mask) or (img, mask, path))
    sample = train_ds[0]
    if isinstance(sample, (list, tuple)):
        sample_img = sample[0]
    else:
        sample_img = sample
    num_bands = int(sample_img.shape[0])

    model = HSIModel(
        num_bands=num_bands,
        spatial_embed_dim=args.spatial_embed_dim,
        spectral_embed_dim=args.spectral_embed_dim,
        patch_size=args.patch_size,
        global_patch_size=args.global_patch_size,
        spectral_window_sizes=args.spectral_window_sizes,
        spectral_stride=args.spectral_stride,
        spectral_pixels_per_chunk=args.spectral_pixels_per_chunk,
        num_classes=num_classes_model,
        verbose=args.verbose_model,
        use_hcmff=getattr(args, 'use_hcmff', False),
        hcmff_tokens=getattr(args, 'hcmff_tokens', 256)
    ).to(device)

    # channels_last disabled per request

    # Warm up model to initialize dynamic components before DataParallel
    if hasattr(model, 'warm_up'):
        model.warm_up(device, input_shape=(1, num_bands, args.crop_size, args.crop_size))

    # Optional compile (must happen before DataParallel)
    if getattr(args, 'compile_model', False):
        try:
            model = torch.compile(model, mode='reduce-overhead', fullgraph=False)  # best effort
            print("[Perf] torch.compile enabled")
        except Exception as e:
            print(f"[Perf] torch.compile failed (continuing uncompiled): {e}")

    # Multi-GPU support
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        total_gpus = torch.cuda.device_count()
        if getattr(args, 'force_all_gpus', False):
            used_gpus = total_gpus
        else:
            used_gpus = max(1, min(total_gpus, args.batch_size))
        device_ids = list(range(used_gpus))
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"Using DataParallel across {used_gpus}/{total_gpus} GPUs")
        if used_gpus < total_gpus and not getattr(args, 'force_all_gpus', False):
            print(f"[INFO] Limiting GPUs to {used_gpus} to avoid zero-size microbatches (batch={args.batch_size}).")

    if args.verbose_model:
        print("Model pipeline: [Spatial Local] + [Spatial Global] -> Fusion -> SpatialTokenizer || SpectralStream -> TCME -> Decoder")
        print(f"Num spectral bands detected: {num_bands}")
        if hasattr(train_ds, 'class_map') and len(train_ds.class_map) > 0:
            print("Using masks (class order): background=0, " + ", ".join(f"{k}={v}" for k,v in train_ds.class_map.items()))

    # Optimizer/Scheduler/Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.t_mult, eta_min=args.eta_min)
    loss_fn = UnifiedFocalLoss(num_classes=num_classes_model, lambda_=args.lambda_u, alpha=args.alpha, gamma=args.gamma, delta=args.delta, smooth=args.smooth, dice_exclude_bg=True)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available(), init_scale=1024)
    early = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    # Checkpoint dirs
    ensure_dir(args.save_dir)
    best_path = os.path.join(args.save_dir, "best.pt")
    last_path = os.path.join(args.save_dir, "last.pt")

    # Optional auto-resume
    start_epoch = 0
    if getattr(args, 'resume', ''):
        ckpt_path = None
        if args.resume.strip().lower() == 'auto':
            if os.path.exists(last_path):
                ckpt_path = last_path
            elif os.path.exists(best_path):
                ckpt_path = best_path
        elif os.path.exists(args.resume):
            ckpt_path = args.resume
        if ckpt_path is not None:
            print(f"[Resume] Loading checkpoint from {ckpt_path}")
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                state = ckpt.get('state_dict', None)
                if state is None and isinstance(ckpt, dict):
                    state = ckpt
                if state is not None:
                    _unwrap(model).load_state_dict(state, strict=False)
                # Try optimizer/scheduler/scaler if present
                try:
                    if 'optimizer' in ckpt:
                        optimizer.load_state_dict(ckpt['optimizer'])
                except Exception: pass
                try:
                    if 'scheduler' in ckpt:
                        scheduler.load_state_dict(ckpt['scheduler'])
                except Exception: pass
                try:
                    if 'scaler' in ckpt and hasattr(scaler, 'load_state_dict'):
                        scaler.load_state_dict(ckpt['scaler'])
                except Exception: pass
                # Seed early stopping best with last val_loss
                if 'val_loss' in ckpt:
                    try:
                        early.best = float(ckpt['val_loss'])
                    except Exception:
                        pass
                start_epoch = int(ckpt.get('epoch', -1)) + 1
                print(f"[Resume] Resuming from epoch {start_epoch}")
            except Exception as e:
                print(f"[Resume] Failed to load checkpoint: {e}")

    # Optional freezing schedule (disabled by default)
    def set_requires_grad(module: nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    # Optional per-batch text log
    batch_log_path = os.path.join(logger.run_dir, 'batches.log') if logger is not None else None

    # Prepare metrics CSV (separate from loss CSV)
    metrics_csv = os.path.join(logger.run_dir, 'metrics_val.csv') if logger is not None else None
    if metrics_csv is not None and (not os.path.exists(metrics_csv)):
        with open(metrics_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'epoch','acc','macro_dice','macro_iou','macro_precision','macro_recall','macro_specificity','macro_f1',
                'macro_dice_no_bg','macro_iou_no_bg','balanced_accuracy','kappa','mcc','auc_macro','hd_mean','hd95_mean',
                'params_million','flops_gmac','val_time_sec','gpu_mem_mb'
            ])

    for epoch in range(start_epoch, args.epochs):
        # Device consistency is handled at warmup/train transitions; avoid moving modules mid-epoch
        
        model.train()
        if args.progressive_unfreeze:
            _m = _unwrap(model)
            if epoch < 20:
                set_requires_grad(_m.local_stream, False)
                set_requires_grad(_m.global_stream, False)
                set_requires_grad(_m.spectral_stream, False)
            elif epoch < 40:
                set_requires_grad(_m.local_stream, False)
                set_requires_grad(_m.global_stream, False)
                set_requires_grad(_m.spectral_stream, True)
            else:
                set_requires_grad(_m, True)

        # Epoch start summaries
        try:
            print(f"[Epoch {epoch+1}] Train: batches={len(train_loader)}, dataset={len(train_loader.dataset)}")
        except Exception:
            pass

        train_loss = 0.0
        seen_train = 0
        verbose_once_done = False
        # Enable model workflow shape tracing either via explicit flags or when very-verbose is requested
        verbose_requested = bool(
            getattr(args, 'verbose_model', False)
            or getattr(args, 'verbose_model_once', False)
            or getattr(args, 'very_verbose', False)
        )

        for batch_idx, batch in enumerate(train_loader):
            if want_paths:
                x, y, paths = batch
            else:
                x, y = batch
            try:
                vflag = (verbose_requested and not verbose_once_done and epoch == 0 and batch_idx == 0)
                u = _unwrap(model)
                u.verbose = vflag
                if hasattr(u, 'decoder'):
                    u.decoder.verbose = vflag
            except Exception:
                pass

            x = ensure_chw_batch(x).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            def _is_nccl_broadcast_error(err: Exception) -> bool:
                s = str(err)
                return ('NCCL' in s) or ('broadcast_coalesced' in s) or ('nccl' in s)

            def _is_replica_compute_error(err: Exception) -> bool:
                s = str(err)
                return ('AcceleratorError' in s) or ('misaligned address' in s) or ('CUDA error' in s)

            # Helper to reduce runtime memory pressure on OOM
            def _reduce_memory(m: nn.Module):
                um = _unwrap(m)
                # Reduce spectral chunk size
                try:
                    if hasattr(um, 'spectral_stream') and hasattr(um.spectral_stream, 'pixels_per_chunk'):
                        um.spectral_stream.pixels_per_chunk = max(1024, int(um.spectral_stream.pixels_per_chunk // 2))
                        print(f"[VRAM] Reduced spectral pixels_per_chunk -> {um.spectral_stream.pixels_per_chunk}")
                except Exception:
                    pass
                # Reduce HCMFF tokens if used
                try:
                    if getattr(um, 'use_hcmff', False):
                        um.hcmff_tokens = max(64, int(um.hcmff_tokens // 2))
                        print(f"[VRAM] Reduced HCMFF tokens -> {um.hcmff_tokens}")
                except Exception:
                    pass

            # Memory-safe forward with retries
            def _safe_forward(_x: torch.Tensor, _y: torch.Tensor):
                attempts = 0
                while True:
                    try:
                        with _sdpa_mem_eff_ctx():
                            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                                outputs = model(_x)
                                logits = outputs['final_logits']
                                if logits.shape[-2:] != _y.shape[-2:]:
                                    logits = F.interpolate(logits, size=_y.shape[-2:], mode='bilinear', align_corners=False)
                                # Sanitize logits to avoid NaNs/Infs from upstream numerical issues
                                if not torch.isfinite(logits).all():
                                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                                # Per-batch class weights (inverse frequency), background gets lower weight
                                with torch.no_grad():
                                    binc = torch.bincount(_y.view(-1), minlength=num_classes_model).float().to(logits.device)
                                    # smooth to avoid div-by-zero; lower bg weight by factor
                                    sm = 1.0
                                    inv = 1.0 / (binc + sm)
                                    inv = inv / inv.max().clamp_min(1e-8)
                                    if inv.numel() > 1:
                                        inv[0] = inv[0] * 0.25
                                loss_local = loss_fn(logits, _y, class_weights=inv)
                                if not torch.isfinite(loss_local):
                                    print("[WARN] Non-finite loss detected; sanitizing and retrying with reduced memory.")
                                    raise RuntimeError("non-finite-loss")
                        return loss_local, logits
                    except RuntimeError as e:
                        msg = str(e).lower()
                        if ('out of memory' in msg) or ('cuda error: device-side assert triggered' in msg) or ('non-finite-loss' in msg):
                            print(f"[WARN] Forward OOM (attempt {attempts+1}). Trying to reduce memory and retry.")
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            _reduce_memory(model)
                            attempts += 1
                            if attempts >= 3:
                                print("[ERROR] Exhausted memory reduction retries.")
                                raise
                            continue
                        else:
                            raise

            try:
                replicas = torch.cuda.device_count() if (torch.cuda.is_available() and isinstance(model, nn.DataParallel)) else 1
                original_bs = x.size(0)
                if getattr(args, 'force_all_gpus', False) and replicas > 1:
                    rem = original_bs % replicas
                    if rem != 0:
                        need = replicas - rem
                        x = torch.cat([x, x[-1:].repeat(need, 1, 1, 1)], dim=0)
                        y = torch.cat([y, y[-1:].repeat(need, 1, 1)], dim=0)
                loss, logits = _safe_forward(x, y)
                if getattr(args, 'force_all_gpus', False) and replicas > 1:
                    eff_bs = x.size(0)
                    if eff_bs != original_bs and eff_bs > 0:
                        loss = loss * (original_bs / float(eff_bs))
                # Update seen count with original (pre-replication) batch size
                try:
                    seen_train += int(original_bs)
                except Exception:
                    pass
                # Per-batch verbose progress
                if getattr(args, 'very_verbose', False) and ((batch_idx % max(1, args.progress_interval)) == 0):
                    seen = min((batch_idx + 1) * original_bs, len(train_loader.dataset))
                    msg = f"[Train] Ep {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | Seen {seen}/{len(train_loader.dataset)} | Loss {loss.item():.4f}"
                    if want_paths:
                        try:
                            base_names = [os.path.basename(p) for p in paths]
                            show = base_names[:min(8, len(base_names))]
                            msg += f" | Files {show}"
                        except Exception:
                            pass
                    print(msg)
                    if batch_log_path is not None:
                        try:
                            with open(batch_log_path, 'a') as bf:
                                bf.write(msg + "\n")
                        except Exception:
                            pass
            except RuntimeError as e:
                if _is_nccl_broadcast_error(e) or _is_replica_compute_error(e):
                    print(f"[WARN] NCCL/DP error on forward: {e}. Falling back to single-GPU and retrying this batch.")
                    try:
                        if torch.cuda.is_available():
                            try:
                                for i in range(torch.cuda.device_count()):
                                    torch.cuda.set_device(i)
                                    torch.cuda.empty_cache()
                            except Exception:
                                torch.cuda.empty_cache()
                        model = _unwrap(model).to(device)
                        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.t_mult, eta_min=args.eta_min)
                        scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
                        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                            outputs = model(x)
                            logits = outputs['final_logits']
                            if logits.shape[-2:] != y.shape[-2:]:
                                logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)
                            loss = loss_fn(logits, y)
                        print(f"[INFO] Successfully switched to single-GPU path.")
                    except Exception as e2:
                        print(f"[ERROR] Fallback forward failed: {e2}")
                        raise
                else:
                    raise
            except Exception as e:
                if _is_replica_compute_error(e):
                    print(f"[WARN] DP replica compute error: {e}. Falling back to single-GPU and retrying this batch.")
                    try:
                        if torch.cuda.is_available():
                            try:
                                for i in range(torch.cuda.device_count()):
                                    torch.cuda.set_device(i)
                                    torch.cuda.empty_cache()
                            except Exception:
                                torch.cuda.empty_cache()
                        model = _unwrap(model).to(device)
                        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.t_mult, eta_min=args.eta_min)
                        scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
                        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                            outputs = model(x)
                            logits = outputs['final_logits']
                            if logits.shape[-2:] != y.shape[-2:]:
                                logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)
                            loss = loss_fn(logits, y)
                        print(f"[INFO] Successfully switched to single-GPU path.")
                    except Exception as e2:
                        print(f"[ERROR] Fallback forward failed: {e2}")
                        raise
                else:
                    raise

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
            verbose_once_done = True

        train_loss /= max(1, len(train_loader.dataset))
        # Epoch end summary (train)
        try:
            tmsg = f"[Epoch {epoch+1}] Train seen {seen_train}/{len(train_loader.dataset)} samples"
            print(tmsg)
            if batch_log_path is not None:
                with open(batch_log_path, 'a') as bf:
                    bf.write(tmsg + "\n")
        except Exception:
            pass

        # Validation
        model.eval()
        try:
            u = _unwrap(model)
            u.verbose = False
            if hasattr(u, 'decoder'):
                u.decoder.verbose = False
        except Exception:
            pass
        val_loss = 0.0
        # Metrics accumulators
        cm_accum = torch.zeros((num_classes_model, num_classes_model), dtype=torch.int64)
        y_true_flat = []
        y_pred_flat = []
        # AUC sampling accumulators
        auc_labels = []
        auc_scores = []
        auc_cap = getattr(args, 'auc_max_pixels', 200000)
        # Timing/memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        import time
        val_start = time.time()
        # HD accumulators
        hd_list = []
        hd95_list = []
        with torch.no_grad():
            try:
                print(f"[Epoch {epoch+1}] Val: batches={len(val_loader)}, dataset={len(val_loader.dataset)}")
            except Exception:
                pass
            seen_val = 0
            for vb_idx, vb in enumerate(val_loader):
                if want_paths:
                    x, y, vpaths = vb
                else:
                    x, y = vb
                # Verbose workflow tracing on first validation batch
                try:
                    vflag = (getattr(args, 'very_verbose', False) or getattr(args, 'verbose_model', False)) and (epoch == 0 and vb_idx == 0)
                    u = _unwrap(model)
                    u.verbose = vflag
                    if hasattr(u, 'decoder'):
                        u.decoder.verbose = vflag
                except Exception:
                    pass
                x = ensure_chw_batch(x).to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                replicas = torch.cuda.device_count() if (torch.cuda.is_available() and isinstance(model, nn.DataParallel)) else 1
                original_bs = x.size(0)
                if getattr(args, 'force_all_gpus', False) and replicas > 1:
                    rem = original_bs % replicas
                    if rem != 0:
                        need = replicas - rem
                        x = torch.cat([x, x[-1:].repeat(need, 1, 1, 1)], dim=0)
                        y = torch.cat([y, y[-1:].repeat(need, 1, 1)], dim=0)
                try:
                    # Reuse safe forward with guards
                    loss, logits = _safe_forward(x, y)
                    if getattr(args, 'very_verbose', False) and ((vb_idx % max(1, args.progress_interval)) == 0):
                        vmsg = f"[Val]   Ep {epoch+1}/{args.epochs} | Batch {vb_idx+1}/{len(val_loader)} | Loss {loss.item():.4f}"
                        if want_paths:
                            try:
                                vshow = [os.path.basename(p) for p in vpaths][:min(8, len(vpaths))]
                                vmsg += f" | Files {vshow}"
                            except Exception:
                                pass
                        print(vmsg)
                        if batch_log_path is not None:
                            try:
                                with open(batch_log_path, 'a') as bf:
                                    bf.write(vmsg + "\n")
                            except Exception:
                                pass
                except Exception as e:
                    s = str(e)
                    if ('misaligned address' in s) or ('AcceleratorError' in s) or ('CUDA error' in s) or ('NCCL' in s):
                        try:
                            u = _unwrap(model)
                            outputs = u(x)
                            logits = outputs['final_logits']
                            if logits.shape[-2:] != y.shape[-2:]:
                                logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)
                            loss = loss_fn(logits, y)
                            print(f"[INFO] Validation fallback to single-GPU succeeded for one batch.")
                        except Exception as e2:
                            print(f"[ERROR] Validation fallback failed: {e2}")
                            raise
                    else:
                        raise
                if getattr(args, 'force_all_gpus', False) and replicas > 1:
                    eff_bs = x.size(0)
                    if eff_bs != original_bs and eff_bs > 0:
                        loss = loss * (original_bs / float(eff_bs))
                try:
                    seen_val += int(original_bs)
                except Exception:
                    pass
                val_loss += loss.item() * original_bs
                # Metrics update
                preds = torch.argmax(logits, dim=1)
                cm_accum += _confusion_matrix_torch(preds.cpu(), y.cpu(), num_classes_model)
                y_true_flat.append(y.view(-1).cpu().numpy())
                y_pred_flat.append(preds.view(-1).cpu().numpy())
                # AUC sampling
                if auc_cap > 0:
                    probs = F.softmax(logits, dim=1).detach().cpu()
                    Bp, Cp, Hp, Wp = probs.shape
                    taken = sum(len(a) for a in auc_labels)
                    rem = max(0, auc_cap - taken)
                    if rem > 0:
                        take = min(rem, Bp * Hp * Wp)
                        idx = torch.randperm(Bp * Hp * Wp)[:take]
                        labels_b = y.view(-1).cpu()[idx]
                        scores_b = probs.permute(0,2,3,1).reshape(-1, Cp)[idx]
                        auc_labels.append(labels_b.numpy())
                        auc_scores.append(scores_b.numpy())
                # HD metrics per image per class (exclude background)
                if getattr(args, 'compute_hd', False):
                    pr_np = preds.cpu().numpy()
                    gt_np = y.cpu().numpy()
                    Cn = num_classes_model
                    for bi in range(pr_np.shape[0]):
                        for c in range(1, Cn):
                            gt_c = (gt_np[bi] == c).astype(np.uint8)
                            pr_c = (pr_np[bi] == c).astype(np.uint8)
                            if gt_c.max() == 0 and pr_c.max() == 0:
                                continue
                            hd, hd95 = _hd95_per_image(gt_c, pr_c)
                            if not np.isnan(hd): hd_list.append(hd)
                            if not np.isnan(hd95): hd95_list.append(hd95)
        val_loss /= max(1, len(val_loader.dataset))
        # Epoch end summary (val)
        try:
            vmsg = f"[Epoch {epoch+1}] Val   seen {seen_val}/{len(val_loader.dataset)} samples"
            print(vmsg)
            if batch_log_path is not None:
                with open(batch_log_path, 'a') as bf:
                    bf.write(vmsg + "\n")
        except Exception:
            pass

        # Aggregate metrics
        cm_np = cm_accum.numpy()
        stats = _per_class_stats_from_cm(cm_np)
        macro = {k: float(np.nanmean(v)) for k, v in stats.items()}
        if num_classes_model > 1:
            nobg = {k: float(np.nanmean(v[1:])) for k, v in stats.items()}
        else:
            nobg = {k: macro[k] for k in macro}
        total = cm_np.sum()
        acc = float(np.trace(cm_np) / total) if total > 0 else 0.0
        bal_acc = float(np.nanmean(stats['recall']))
        # Kappa from cm
        po = np.trace(cm_np) / total if total > 0 else 0.0
        pe = (cm_np.sum(axis=0) * cm_np.sum(axis=1)).sum() / (total * total) if total > 0 else 0.0
        kappa = float((po - pe) / (1 - pe)) if (1 - pe) > 0 else 0.0
        # MCC from cm (multiclass)
        mcc = _mcc_from_cm(cm_np)
        # AUC macro
        if auc_labels:
            y_auc = np.concatenate(auc_labels)
            s_auc = np.concatenate(auc_scores)
            try:
                auc_macro = float(roc_auc_score(y_auc, s_auc, multi_class='ovr', average='macro'))
            except Exception:
                auc_macro = float('nan')
        else:
            auc_macro = float('nan')
        # HD aggregates
        hd_mean = float(np.nanmean(hd_list)) if hd_list else float('nan')
        hd95_mean = float(np.nanmean(hd95_list)) if hd95_list else float('nan')
        # Params and FLOPs
        params_m = sum(p.numel() for p in _unwrap(model).parameters() if p.requires_grad) / 1e6
        flops_gmac = float('nan')
        try:
            if epoch == 0 and thop_profile is not None:
                dummy = torch.randn(1, num_bands, args.crop_size, args.crop_size, device=device)
                _unwrap(model).eval()
                macs, _ = thop_profile(_unwrap(model), inputs=(dummy,), verbose=False)
                flops_gmac = macs / 1e9
        except Exception:
            flops_gmac = float('nan')
        # Time/mem
        import time
        val_time_sec = time.time() - val_start
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

        # Console table
        table = [
            ['Acc', f"{acc:.4f}"],
            ['Dice(macro)', f"{macro['dice']:.4f}"],
            ['IoU(macro)', f"{macro['iou']:.4f}"],
            ['Prec(macro)', f"{macro['precision']:.4f}"],
            ['Rec(macro)', f"{macro['recall']:.4f}"],
            ['Spec(macro)', f"{macro['specificity']:.4f}"],
            ['F1(macro)', f"{macro['f1']:.4f}"],
            ['Dice(!bg)', f"{nobg['dice']:.4f}"],
            ['IoU(!bg)', f"{nobg['iou']:.4f}"],
            ['BalAcc', f"{bal_acc:.4f}"],
            ['Kappa', f"{kappa:.4f}"],
            ['MCC', f"{mcc:.4f}"],
            ['AUC(macro)', f"{auc_macro:.4f}"],
            ['HD', f"{hd_mean:.2f}"],
            ['HD95', f"{hd95_mean:.2f}"],
            ['Params(M)', f"{params_m:.2f}"],
            ['FLOPs(GMac)', f"{(flops_gmac if not np.isnan(flops_gmac) else 0):.2f}"],
            ['ValTime(s)', f"{val_time_sec:.2f}"],
            ['GPU Mem(MB)', f"{gpu_mem_mb:.1f}"],
        ]
        print("\nValidation metrics (epoch-level):")
        print(tabulate(table, headers=['Metric', 'Value'], tablefmt='github'))

        # Write CSV
        if metrics_csv is not None:
            try:
                with open(metrics_csv, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow([
                        epoch+1, acc, macro['dice'], macro['iou'], macro['precision'], macro['recall'], macro['specificity'], macro['f1'],
                        nobg['dice'], nobg['iou'], bal_acc, kappa, mcc, auc_macro, hd_mean, hd95_mean,
                        params_m, (flops_gmac if not np.isnan(flops_gmac) else ''), val_time_sec, gpu_mem_mb
                    ])
            except Exception:
                pass

        # Scheduler step (with warmup)
        if epoch < args.warmup_epochs:
            lr_scale = min(1.0, float(epoch + 1) / args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * lr_scale
        else:
            scheduler.step(epoch - args.warmup_epochs)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | LR {current_lr:.6f}")

        if logger is not None:
            logger.log_epoch(epoch + 1, train_loss, val_loss, current_lr)

        # Checkpointing (extended last.pt for resume state)
        state = _unwrap(model).state_dict()
        try:
            prev_best = torch.load(best_path, map_location='cpu').get('val_loss', float('inf')) if os.path.exists(best_path) else float('inf')
        except Exception:
            prev_best = float('inf')
        if (val_loss < prev_best):
            torch.save({'state_dict': state, 'val_loss': val_loss, 'epoch': epoch}, best_path)
        torch.save({
            'state_dict': state,
            'val_loss': val_loss,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': (scaler.state_dict() if hasattr(scaler, 'state_dict') else {})
        }, last_path)

        if early.step(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

    best = torch.load(best_path, map_location='cpu')
    try:
        import gc
        del model, train_loader, val_loader, optimizer, scheduler, scaler, loss_fn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return best_path, best.get('val_loss', float('inf'))


def discover_npz_files(root: str, pattern: str = "*.npz") -> List[str]:
    # Collect both top-level and recursive files, then deduplicate
    top = glob.glob(os.path.join(root, pattern))
    rec = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    files = sorted(set(top) | set(rec))
    if len(files) == 0:
        raise FileNotFoundError(f"No NPZ files found under {root}")
    return files


def discover_mask_keys_in_npz(files: List[str], prefix: str = 'mask_', limit: int = 1000) -> List[str]:
    """Scan up to 'limit' NPZ files and collect keys that start with prefix.
    Returns a sorted list of discovered keys.
    """
    seen = set()
    for i, p in enumerate(files):
        if i >= limit:
            break
        try:
            with np.load(p, allow_pickle=False) as npz:
                for k in npz.keys():
                    if k.startswith(prefix):
                        seen.add(k)
        except Exception:
            continue
    return sorted(seen)


def summarize_class_presence(files: List[str], mask_keys: List[str], limit: int = 2000, merge_aliases: Optional[Dict[str, List[str]]] = None) -> Dict[str, int]:
    """Count how many images contain at least one positive pixel for each mask key.
    Returns a dict key->count (over up to 'limit' files).
    """
    counts = {k: 0 for k in mask_keys}
    merge_aliases = merge_aliases or {}
    for i, p in enumerate(files):
        if i >= limit:
            break
        try:
            with np.load(p, allow_pickle=False) as npz:
                for k in mask_keys:
                    # Try key and its single/double underscore alias and comma-removed variant
                    keys_to_try = [k]
                    if 'mask__' in k:
                        keys_to_try.append(k.replace('mask__', 'mask_', 1))
                    elif 'mask_' in k:
                        keys_to_try.append(k.replace('mask_', 'mask__', 1))
                    if ',' in k:
                        keys_to_try.append(k.replace(',', ''))
                    if k.endswith('_icg') and ',_icg' not in k:
                        keys_to_try.append(k[:-4] + ',_icg')
                    # If we are merging icg into base, also consider extra aliases for this base key
                    for extra in merge_aliases.get(k, []):
                        keys_to_try.append(extra)
                        if 'mask__' in extra:
                            keys_to_try.append(extra.replace('mask__', 'mask_', 1))
                        elif 'mask_' in extra:
                            keys_to_try.append(extra.replace('mask_', 'mask__', 1))
                        if ',' in extra:
                            keys_to_try.append(extra.replace(',', ''))
                        if extra.endswith('_icg') and ',_icg' not in extra:
                            keys_to_try.append(extra[:-4] + ',_icg')
                    found = False
                    for kk in keys_to_try:
                        if kk in npz:
                            m = npz[kk]
                            if m.ndim == 3 and m.shape[-1] == 1:
                                m = m[..., 0]
                            if np.any(m > 0):
                                counts[k] += 1
                            found = True
                            break
        except Exception:
            continue
    return counts


def main():
    # Optimize CUDA memory allocation for better performance on HPC
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,roundup_power2_divisions:16')
    
    parser = argparse.ArgumentParser(description="Train HSI model on NPZ dataset (ADA-ready)")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/ssd_scratch/placenta/Placenta',
        help='Root directory containing NPZ files, with train/val subdirectories.'
    )
    parser.add_argument('--save-dir', type=str, default='saved/models', help='Directory to save checkpoints and weights (defaults here; no need to pass)')
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    # Loss hyperparameters
    parser.add_argument('--lambda-u', dest='lambda_u', type=float, default=0.5, help='Weighting between focal CE and focal Dice (0..1)')
    parser.add_argument('--alpha', type=float, default=0.25, help='Focal loss alpha')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--delta', type=float, default=0.5, help='Dice focal delta')
    parser.add_argument('--smooth', type=float, default=1e-8, help='Dice smoothing epsilon')
    parser.add_argument('--t0', type=int, default=20)
    parser.add_argument('--t-mult', type=int, default=2)
    parser.add_argument('--eta-min', type=float, default=1e-6)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min-delta', type=float, default=0.001)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--image-key', type=str, default='image')
    parser.add_argument('--mask-key', type=str, default='mask', help='Fallback single-mask key if --mask-keys is not provided')
    parser.add_argument('--mask-keys', type=str, default='mask__artery,mask__vein,mask__suture,mask__stroma,mask__stroma_icg,mask__artery_icg,mask__umbilical_cord', help='Comma-separated mask keys for multiclass; order defines class ids 1..N (0=background)')
    parser.add_argument('--ensemble-eval', action='store_true', help='Run fold ensemble on the last fold validation set')
    parser.add_argument('--log-dir', type=str, default='saved/models/logs', help='Directory to store logs (CSV/TensorBoard)')
    parser.add_argument('--run-name', type=str, default=None, help='Optional run name suffix for logs and checkpoints')
    parser.add_argument('--progressive-unfreeze', action='store_true', help='Enable progressive unfreezing schedule')
    # Model verbosity and sizes
    parser.add_argument('--verbose-model', action='store_true', help='Print detailed model forward pass shapes (first training batch only)')
    # Deprecated full-loop spam; we limit verbose to first batch by default. This flag is kept for clarity.
    parser.add_argument('--verbose-model-once', action='store_true', help='Alias: verbose model prints only once (first training batch)')
    # Training loop verbosity
    parser.add_argument('--very-verbose', action='store_true', help='Per-batch progress prints with filenames')
    parser.add_argument('--print-batch-files', action='store_true', help='Include file basenames in per-batch prints (implied by --very-verbose)')
    parser.add_argument('--progress-interval', type=int, default=1, help='Print every N batches when very-verbose is enabled')
    parser.add_argument('--spatial-embed-dim', type=int, default=128, help='Spatial embedding dimension')
    parser.add_argument('--spectral-embed-dim', type=int, default=128, help='Spectral embedding dimension')
    parser.add_argument('--patch-size', type=int, default=16, help='Spatial tokenizer patch size')
    parser.add_argument('--global-patch-size', type=int, default=4, help='Global stream patch size (stride)')
    parser.add_argument('--spectral-window-sizes', type=str, default='8,16,32', help='Comma-separated spectral window sizes for Mamba blocks')
    parser.add_argument('--spectral-stride', type=int, default=4, help='Stride for spectral sliding windows')
    parser.add_argument('--spectral-pixels-per-chunk', type=int, default=8192, help='Process spectral tokens in chunks to save memory')
    parser.add_argument('--crop-size', type=int, default=512, help='Optional center crop size (HxW) to reduce memory')
    parser.add_argument('--force-all-gpus', action='store_true', help='Use all visible GPUs even if batch-size < num_gpus by replicating microbatches and scaling loss')
    # Performance / quality-of-life flags
    parser.add_argument('--fast-mode', action='store_true', help='Enable TF32 and cudnn.benchmark for faster training (non-deterministic)')
    parser.add_argument('--compile-model', action='store_true', help='Attempt torch.compile for performance (best effort)')
    # Optional HCMFF fusion path
    parser.add_argument('--use-hcmff', action='store_true', help='Use HCMFF fusion (compresses tokens to --hcmff-tokens before fusion)')
    parser.add_argument('--hcmff-tokens', type=int, default=256, help='Number of tokens to use for HCMFF after compression (compute control)')
    # Dataset convenience
    parser.add_argument('--auto-discover-mask-keys', action='store_true', help='Scan dataset to auto-discover mask_* keys and override --mask-keys')
    parser.add_argument('--merge-icg-to-base', action='store_true', help='Merge mask__stroma_icg into mask__stroma and mask__artery_icg into mask__artery (treated as the same class)')
    # Metrics controls
    parser.add_argument('--compute-hd', action='store_true', help='Compute Hausdorff/HD95 metrics during validation (can be slow)')
    parser.add_argument('--auc-max-pixels', type=int, default=200000, help='Max pixels sampled for AUC computation during validation; set 0 to disable AUC')
    # Resume training support
    parser.add_argument('--resume', type=str, default='', help="Path to checkpoint to resume from, or 'auto' to load saved/models/last.pt if present (fallback to best.pt)")
    args = parser.parse_args()
    # Apply fast-mode backend toggles
    if getattr(args, 'fast_mode', False):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print("[Perf] Fast mode: TF32 enabled, cudnn.benchmark on, deterministic off")
        except Exception as e:
            print(f"[Perf] Failed to set fast-mode toggles: {e}")
    # Parse spectral window sizes string into list[int]
    if isinstance(args.spectral_window_sizes, str):
        try:
            args.spectral_window_sizes = [int(s.strip()) for s in args.spectral_window_sizes.split(',') if s.strip()]
        except Exception:
            print(f"[WARN] Could not parse --spectral-window-sizes='{args.spectral_window_sizes}', falling back to [8,16,32]")
            args.spectral_window_sizes = [8, 16, 32]

    # Parse mask keys
    if isinstance(args.mask_keys, str) and args.mask_keys.strip():
        args.mask_keys = [k.strip() for k in args.mask_keys.split(',') if k.strip()]
    else:
        args.mask_keys = []
    # If merging ICG into base, drop *_icg keys from the class list (their pixels will be painted into the base class)
    if getattr(args, 'merge_icg_to_base', False) and args.mask_keys:
        to_drop = []
        for icg_k, base_k in [('mask__stroma_icg', 'mask__stroma'), ('mask__artery_icg', 'mask__artery')]:
            if base_k in args.mask_keys and icg_k in args.mask_keys:
                to_drop.append(icg_k)
        if to_drop:
            args.mask_keys = [k for k in args.mask_keys if k not in to_drop]

    set_seed(args.seed)
    
    # Create organized folder structure
    # If run_name is provided, create a dedicated experiment folder
    if args.run_name:
        experiment_base = os.path.join(args.save_dir, args.run_name)
        ensure_dir(experiment_base)
        # Update paths to be within the experiment folder
        args.save_dir = os.path.join(experiment_base, 'weights')
        args.log_dir = os.path.join(experiment_base, 'logs')
    
    ensure_dir(args.save_dir)
    ensure_dir(args.log_dir)
    # One-time GPU summary to clarify multi-GPU visibility
    print_gpu_summary()

    # Resolve dataset root directory
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}.")

    # Initialize logger (now logs will go to organized structure)
    logger = CSVLogger(
        log_dir=args.log_dir,
        run_name=None,  # No additional subfolder since we're already organized
        config={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(args).items()}
    )
    print(f"Logging to: {logger.run_dir}")
    print(f"Weights will be saved to: {args.save_dir}")

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir)):
        raise FileNotFoundError(
            f"Expected 'train', 'val', and 'test' subdirectories in {data_dir}, but not all were found."
        )

    print("Found train/val/test subdirectories, running a single training and testing session.")
    train_files = discover_npz_files(train_dir)
    val_files = discover_npz_files(val_dir)
    test_files = discover_npz_files(test_dir)
    print(f"Discovered {len(train_files)} training, {len(val_files)} validation, and {len(test_files)} test files.")

    # Optionally auto-discover mask keys
    if args.auto_discover_mask_keys:
        discovered = discover_mask_keys_in_npz(train_files + val_files + test_files)
        if discovered:
            print(f"[Auto] Discovered mask keys: {discovered}")
            args.mask_keys = discovered
        else:
            print("[Auto] No mask_* keys discovered; keeping provided --mask-keys.")

    # Summarize class presence across dataset and write to run dir
    if args.mask_keys:
        # Build merge aliases if requested
        presence_merge_aliases: Dict[str, List[str]] = {}
        if getattr(args, 'merge_icg_to_base', False):
            if 'mask__stroma' in args.mask_keys:
                presence_merge_aliases.setdefault('mask__stroma', []).extend(['mask__stroma_icg'])
            if 'mask__artery' in args.mask_keys:
                presence_merge_aliases.setdefault('mask__artery', []).extend(['mask__artery_icg'])

        presence_all = summarize_class_presence(train_files + val_files + test_files, args.mask_keys, merge_aliases=presence_merge_aliases)
        presence_train = summarize_class_presence(train_files, args.mask_keys, merge_aliases=presence_merge_aliases)
        presence_val = summarize_class_presence(val_files, args.mask_keys, merge_aliases=presence_merge_aliases)
        presence_test = summarize_class_presence(test_files, args.mask_keys, merge_aliases=presence_merge_aliases)
        # Print concise summary
        print("Class presence (images with >0 pixels per class):")
        for k in args.mask_keys:
            print(f"  - {k}: total={presence_all.get(k, 0)} | train={presence_train.get(k, 0)} | val={presence_val.get(k, 0)} | test={presence_test.get(k, 0)}")
        # Save CSV with per-split columns
        try:
            csv_path = os.path.join(logger.run_dir, 'class_presence.csv')
            with open(csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["mask_key", "total", "train", "val", "test"])
                for k in args.mask_keys:
                    w.writerow([k, presence_all.get(k, 0), presence_train.get(k, 0), presence_val.get(k, 0), presence_test.get(k, 0)])
            print(f"Saved class presence summary to {csv_path}")
        except Exception as e:
            print(f"[WARN] Could not write class presence CSV: {e}")

    # Single training run
    best_path, best_loss = train_one_run(train_files, val_files, args, logger)
    print(f"--- Training Complete ---")
    print(f"Best Val Loss: {best_loss:.4f} | Model saved to: {best_path}")
    print(f"-------------------------")

    print("\nTo evaluate the best model on the test set, run the following command:")
    print(f"python tests/evaluate.py --model-path {best_path} --data-dir {args.data_dir}")

    # Close logger
    logger.close()


if __name__ == '__main__':
    main()