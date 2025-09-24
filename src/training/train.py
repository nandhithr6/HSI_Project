"""
Production-ready HSI training script for ADA servers

Features:
- NPZ dataset loading from a directory (one .npz per sample with image+mask)
- 5-fold cross-validation (shuffle, fixed seed)
- Joint spatial+spectral augmentations
- Unified Focal Loss (lambda-weighted focal CE + focal Dice)
- AdamW, gradient clipping, cosine warm restarts + 5-epoch warmup
- Mixed precision (AMP), early stopping, checkpointing per fold
- Optional ensemble over folds

Notes:
- On ADA, datasets live under /ssd_scratch; pass --data-dir "/ssd_scratch/<user>/<dataset>"
- Checkpoints/weights saved to saved/models by default (tracked via Git LFS)
"""

import os
import glob
import argparse
import random
import csv
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from typing import List, Tuple, Optional, Dict

try:
    from torch.utils.tensorboard import SummaryWriter  # requires tensorboard pkg
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from src.models import HSIModel


# ===================== UTILS =====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class CSVLogger:
    """Lightweight CSV logger with optional TensorBoard support.

    Creates a timestamped run directory under log_dir and writes:
    - metrics.csv: per-epoch metrics (fold, epoch, train_loss, val_loss, lr)
    - config.json: run hyperparameters
    - tensorboard/ (optional): TensorBoard event files if available
    """

    def __init__(self, log_dir: str, run_name: Optional[str], config: Dict):
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        base = ts if not run_name else f"{ts}-{run_name}"
        self.run_dir = os.path.join(log_dir, base)
        ensure_dir(self.run_dir)
        self.csv_path = os.path.join(self.run_dir, 'metrics.csv')
        self.tb_dir = os.path.join(self.run_dir, 'tensorboard')

        # Write config
        with open(os.path.join(self.run_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # Init CSV with header
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['fold', 'epoch', 'train_loss', 'val_loss', 'lr'])

        # Init TensorBoard if available
        self.writer = None
        if SummaryWriter is not None:
            try:
                ensure_dir(self.tb_dir)
                self.writer = SummaryWriter(log_dir=self.tb_dir)
            except Exception:
                self.writer = None

    def log_epoch(self, fold: int, epoch: int, train_loss: float, val_loss: float, lr: float):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([fold, epoch, train_loss, val_loss, lr])
        if self.writer is not None:
            self.writer.add_scalar(f'Fold{fold}/TrainLoss', train_loss, epoch)
            self.writer.add_scalar(f'Fold{fold}/ValLoss', val_loss, epoch)
            self.writer.add_scalar(f'Fold{fold}/LR', lr, epoch)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


# ===================== DATASET =====================
class NPZHSIDataset(Dataset):
    """NPZ dataset: one file per sample, containing image and mask.

    Expected keys (configurable):
    - image_key: 'image' (array shape [B,H,W] or [H,W,B])
    - mask_key: 'mask' (array shape [H,W])
    """

    def __init__(self,
                 files: List[str],
                 image_key: str = 'image',
                 mask_key: str = 'mask',
                 augment: bool = True,
                 spectral_dropout_p: float = 0.2,
                 spectral_dropout_ratio: float = 0.1,
                 crop_size: Optional[int] = None):
        self.files = files
        self.image_key = image_key
        self.mask_key = mask_key
        self.augment = augment
        self.spectral_dropout_p = spectral_dropout_p
        self.spectral_dropout_ratio = spectral_dropout_ratio
        self.crop_size = crop_size

        # Discover all mask_* keys across the dataset for global class mapping
        all_mask_keys = set()
        for f in files:
            try:
                data = np.load(f)
                keys = [k for k in data.keys() if k.startswith('mask_') and k != 'wavelengths']
                all_mask_keys.update(keys)
            except Exception:
                continue
        self.class_keys = sorted(all_mask_keys)
        self.class_map = {k: i+1 for i, k in enumerate(self.class_keys)}  # 0=background
        if len(self.class_map) > 0:
            print(f"[INFO] Discovered mask classes: background=0, " + ", ".join(f"{k}={v}" for k,v in self.class_map.items()))
        else:
            print("[WARN] No mask_* keys found in dataset; will fallback to 'mask' if present.")

    def __len__(self):
        return len(self.files)

    def _load_npz(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(path)
        # Try to infer keys if not present
        img_key = self.image_key if self.image_key in data else list(data.keys())[0]
        img = data[img_key]
        # Ensure image shape [B,H,W] by assuming the smallest dim is bands
        if img.ndim == 3:
            dims = list(img.shape)
            band_axis = int(np.argmin(dims))
            if band_axis != 0:
                img = np.moveaxis(img, band_axis, 0)
        else:
            raise ValueError(f"Unsupported image ndim: {img.ndim} in {path}")

        # Ensure mask shape [H,W]
        H, W = img.shape[1], img.shape[2]
        label = np.zeros((H, W), dtype=np.int64)
        # Use discovered class_map for mask_* keys
        found_any = False
        for k, class_id in self.class_map.items():
            if k in data:
                m = data[k]
                if m.ndim == 3:
                    m = np.squeeze(m)
                if m.ndim == 1 and m.size == H * W:
                    m = m.reshape(H, W)
                if m.ndim != 2 or m.shape != (H, W):
                    continue
                m_bin = (m > 0).astype(np.uint8)
                new_pixels = (label == 0) & (m_bin > 0)
                label[new_pixels] = class_id
                found_any = True
        # Fallback: if no mask_* found, try 'mask'
        if not found_any and 'mask' in data:
            m = data['mask']
            if m.ndim == 3:
                m = np.squeeze(m)
            if m.ndim == 1 and m.size == H * W:
                m = m.reshape(H, W)
            if m.ndim == 2 and m.shape == (H, W):
                label = (m > 0).astype(np.int64)
        elif not found_any:
            print(f"[WARN] No valid mask keys found in {path}; creating zero mask [{H},{W}].")
        return img.astype(np.float32), label.astype(np.int64)

    def _augment(self, img: np.ndarray, msk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Ensure mask is 2D before spatial ops; otherwise, skip spatial aug on mask
        mask_is_2d = (msk.ndim == 2)
        # Joint flips
        if random.random() < 0.5:
            img = np.flip(img, axis=2)
            if mask_is_2d:
                msk = np.flip(msk, axis=1)
        if random.random() < 0.5:
            img = np.flip(img, axis=1)
            if mask_is_2d:
                msk = np.flip(msk, axis=0)
        # 90-degree rotations
        if random.random() < 0.5:
            k = random.randint(1, 3)
            img = np.rot90(img, k=k, axes=(1, 2))
            if mask_is_2d:
                msk = np.rot90(msk, k=k, axes=(0, 1))
        # Brightness/contrast jitter
        if random.random() < 0.3:
            scale = 0.9 + 0.2 * random.random()
            shift = 0.05 * (random.random() - 0.5)
            img = img * scale + shift
        # Gaussian noise
        if random.random() < 0.3:
            img = img + np.random.normal(0, 0.01, img.shape).astype(img.dtype)
        # Spectral-band dropout
        if random.random() < self.spectral_dropout_p:
            n_bands = img.shape[0]
            n_drop = max(1, int(n_bands * self.spectral_dropout_ratio))
            drop_idx = np.random.choice(n_bands, size=n_drop, replace=False)
            img[drop_idx, :, :] = 0
        return img, msk

    def __getitem__(self, idx):
        img, msk = self._load_npz(self.files[idx])
        if self.augment:
            img, msk = self._augment(img, msk)
        # Optional center crop to reduce memory
        if self.crop_size is not None:
            H, W = img.shape[1], img.shape[2]
            cs = int(self.crop_size)
            if H >= cs and W >= cs:
                top = (H - cs) // 2
                left = (W - cs) // 2
                img = img[:, top:top+cs, left:left+cs]
                if msk.ndim == 2:
                    msk = msk[top:top+cs, left:left+cs]
        # Normalize per-band to zero mean, unit variance (robust)
        eps = 1e-6
        mean = img.reshape(img.shape[0], -1).mean(axis=1, keepdims=True)
        std = img.reshape(img.shape[0], -1).std(axis=1, keepdims=True)
        img = (img - mean[:, None, :]) / (std[:, None, :] + eps)
        # Ensure positive strides and contiguous memory before converting to torch
        img = np.ascontiguousarray(img, dtype=np.float32)
        msk = np.ascontiguousarray(msk, dtype=np.int64)
        return torch.from_numpy(img), torch.from_numpy(msk)


# ===================== LOSS =====================
class UnifiedFocalLoss(nn.Module):
    def __init__(self, num_classes=5, lambda_=0.5, alpha=0.25, gamma=2.0, delta=0.5, smooth=1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_ = lambda_
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # Resize targets to logits size if needed (decoder outputs 512x512)
        if logits.shape[-2:] != targets.shape[-2:]:
            targets = F.interpolate(targets.unsqueeze(1).float(), size=logits.shape[-2:], mode='nearest').squeeze(1).long()

        # Focal Cross-Entropy
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal_ce = self.alpha * (1 - pt) ** self.gamma * ce
        focal_ce = focal_ce.mean()

        # Focal Dice
        probs = torch.softmax(logits, dim=1)
        with torch.no_grad():
            targets_onehot = torch.zeros_like(probs)
            targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        focal_dice = self.alpha * (1 - dice) ** self.gamma * dice
        focal_dice = 1 - focal_dice.mean()

        return self.lambda_ * focal_ce + (1 - self.lambda_) * focal_dice


# ===================== TRAIN/VAL =====================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float('inf')
        self.stop = False

    def step(self, val_loss: float):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        self.stop = self.counter >= self.patience
        return self.stop


def train_one_fold(fold: int,
                   train_files: List[str],
                   val_files: List[str],
                   args,
                   logger: Optional[CSVLogger] = None) -> Tuple[str, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Fold {fold}] Device: {device}")

    # Datasets/Loaders
    train_ds = NPZHSIDataset(train_files, image_key=args.image_key, mask_key=args.mask_key, augment=True, crop_size=args.crop_size)
    val_ds = NPZHSIDataset(val_files, image_key=args.image_key, mask_key=args.mask_key, augment=False, crop_size=args.crop_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    sample_img, _ = train_ds[0]
    num_bands = sample_img.shape[0]
    model = HSIModel(
        num_bands=num_bands,
        spatial_embed_dim=args.spatial_embed_dim,
        spectral_embed_dim=args.spectral_embed_dim,
        patch_size=args.patch_size,
        global_patch_size=args.global_patch_size,
        spectral_window_sizes=args.spectral_window_sizes,
        spectral_stride=args.spectral_stride,
        spectral_pixels_per_chunk=args.spectral_pixels_per_chunk,
        num_classes=args.num_classes,
        verbose=args.verbose_model
    ).to(device)
    # Multi-GPU support
    def _unwrap(m):
        return m.module if isinstance(m, nn.DataParallel) else m
    if torch.cuda.device_count() > 1 and args.verbose_model:
        print(f"[Fold {fold}] Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if args.verbose_model:
        # One-time pipeline summary for clarity
        print("Model pipeline: [Spatial Local] + [Spatial Global] -> Fusion -> SpatialTokenizer || SpectralStream -> TCME -> HCMFF -> Decoder")
        print(f"Num spectral bands detected: {num_bands}")
        # Print discovered mask classes from dataset
        if hasattr(train_ds, 'class_map') and len(train_ds.class_map) > 0:
            print(f"Using masks (class order): background=0, " + ", ".join(f"{k}={v}" for k,v in train_ds.class_map.items()))

    # Optimizer/Scheduler/Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.t_mult, eta_min=args.eta_min)
    loss_fn = UnifiedFocalLoss(num_classes=args.num_classes, lambda_=args.lambda_u, alpha=args.alpha, gamma=args.gamma, delta=args.delta, smooth=args.smooth)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    early = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    # Checkpoint dirs
    fold_dir = os.path.join(args.save_dir, f"fold_{fold}")
    ensure_dir(fold_dir)
    best_path = os.path.join(fold_dir, "best.pt")
    last_path = os.path.join(fold_dir, "last.pt")

    # Optional freezing schedule (disabled by default)
    def set_requires_grad(module: nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    for epoch in range(args.epochs):
        model.train()
        if args.progressive_unfreeze:
            _m = _unwrap(model)
            if epoch < 20:
                # freeze most of backbone (example: spatial and spectral streams)
                set_requires_grad(_m.local_stream, False)
                set_requires_grad(_m.global_stream, False)
                set_requires_grad(_m.spectral_stream, False)
            elif epoch < 40:
                set_requires_grad(_m.local_stream, False)
                set_requires_grad(_m.global_stream, False)
                set_requires_grad(_m.spectral_stream, True)
            else:
                set_requires_grad(_m, True)

        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(x)
                logits = outputs['final_logits']
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                outputs = model(x)
                logits = outputs['final_logits']
                loss = loss_fn(logits, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        # Scheduler step (with warmup)
        if epoch < args.warmup_epochs:
            lr_scale = min(1.0, float(epoch + 1) / args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * lr_scale
        else:
            scheduler.step(epoch - args.warmup_epochs)

        # Current LR (from first param group)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Fold {fold}] Epoch {epoch+1}/{args.epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | LR {current_lr:.6f}")

        # Logging
        if logger is not None:
            logger.log_epoch(fold, epoch + 1, train_loss, val_loss, current_lr)

        # Checkpointing
        state = _unwrap(model).state_dict()
        if not os.path.exists(best_path) or val_loss < torch.load(best_path, map_location='cpu')['val_loss']:
            torch.save({'state_dict': state, 'val_loss': val_loss, 'epoch': epoch}, best_path)
        torch.save({'state_dict': state, 'val_loss': val_loss, 'epoch': epoch}, last_path)

        # Early stopping
        if early.step(val_loss):
            print(f"[Fold {fold}] Early stopping at epoch {epoch+1}")
            break

    best = torch.load(best_path, map_location='cpu')
    return best_path, best['val_loss']


def discover_npz_files(root: str, pattern: str = "*.npz") -> List[str]:
    files = sorted(glob.glob(os.path.join(root, pattern)))
    if len(files) == 0:
        # try recursive
        files = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if len(files) == 0:
        raise FileNotFoundError(f"No NPZ files found under {root}")
    return files


def main():
    parser = argparse.ArgumentParser(description="Train HSI model on NPZ dataset (ADA-ready)")
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Root directory containing NPZ files. If omitted, uses env HSI_DATA_DIR or \'/ssd_scratch/placenta/Placenta\''
    )
    parser.add_argument('--save-dir', type=str, default='saved/models', help='Directory to save checkpoints and weights')
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--folds', type=int, default=5)
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
    parser.add_argument('--mask-keys', type=str, default='mask_artery,mask_stroma,mask_specular_reflection', help='Comma-separated mask keys for multiclass; order defines class ids 1..N (0=background)')
    parser.add_argument('--ensemble-eval', action='store_true', help='Run fold ensemble on the last fold validation set')
    parser.add_argument('--log-dir', type=str, default='saved/models/logs', help='Directory to store logs (CSV/TensorBoard)')
    parser.add_argument('--run-name', type=str, default=None, help='Optional run name suffix for logs and checkpoints')
    parser.add_argument('--progressive-unfreeze', action='store_true', help='Enable progressive unfreezing schedule')
    # Model verbosity and sizes
    parser.add_argument('--verbose-model', action='store_true', help='Print detailed model forward pass shapes')
    parser.add_argument('--spatial-embed-dim', type=int, default=256, help='Spatial embedding dimension')
    parser.add_argument('--spectral-embed-dim', type=int, default=128, help='Spectral embedding dimension')
    parser.add_argument('--patch-size', type=int, default=16, help='Spatial tokenizer patch size')
    parser.add_argument('--global-patch-size', type=int, default=4, help='Global stream patch size (stride)')
    parser.add_argument('--spectral-window-sizes', type=str, default='8,16,32', help='Comma-separated spectral window sizes for Mamba blocks')
    parser.add_argument('--spectral-stride', type=int, default=4, help='Stride for spectral sliding windows')
    parser.add_argument('--spectral-pixels-per-chunk', type=int, default=8192, help='Process spectral tokens in chunks to save memory')
    parser.add_argument('--crop-size', type=int, default=None, help='Optional center crop size (HxW) to reduce memory')
    args = parser.parse_args()
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

    set_seed(args.seed)
    ensure_dir(args.save_dir)
    ensure_dir(args.log_dir)

    # Resolve dataset root directory with sensible ADA defaults
    data_dir = args.data_dir or os.environ.get('HSI_DATA_DIR') or '/ssd_scratch/placenta/Placenta'
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. "
            "Pass --data-dir, set HSI_DATA_DIR, or create the directory."
        )

    files = discover_npz_files(data_dir)
    print(f"Discovered {len(files)} NPZ files under {data_dir}")

    # Initialize logger
    logger = CSVLogger(
        log_dir=args.log_dir,
        run_name=args.run_name,
        config={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(args).items()}
    )
    print(f"Logging to: {logger.run_dir}")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_paths = []
    fold_losses = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(files), start=1):
        train_files = [files[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]
        best_path, best_loss = train_one_fold(fold, train_files, val_files, args, logger)
        fold_paths.append(best_path)
        fold_losses.append(best_loss)
        print(f"[Fold {fold}] Best Val Loss: {best_loss:.4f} | Saved: {best_path}")

    print("=== Cross-Validation Summary ===")
    for i, (p, l) in enumerate(zip(fold_paths, fold_losses), start=1):
        print(f"Fold {i}: {l:.4f} ({p})")

    if args.ensemble_eval and len(fold_paths) > 1:
        # Run a simple ensemble on the last fold's validation set
        print("Running simple ensemble on last fold val set...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        splits_list = list(KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(files))
        val_idx_last = splits_list[-1][1]
        val_files = [files[i] for i in val_idx_last]
        val_ds = NPZHSIDataset(val_files, image_key=args.image_key, mask_key=args.mask_key, mask_keys=args.mask_keys, augment=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        models_list = []
        for p in fold_paths:
            ckpt = torch.load(p, map_location='cpu')
            # Reconstruct model
            sample_img, _ = val_ds[0]
            num_bands = sample_img.shape[0]
            m = HSIModel(
                num_bands=num_bands,
                spatial_embed_dim=args.spatial_embed_dim,
                spectral_embed_dim=args.spectral_embed_dim,
                patch_size=args.patch_size,
                global_patch_size=args.global_patch_size,
                spectral_window_sizes=args.spectral_window_sizes,
                spectral_stride=args.spectral_stride,
                spectral_pixels_per_chunk=args.spectral_pixels_per_chunk,
                num_classes=args.num_classes,
                verbose=args.verbose_model
            ).to(device)
            m.load_state_dict(ckpt['state_dict'])
            m.eval()
            models_list.append(m)

        all_preds = []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                logits_sum = None
                for m in models_list:
                    out = m(x)
                    logits = out['final_logits']
                    if logits_sum is None:
                        logits_sum = logits
                    else:
                        logits_sum = logits_sum + logits
                preds = torch.argmax(logits_sum, dim=1).cpu().numpy()
                all_preds.append(preds)
    all_preds = np.concatenate(all_preds, axis=0)
    print("Ensemble predictions shape:", all_preds.shape)

    # Close logger
    logger.close()


if __name__ == '__main__':
    main()
