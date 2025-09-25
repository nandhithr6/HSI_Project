
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
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    if __name__ == "__main__":
        main()
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
        num_classes=num_classes_model,
        verbose=args.verbose_model
    ).to(device)
    # Multi-GPU support
    def _unwrap(m):
        return m.module if isinstance(m, nn.DataParallel) else m
    if torch.cuda.device_count() > 1:
        total_gpus = torch.cuda.device_count()
        if getattr(args, 'force_all_gpus', False):
            used_gpus = total_gpus
        else:
            used_gpus = max(1, min(total_gpus, args.batch_size))
        device_ids = list(range(used_gpus))
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"[Fold {fold}] Using DataParallel across {used_gpus}/{total_gpus} GPUs")
        if used_gpus < total_gpus and not getattr(args, 'force_all_gpus', False):
            print(f"[INFO] Limiting GPUs to {used_gpus} to avoid zero-size microbatches (batch={args.batch_size}).")
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
    loss_fn = UnifiedFocalLoss(num_classes=num_classes_model, lambda_=args.lambda_u, alpha=args.alpha, gamma=args.gamma, delta=args.delta, smooth=args.smooth)
    scaler = torch.amp.GradScaler(device_type='cuda', enabled=torch.cuda.is_available(), init_scale=1024)
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
        # Enable very verbose model logs only on the first training batch per fold
        verbose_once_done = False
        verbose_requested = bool(getattr(args, 'verbose_model', False) or getattr(args, 'verbose_model_once', False))
        for batch_idx, (x, y) in enumerate(train_loader):
            # Toggle model + submodules verbose only for the very first batch of the first epoch if requested
            try:
                vflag = (verbose_requested and not verbose_once_done and epoch == 0 and batch_idx == 0)
                u = _unwrap(model)
                u.verbose = vflag
                if hasattr(u, 'hcmff'):
                    u.hcmff.verbose = vflag
                if hasattr(u, 'decoder'):
                    u.decoder.verbose = vflag
            except Exception:
                pass
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # Forward with automatic DP->single-GPU fallback on NCCL/broadcast errors
            def _is_nccl_broadcast_error(err: Exception) -> bool:
                s = str(err)
                return ('NCCL' in s) or ('broadcast_coalesced' in s) or ('nccl' in s)
            def _is_replica_compute_error(err: Exception) -> bool:
                s = str(err)
                # Catch common DP replica failures
                return ('AcceleratorError' in s) or ('misaligned address' in s) or ('CUDA error' in s)

            try:
                # If forcing all GPUs, ensure batch divisible by number of replicas by light replication
                replicas = torch.cuda.device_count() if (torch.cuda.is_available() and isinstance(model, nn.DataParallel)) else 1
                original_bs = x.size(0)
                if getattr(args, 'force_all_gpus', False) and replicas > 1:
                    rem = original_bs % replicas
                    if rem != 0:
                        need = replicas - rem
                        x = torch.cat([x, x[-1:].repeat(need, 1, 1, 1)], dim=0)
                        y = torch.cat([y, y[-1:].repeat(need, 1, 1)], dim=0)
                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    outputs = model(x)
                    logits = outputs['final_logits']
                    loss = loss_fn(logits, y)
                    # Rescale loss to account for any temporary replication so that effective batch remains original_bs
                    if getattr(args, 'force_all_gpus', False) and replicas > 1:
                        eff_bs = x.size(0)
                        if eff_bs != original_bs and eff_bs > 0:
                            loss = loss * (original_bs / float(eff_bs))
            except RuntimeError as e:
                if _is_nccl_broadcast_error(e) or _is_replica_compute_error(e):
                    print(f"[Fold {fold}] [WARN] NCCL/DP error on forward: {e}. Falling back to single-GPU and retrying this batch.")
                    # Unwrap DP and rebuild optimizer/scheduler/scaler
                    try:
                        # Free CUDA caches on all visible devices
                        if torch.cuda.is_available():
                            try:
                                for i in range(torch.cuda.device_count()):
                                    torch.cuda.set_device(i)
                                    torch.cuda.empty_cache()
                            except Exception:
                                torch.cuda.empty_cache()
                        # Switch to single-GPU model
                        model = _unwrap(model).to(device)
                        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.t_mult, eta_min=args.eta_min)
                        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
                        # Retry forward once
                        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                            outputs = model(x)
                            logits = outputs['final_logits']
                            loss = loss_fn(logits, y)
                        print(f"[Fold {fold}] [INFO] Successfully switched to single-GPU path.")
                    except Exception as e2:
                        print(f"[Fold {fold}] [ERROR] Fallback forward failed: {e2}")
                        raise
                else:
                    raise
            except Exception as e:
                # Last-chance fallback for unexpected DP replica errors
                if _is_replica_compute_error(e):
                    print(f"[Fold {fold}] [WARN] DP replica compute error: {e}. Falling back to single-GPU and retrying this batch.")
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
                        scaler = torch.amp.GradScaler(device_type='cuda', enabled=torch.cuda.is_available())
                        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                            outputs = model(x)
                            logits = outputs['final_logits']
                            loss = loss_fn(logits, y)
                        print(f"[Fold {fold}] [INFO] Successfully switched to single-GPU path.")
                    except Exception as e2:
                        print(f"[Fold {fold}] [ERROR] Fallback forward failed: {e2}")
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
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        # Silence model and submodules verbose during validation to avoid loops
        try:
            u = _unwrap(model)
            u.verbose = False
            if hasattr(u, 'hcmff'):
                u.hcmff.verbose = False
            if hasattr(u, 'decoder'):
                u.decoder.verbose = False
        except Exception:
            pass
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                # Replicate batch if forcing all GPUs so every replica gets data
                replicas = torch.cuda.device_count() if (torch.cuda.is_available() and isinstance(model, nn.DataParallel)) else 1
                original_bs = x.size(0)
                if getattr(args, 'force_all_gpus', False) and replicas > 1:
                    rem = original_bs % replicas
                    if rem != 0:
                        need = replicas - rem
                        x = torch.cat([x, x[-1:].repeat(need, 1, 1, 1)], dim=0)
                        y = torch.cat([y, y[-1:].repeat(need, 1, 1)], dim=0)
                try:
                    outputs = model(x)
                    logits = outputs['final_logits']
                    loss = loss_fn(logits, y)
                except Exception as e:
                    # If DP causes a replica error during eval, fall back to single GPU and retry this batch
                    s = str(e)
                    if ('misaligned address' in s) or ('AcceleratorError' in s) or ('CUDA error' in s) or ('NCCL' in s):
                        try:
                            u = _unwrap(model)
                            outputs = u(x)
                            logits = outputs['final_logits']
                            loss = loss_fn(logits, y)
                            print(f"[Fold {fold}] [INFO] Validation fallback to single-GPU succeeded for one batch.")
                        except Exception as e2:
                            print(f"[Fold {fold}] [ERROR] Validation fallback failed: {e2}")
                            raise
                    else:
                        raise
                if getattr(args, 'force_all_gpus', False) and replicas > 1:
                    eff_bs = x.size(0)
                    if eff_bs != original_bs and eff_bs > 0:
                        loss = loss * (original_bs / float(eff_bs))
                val_loss += loss.item() * original_bs
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
    # Proactive cleanup to reduce CUDA fragmentation before next fold
    try:
        import gc
        del model, train_loader, val_loader, optimizer, scheduler, scaler, loss_fn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
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
        default='/ssd_scratch/placenta/Placenta',
        help='Root directory containing NPZ files, with train/val subdirectories.'
    )
    parser.add_argument('--save-dir', type=str, default='saved/models', help='Directory to save checkpoints and weights (defaults here; no need to pass)')
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
    parser.add_argument('--verbose-model', action='store_true', help='Print detailed model forward pass shapes (first training batch only)')
    # Deprecated full-loop spam; we limit verbose to first batch by default. This flag is kept for clarity.
    parser.add_argument('--verbose-model-once', action='store_true', help='Alias: verbose model prints only once (first training batch)')
    parser.add_argument('--spatial-embed-dim', type=int, default=256, help='Spatial embedding dimension')
    parser.add_argument('--spectral-embed-dim', type=int, default=128, help='Spectral embedding dimension')
    parser.add_argument('--patch-size', type=int, default=16, help='Spatial tokenizer patch size')
    parser.add_argument('--global-patch-size', type=int, default=4, help='Global stream patch size (stride)')
    parser.add_argument('--spectral-window-sizes', type=str, default='8,16,32', help='Comma-separated spectral window sizes for Mamba blocks')
    parser.add_argument('--spectral-stride', type=int, default=4, help='Stride for spectral sliding windows')
    parser.add_argument('--spectral-pixels-per-chunk', type=int, default=8192, help='Process spectral tokens in chunks to save memory')
    parser.add_argument('--crop-size', type=int, default=512, help='Optional center crop size (HxW) to reduce memory')
    parser.add_argument('--force-all-gpus', action='store_true', help='Use all visible GPUs even if batch-size < num_gpus by replicating microbatches and scaling loss')
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
    # One-time GPU summary to clarify multi-GPU visibility
    print_gpu_summary()

    # Resolve dataset root directory
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}.")

    # Initialize logger
    logger = CSVLogger(
        log_dir=args.log_dir,
        run_name=args.run_name,
        config={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(args).items()}
    )
    print(f"Logging to: {logger.run_dir}")

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

    # Single training run
    best_path, best_loss = train_one_fold(1, train_files, val_files, args, logger)
    print(f"--- Training Complete ---")
    print(f"Best Val Loss: {best_loss:.4f} | Model saved to: {best_path}")
    print(f"-------------------------")

    print("\nTo evaluate the best model on the test set, run the following command:")
    print(f"python tests/evaluate.py --model-path {best_path} --data-dir {args.data_dir}")

    # Close logger
    logger.close()


if __name__ == '__main__':
    main()
