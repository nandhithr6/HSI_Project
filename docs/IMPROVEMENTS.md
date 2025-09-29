# HSI Project – Improvements and How-To Guide

This document summarizes all major updates we implemented across the codebase and provides practical guidance for training, evaluating, and scaling the HSI segmentation model.

## Overview

Focus areas:
- Clearer architecture flow: TCME as token enhancer vs HCMFF as fusion, with an optional TCME → HCMFF pipeline.
- Better logging and organized run outputs to debug per-batch and per-block execution.
- Stronger training defaults for class imbalance and numerical stability.
- High-efficiency GPU usage on 1–4 GPUs (12 GB cards): TF32, channels_last, gradient checkpointing, accumulation, EMA, memory-friendly SDPA.
- Robust evaluation: safer AUC/HD metrics and confusion matrix exports.
- Distributed training: finalized DistributedDataParallel (DDP) support with torchrun, rank-0 logging/checkpointing, and synchronized validation metrics.
- Resume and graceful interrupt checkpoints.

Key touched files:
- `src/training/train.py`
- `src/models/full_model.py`
- `src/models/spatial_stream/global_main.py`
- `src/models/spectral_stream/main.py`
- `tests/evaluate.py`


## Architecture clarity: TCME vs HCMFF

- TCME (TokenCrossModalEnhancer): enhancer operating over spatial and spectral tokens using cross-attention to refine token quality.
- HCMFF (HierarchicalCrossModalityFrequencyFusion): fusion module in frequency domain; optionally compresses tokens to `--hcmff-tokens` for efficiency.
- New option: `--both-fusions` to apply TCME first and then refine with HCMFF (TCME → HCMFF).
- Implementation safeguards:
  - Token alignment and optional projection to meet HCMFF token count when enabled.
  - Model “warm_up” initializes dynamic parts (decoder, projections) before wrapping for multi-GPU, avoiding runtime module creation.


## Logging and run organization

- Per-batch progress: `--very-verbose` with `--progress-interval N` and optional filenames via `--print-batch-files`.
- Per-block workflow logs across streams/fusions/decoder: `--workflow-logs` with `--workflow-interval N` gate; logs go to console and `<run>/logs/batches.log`.
- Organized outputs when `--run-name NAME` is provided:
  - `<common_base>/<NAME>/weights`: checkpoints (`best.pt`, `last.pt`, `epoch_XXX.pt`)
  - `<common_base>/<NAME>/logs`: `epochs.csv`, `metrics_val.csv`, `batches.log`, `config.json`
- Class presence scanning across train/val/test saved to `class_presence.csv` with a concise terminal summary.


## Training improvements (robustness + imbalance)

- Unified loss: Lambda-weighted Focal CE + Focal Dice with safer defaults for imbalance.
  - Flags: `--alpha`, `--gamma`, `--lambda-u`, `--delta`, `--smooth`.
  - Per-batch dynamic class weights: aggressively down-weights background and boosts very rare classes.
- Scheduler: Cosine annealing with warm restarts and longer warm-up.
- Early stopping with `--patience` and `--min-delta`.
- Gradient clipping: `--max-grad-norm`.
- Optional progressive unfreezing: `--progressive-unfreeze`.
- Dataset quality-of-life:
  - `--auto-discover-mask-keys` to scan mask_* keys.
  - `--merge-icg-to-base` to merge `*_icg` into base classes.


## Performance and VRAM efficiency

- Fast mode: `--fast-mode` enables TF32, cudnn.benchmark, and high matmul precision for speed.
- Channels-last: `--channels-last` for improved CNN throughput.
- Fused AdamW (fallback-safe) for faster optimizer steps.
- Gradient checkpointing for local/global/spectral streams: `--grad-checkpoint` (default on), plus gradient accumulation via `--accum-steps`.
- EMA weights: `--ema-decay 0.999` (apply during validation, restore after).
- SDPA kernel preference: prefer FlashAttention (Ampere+) else memory-efficient; no-op fallback.
- CUDA allocator tuning: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `max_split_size_mb` preset.
- Spectral stream and HCMFF adaptive compute reduction on OOM retries (halve chunk size/tokens).


## Multi-GPU: DataParallel and DDP (torchrun)

- Default behavior: If `--ddp` is not set, training uses single GPU or falls back to `nn.DataParallel` when multiple GPUs are visible (with safe loss scaling and microbatch replication when `--force-all-gpus`).
- DDP mode: `--ddp` activates true DistributedDataParallel when launched with torchrun.
  - Initialization via `dist.init_process_group`, device selection by `LOCAL_RANK`.
  - `DistributedSampler` for train/val; calls `sampler.set_epoch(epoch)`.
  - Rank-0 only printing and checkpointing to prevent duplicate outputs.
  - Validation loss and confusion matrix are reduced across ranks.
  - Early stopping computed on rank 0 and broadcast to all ranks.
  - AUC sampling only on rank 0 to avoid cross-process coordination overhead.


## Resume and graceful interrupts

- Resume: `--resume auto` (prefers `last.pt` then `best.pt`) or `--resume /path/to/ckpt.pt`.
- Interrupt handling: Ctrl-C or SIGTERM saves `interrupt.pt` (rank-0 only under DDP) with model+optimizer+scheduler+scaler states.


## Evaluation improvements and outputs

- Safer metric computation: guards for background-only predictions and degenerate AUC/HD cases.
- Optional HD/HD95 per-class per-image: `--compute-hd` (can be slow).
- AUC sampling cap: `--auc-max-pixels` to bound memory/time.
- Outputs saved under `evaluation_results/` (or `--save-dir`):
  - `metrics_test.csv` (summary metrics)
  - `confusion_matrix.csv` and `confusion_matrix.png` (seaborn if available; matplotlib fallback)
  - `visualizations/prediction_sample_*.png` (first few samples)


## How to run

### Single-GPU

```bash
python src/training/train.py \
  --data-dir /path/to/Placenta \
  --run-name exp_placenta_sg \
  --epochs 100 --batch-size 4 \
  --fast-mode --channels-last --grad-checkpoint --accum-steps 2 \
  --ema-decay 0.999 \
  --both-fusions --hcmff-tokens 128 \
  --workflow-logs --workflow-interval 10 --very-verbose --progress-interval 10
```

### Multi-GPU (DataParallel)

```bash
python src/training/train.py \
  --data-dir /path/to/Placenta \
  --run-name exp_placenta_dp \
  --epochs 100 --batch-size 4 \
  --fast-mode --channels-last --grad-checkpoint --accum-steps 2 \
  --ema-decay 0.999
```

### Multi-GPU (DistributedDataParallel)

Launch with torchrun and pass `--ddp`:

```bash
torchrun --nproc_per_node=4 src/training/train.py \
  --ddp \
  --data-dir /path/to/Placenta \
  --run-name exp_placenta_ddp \
  --epochs 100 --batch-size 4 \
  --fast-mode --channels-last --grad-checkpoint --accum-steps 2 \
  --ema-decay 0.999 \
  --both-fusions --hcmff-tokens 128
```

### Resume training

```bash
python src/training/train.py --data-dir /path/to/Placenta --run-name exp_resume --resume auto
```

### Evaluate a checkpoint

```bash
python tests/evaluate.py \
  --model-path <path-to>/best.pt \
  --data-dir /path/to/Placenta \
  --save-dir saved/evaluation_results \
  --channels-last --fast-mode \
  --compute-hd --auc-max-pixels 200000
```


## Troubleshooting and tips

- Out-of-memory (OOM):
  - Reduce `--crop-size`, increase `--accum-steps`, or enable `--grad-checkpoint` (on by default).
  - Spectral stream and HCMFF auto-reduce chunk/tokens on OOM during forward.
  - Ensure `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set automatically in training).
- Numerical issues (NaNs/Infs):
  - Logits are sanitized before loss; if it persists, try lowering `--lr` or raising `--smooth`.
  - Prefer Flash SDPA when available (enabled automatically); if needed, set `--fast-mode`.
- Underfitting/background-only predictions:
  - Keep the improved defaults; try longer training (`--epochs`), higher `--gamma` (focal), or validate class keys.
  - Use `--both-fusions` and ensure `--hcmff-tokens` isn’t too small.
- DDP specifics:
  - Only rank 0 writes logs/weights.
  - AUC sampling happens on rank 0 only; macro AUC may slightly vary vs single-process.
- Evaluation HD distances:
  - In training-time validation under DDP, HD summary is approximate unless we gather. For precise test-time HD, run `tests/evaluate.py` single-GPU.


## Change log (high level)

- Training (`src/training/train.py`)
  - Added: fast-mode toggles, channels_last, fused AdamW, grad accumulation, EMA, allocator settings, Flash SDPA context.
  - Improved: dynamic class weighting per batch; longer warmup; robust resume; signal-based interrupt checkpoint.
  - Multi-GPU: DataParallel fallback tuned; full DDP with samplers, rank-0 I/O, reduced val loss & confusion matrix, early-stop broadcast.
  - Logging: per-batch progress + per-block workflow logs; organized `run_name/{logs,weights}`.
  - Metrics: safer AUC/HD guards; epoch-level `metrics_val.csv` with Params/FLOPs/Time/GPU Mem.
- Model (`src/models/full_model.py`, streams)
  - TCME→HCMFF optional pipeline; warm_up pathway; gradient checkpointing controls wired into local/global/spectral streams.
- Evaluation (`tests/evaluate.py`)
  - Fast-mode and channels_last options; DP-capable when batch-size permits.
  - Safer macro metrics; confusion matrix CSV+PNG; prediction diagnostics; optional HD/HD95.


## Next steps (optional)

- Add per-epoch confusion matrix export during training (rank-0) for quick monitoring.
- Implement exact HD aggregation under DDP by gathering per-image metrics (higher comms cost).
- Lightweight wandb/MLflow integration guarded to rank-0.

---
If you need me to tailor a ready-to-run command set for your exact server paths and GPU counts, tell me the dataset location and desired epochs/batch size and I’ll add a short launcher snippet.

