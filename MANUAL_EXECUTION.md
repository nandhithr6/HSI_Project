# HSI Training - ADA SLURM Batch Job

## Quick Start (Recommended - Batch Job):
```bash
cd ~/HSI_Project

# No need to create directory - the job will create it

# Submit the batch job (this runs automatically)
sbatch run_training_ada.sbatch
```

## Check job status:
```bash
# See your jobs  
squeue -u chinmay.majithia

# Monitor output (replace JOBID with actual job number)
tail -f /tmp/chinmay.majithia_hsi/slurm_JOBID.out

# Check for errors
tail -f /tmp/chinmay.majithia_hsi/slurm_JOBID.err
```

## Alternative: Manual Interactive Session (if you need to debug):

### Step 1: Get interactive allocation
```bash
srun --partition=gpu --gres=gpu:4 --cpus-per-task=40 --time=20:00:00 --pty bash
```

### Step 2: Once you get the shell on GPU node:
```bash
# Set optimal CUDA settings
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,roundup_power2_divisions:16

# Create results directory
mkdir -p /share6/chinmay.majithia/hsi_training

# Run training (copy-paste this entire command)
python -m src.training.train \
  --data-dir /ssd_scratch/placenta/Placenta \
  --save-dir /share6/chinmay.majithia/hsi_training \
  --epochs 200 \
  --batch-size 8 \
  --lr 1e-4 \
  --num-workers 12 \
  --use-hcmff \
  --hcmff-tokens 128 \
  --merge-icg-to-base \
  --crop-size 512 \
  --force-all-gpus \
  --fast-mode \
  --run-name "manual-1st-run-$(date +%Y%m%d-%H%M%S)"
```

## 3. Later, for evaluation (run from your home directory):
```bash
# Evaluate the best model from training
python tests/evaluate.py \
  --model-path /share6/chinmay.majithia/hsi_training/manual-1st-run-XXXXXX/weights/best.pt \
  --data-dir /ssd_scratch/placenta/Placenta \
  --save-dir /share6/chinmay.majithia/hsi_training/evaluation_results \
  --batch-size 2
```

## 4. Monitor during training:
- Watch GPU usage: `nvidia-smi -l 5`
- Check progress: `tail -f /share6/chinmay.majithia/hsi_training/manual-1st-run-*/logs/*/epochs.csv`
- If errors occur: Copy-paste the error messages to share with me!