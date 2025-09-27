#!/bin/bash
#SBATCH -A hai
#SBATCH --qos=medium
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --job-name=hsi_training
#SBATCH --output=/tmp/chinmay.majithia_hsi/slurm_%j.out
#SBATCH --error=/tmp/chinmay.majithia_hsi/slurm_%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Load modules if needed (uncomment if required)
# module load cuda/11.8
# module load python/3.9

# Create output directory
mkdir -p /tmp/chinmay.majithia_hsi

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hsi

# Set optimal CUDA settings
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,roundup_power2_divisions:16

# Set environment variables for stability
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

# Print environment info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Run training
echo "Starting HSI training..."
python -m src.training.train \
  --data-dir /ssd_scratch/placenta/Placenta \
  --save-dir /tmp/chinmay.majithia_hsi \
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
  --run-name "ada-batch-run-$(date +%Y%m%d-%H%M%S)"

echo "Training completed at: $(date)"
echo "Final job status: $?"

# Copy results back to home directory for permanent storage
echo "Copying results to home directory..."
mkdir -p ~/hsi_training_results
cp -r /tmp/chinmay.majithia_hsi/* ~/hsi_training_results/ 2>/dev/null || true
echo "Results copied to ~/hsi_training_results/"