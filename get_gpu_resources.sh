#!/bin/bash

# Simple script to get interactive GPU allocation
# Run this with: bash get_gpu_resources.sh

echo "Requesting interactive GPU session..."
echo "This will give you a shell on a GPU node for 20 hours"
echo "Press Ctrl+C to cancel if needed"
echo ""

srun --partition=gpu --gres=gpu:4 --cpus-per-task=40 --time=20:00:00 --pty bash