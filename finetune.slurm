#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=50:00:00

#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1

FUNCTION ="$1"
SEED="$2"

source .venv/bin/activate
python fine_tuning/finetune.py  -c configs/gpu-100k.yml -p datasets/100k_gutenberg.train -t "$FUNCTION"  > "${FUNCTION}_${SEED}.log" 2>&1