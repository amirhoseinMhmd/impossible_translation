#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=50:00:00

#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1

set -euo pipefail

FUNCTION="$1"
SEED="$2"
OUTPUT_DIR="${3:-/home/amohammadi/models/${FUNCTION}-gutenberg-${SEED}}"

source .venv/bin/activate
python fine_tuning/finetune.py \
  -c configs/fine-tuning-gpu-config.yml \
  -p datasets/100k_gutenberg.train \
  -t "$FUNCTION" \
  -o "$OUTPUT_DIR" \
  --seed "$SEED" \
  > "logs/${FUNCTION}_${SEED}.log" 2>&1
