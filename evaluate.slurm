#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=20:00:00

#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1

set -euo pipefail

INPUT_DIR="$1"
PERTURBATION="$2"
OUTPUT_DIR="${INPUT_DIR}/evaluation"
SEED=$3

mkdir -p logs
source .venv/bin/activate

python evaluation/evaluate.py \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR" \
  -t "$PERTURBATION" \
  --batch_size 128 \
  > "logs/evaluate_${PERTURBATION}_${SEED}.log" 2>&1
