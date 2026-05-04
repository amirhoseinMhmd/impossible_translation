#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=20:00:00

#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1

source .venv/bin/activate
python evaluation/evaluate.py   -i experiments/full-samples  -o evaluation/checkpoint-results   -t partialReverse   --batch_size 128