#!/bin/bash
#SBATCH --job-name=mi_locality
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/home2/amohammadi/out_mi_%j.txt
#SBATCH --error=/gpfs/home2/amohammadi/err_mi_%j.txt

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source /gpfs/home2/amohammadi/projects/gpt_noise_elimination/.venv/bin/activate

python3 /gpfs/home2/amohammadi/mi_locality.py
