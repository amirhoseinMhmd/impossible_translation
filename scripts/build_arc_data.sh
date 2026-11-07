#!/bin/bash
#SBATCH --job-name=arc_v2
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/gpfs/home2/amohammadi/out_arcv2_%j.txt
#SBATCH --error=/gpfs/home2/amohammadi/err_arcv2_%j.txt

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /gpfs/home2/amohammadi/projects/gpt_noise_elimination/.venv/bin/activate

cd /gpfs/home2/amohammadi/arc_v2
python3 build_arc_data.py --root /gpfs/home2/amohammadi/arc_v2/evaluation_results --out /gpfs/home2/amohammadi/arc_v2/out
