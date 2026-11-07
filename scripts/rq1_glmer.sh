#!/bin/bash
#SBATCH --job-name=rq1_glmer
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/gpfs/home2/amohammadi/out_glmer_%j.txt
#SBATCH --error=/gpfs/home2/amohammadi/err_glmer_%j.txt

module load 2023
module load R-bundle-CRAN/2023.12-foss-2023a

cd /gpfs/home2/amohammadi/arc_v2
Rscript rq1_glmer.R /gpfs/home2/amohammadi/arc_v2/out
