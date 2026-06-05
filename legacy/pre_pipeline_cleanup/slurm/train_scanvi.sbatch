#!/bin/bash
#SBATCH --job-name=nk_scanvi
#SBATCH --output=nk_scanvi_%j.out
#SBATCH --error=nk_scanvi_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00

cd /path/to/NK_project
python scripts/train_scanvi.py
