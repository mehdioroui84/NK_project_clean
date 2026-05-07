#!/bin/bash
#SBATCH --job-name=nk_scvi
#SBATCH --output=nk_scvi_%j.out
#SBATCH --error=nk_scvi_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00

cd /path/to/NK_project
python scripts/01_train_scvi.py
