#!/bin/bash
#SBATCH --job-name=nk_leiden
#SBATCH --output=nk_leiden_%j.out
#SBATCH --error=nk_leiden_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00

cd /path/to/NK_project
python scripts/run_leiden_discovery.py
