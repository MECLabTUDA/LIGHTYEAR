#!/bin/bash
#SBATCH --job-name=P2P_eval
#SBATCH --output=logs/eval.out
#SBATCH --error=logs/eval.err
#SBATCH --time=0-05:00:00
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate p2p

python3 eval.py --algo LIGHTYEAR --topology p2p --dataset camelyon17 --malfunction ana >> results/LIGHTYEAR_ana.txt