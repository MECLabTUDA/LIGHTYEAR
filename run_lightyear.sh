#!/bin/bash
#SBATCH --job-name=P2P_NTKL
#SBATCH --output=logs/lightyear_ntk.out
#SBATCH --error=logs/lightyear_ntk.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate p2p

python3 main.py --workspace NTKLightyear/sfa_1 --dataset camelyon17 --gpus 1 --clients 5 --malf sfa --malclients 1 --algo NTKLightyear
