#!/bin/bash
# realistic_ac rollout analysis (stage 1: compute the scalar metrics on Euler, where the heavy
# .pth checkpoints already live), then reproduce the figures headlessly so they rsync down with
# the tiny metrics.json. The checkpoints never leave the cluster.
#
# Run AFTER sweep.sbatch has produced output/allen_cahn/realistic_ac/{results,checkpoints}.
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/allen_cahn/realistic_ac/analyze.sbatch
# Logs: logs/realistic_ac_analyze_<jobid>.out
#
#SBATCH --job-name=pils-realistic-ac-an
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/realistic_ac_analyze_%j.out

source ~/env_tensorpils.sh
set -eo pipefail

cd ~/TensorPILS
# The 100-step convex-concave reference at 256^2 is the expensive part; --n_samples caps each split.
python experiments/allen_cahn/realistic_ac/compute_rollout.py \
    --results_dir output/allen_cahn/realistic_ac/results \
    --checkpoints_dir output/allen_cahn/realistic_ac/checkpoints \
    --out output/allen_cahn/realistic_ac/metrics.json \
    --steps 100 \
    --n_samples 128 \
    --splits test train \
    --device cuda

# Scalar figures (headless Agg) so they come down with metrics.json.
python experiments/allen_cahn/realistic_ac/plot_rollout.py \
    --metrics output/allen_cahn/realistic_ac/metrics.json \
    --out_dir output/allen_cahn/realistic_ac

# Field heatmaps (needs the checkpoints -> stays on Euler; download the PNGs).
python experiments/allen_cahn/realistic_ac/heatmaps.py \
    --steps 100 --split test --samples 0 1 2 \
    --out_dir output/allen_cahn/realistic_ac/heatmaps
