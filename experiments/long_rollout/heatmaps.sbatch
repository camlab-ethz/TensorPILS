#!/bin/bash
# Long-rollout 2D field heatmaps (vs the convex-concave ground truth). Unlike the scalar
# compute_long_rollout.py this needs the actual fields, so it reads the checkpoints and must run on a
# compute node (GPU: model rollouts + the CC reference solve). Emits PNGs to output/long_rollout/heatmaps.
#
# It imports the shared IC / reference / rollout helpers from compute_long_rollout.py, so it is run as
# a module (python -m ...), from the repo root.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/long_rollout/heatmaps.sbatch
# Logs: logs/long_rollout_heatmaps_<jobid>.out
#
#SBATCH --job-name=pils-lr-heatmaps
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/long_rollout_heatmaps_%j.out

source ~/env_tensorpils.sh
set -eo pipefail

cd ~/TensorPILS
python -m experiments.long_rollout.heatmaps \
    --results_dir output/compare_bptt/results \
    --checkpoints_dir output/compare_bptt/checkpoints \
    --out_dir output/long_rollout/heatmaps \
    --steps 40 \
    --samples 0 1 2 \
    --device cuda
