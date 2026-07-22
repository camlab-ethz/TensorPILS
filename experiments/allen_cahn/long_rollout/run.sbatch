#!/bin/bash
# Long-rollout analysis of the compare_bptt checkpoints (stage 1: compute the metrics on Euler,
# where the heavy .pth files already live). Writes a tiny output/allen_cahn/long_rollout/metrics.json; plot
# from it anywhere with plot_long_rollout.py -- the checkpoints never leave the cluster.
#
# It is checkpoint-agnostic: it globs whatever *_best.pth are in output/allen_cahn/compare_bptt/checkpoints,
# so it runs on the current run and again (unchanged) after a longer/converged sweep overwrites them.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/allen_cahn/long_rollout/run.sbatch
# Logs: logs/long_rollout_<jobid>.out
#
#SBATCH --job-name=pils-long-rollout
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/long_rollout_%j.out

source ~/env_tensorpils.sh
set -eo pipefail

cd ~/TensorPILS
python experiments/allen_cahn/long_rollout/compute_long_rollout.py \
    --results_dir output/allen_cahn/compare_bptt/results \
    --checkpoints_dir output/allen_cahn/compare_bptt/checkpoints \
    --out output/allen_cahn/long_rollout/metrics.json \
    --steps 40 \
    --n_samples 256 \
    --device cuda

# Reproduce the energy/error figures right here (headless Agg) so they rsync down with the metrics.
python experiments/allen_cahn/long_rollout/plot_long_rollout.py \
    --metrics output/allen_cahn/long_rollout/metrics.json \
    --out_dir output/allen_cahn/long_rollout
