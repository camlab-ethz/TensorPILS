#!/bin/bash
# realistic_ac_K16 rollout analysis. Identical to ../realistic_ac/analyze.sbatch but pointed at the
# K16 output tree -- it REUSES the realistic_ac Python scripts unchanged (only the paths differ).
# compute_rollout reads K straight from each run's JSON, so it rebuilds the CC reference at K=16
# automatically.
#
# Run AFTER sweep.sbatch has produced output/allen_cahn/realistic_ac_K16/{results,checkpoints}.
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/allen_cahn/realistic_ac_K16/analyze.sbatch
# Logs: logs/realistic_ac_K16_analyze_<jobid>.out
#
#SBATCH --job-name=pils-realistic-ac-K16-an
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/realistic_ac_K16_analyze_%j.out

source ~/env_tensorpils.sh
set -eo pipefail

cd ~/TensorPILS
OUT=output/allen_cahn/realistic_ac_K16

# Scalar metrics (reuses the realistic_ac compute script; K is read from the run JSONs).
python experiments/allen_cahn/realistic_ac/compute_rollout.py \
    --results_dir "${OUT}/results" \
    --checkpoints_dir "${OUT}/checkpoints" \
    --out "${OUT}/metrics.json" \
    --steps 100 \
    --n_samples 128 \
    --splits test train \
    --grid_resolution 128 \
    --device cuda

# Scalar figures (headless Agg) so they come down with metrics.json.
python experiments/allen_cahn/realistic_ac/plot_rollout.py \
    --metrics "${OUT}/metrics.json" \
    --out_dir "${OUT}"

# Field heatmaps (needs the checkpoints -> stays on Euler; download the PNGs).
python experiments/allen_cahn/realistic_ac/heatmaps.py \
    --results_dir "${OUT}/results" \
    --checkpoints_dir "${OUT}/checkpoints" \
    --steps 100 --split test --samples 0 1 2 \
    --grid_resolution 128 \
    --out_dir "${OUT}/heatmaps"
