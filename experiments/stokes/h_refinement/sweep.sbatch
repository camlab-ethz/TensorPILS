#!/bin/bash
# Stokes h-refinement: the three losses at three velocity grids, everything else fixed.
# One SLURM array task per (grid, loss) pair; all tasks share one output dir (the run
# prefix carries the grid and the loss, so files do not collide).
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/stokes/h_refinement/sweep.sbatch
# Watch:   squeue --me      Logs: logs/stokes_href_<jobid>_<task>.out
# Plot:    python experiments/stokes/h_refinement/plot_h_refinement.py \
#              --results_dir output/stokes/h_refinement/results
#
#SBATCH --job-name=pils-stokes-href
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/stokes_href_%A_%a.out
#SBATCH --array=0-8

source ~/env_tensorpils.sh
set -eo pipefail

# --- sweep configuration -----------------------------------------------------
# Tasks 0-2: 33^2, tasks 3-5: 65^2, tasks 6-8: 129^2. Within each grid: pls, galerkin, data.
GRIDS=(33 33 33 65 65 65 129 129 129)
LOSSES=(pls galerkin data pls galerkin data pls galerkin data)

GR=${GRIDS[$SLURM_ARRAY_TASK_ID]}
LOSS=${LOSSES[$SLURM_ARRAY_TASK_ID]}

# 129^2 has ~37.5k DOFs per sample; shrink the batched reference solve to stay in memory.
CHUNK=64
[ "${GR}" -ge 129 ] && CHUNK=32

OUT=~/TensorPILS/output/stokes/h_refinement
# -----------------------------------------------------------------------------

echo "=== stokes h-refinement: grid ${GR}^2, loss ${LOSS} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --pde stokes \
    --loss "${LOSS}" \
    --grid_resolution "${GR}" \
    --stokes_ref_chunk "${CHUNK}" \
    --n_train 512 --n_val 64 --n_test 128 \
    -k 4 \
    --epochs 300 \
    --batch_size 32 \
    --seed 42 \
    --optimizer adam --lr 1e-3 --lr_min 1e-6 \
    --device cuda \
    --output_dir "${OUT}"
