#!/bin/bash
# Poisson baselines: PINO and PI-DeepONet against the existing supervised / bare / preconditioned
# arms, all at the published settings (64^2, K=4, n_train=1024, 500 epochs, batch 32).
#
# One SLURM array task per (arch, loss). All tasks share one output dir -- the run prefix carries
# the architecture and the loss, so nothing collides.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/baselines/poisson/sweep.sbatch          # the 7-arm table
#     sbatch experiments/baselines/poisson/lr_sweep.sbatch       # run this FIRST (see README)
# Watch:   squeue --me      Logs: logs/base_poisson_<jobid>_<task>.out
# Plot:    python experiments/baselines/poisson/plot_baselines.py \
#              --results_dir output/baselines/poisson/results
#
#SBATCH --job-name=pils-base-poisson
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/base_poisson_%A_%a.out
#SBATCH --array=0-20

source ~/env_tensorpils.sh
set -eo pipefail

# --- sweep configuration -----------------------------------------------------
# 7 arms x 3 seeds. Within a seed block: the four FNO arms, then the three DeepONet arms.
ARMS_MODEL=(fno      fno      fno   fno  deeponet deeponet   deeponet)
ARMS_LOSS=( data     galerkin pino  pls  data     pideeponet pls)
# Per-arm learning rate. The four existing arms keep their published 1e-3; the two baselines
# and the DeepONet arms get whatever lr_sweep.sbatch selected -- EDIT THESE before the run.
ARMS_LR=(   1e-3     1e-3     1e-3  1e-3 1e-3     1e-3       1e-3)
SEEDS=(42 43 44)

N_ARMS=${#ARMS_LOSS[@]}
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_ARMS ))
ARM_IDX=$((  SLURM_ARRAY_TASK_ID % N_ARMS ))

MODEL=${ARMS_MODEL[$ARM_IDX]}
LOSS=${ARMS_LOSS[$ARM_IDX]}
LR=${ARMS_LR[$ARM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUT=~/TensorPILS/output/baselines/poisson/seed${SEED}
# -----------------------------------------------------------------------------

echo "=== poisson baselines: model=${MODEL} loss=${LOSS} lr=${LR} seed=${SEED}"
echo "    (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --pde poisson \
    --model "${MODEL}" \
    --loss "${LOSS}" \
    --grid_resolution 64 \
    --n_train 1024 --n_val 128 --n_test 256 \
    -k 4 \
    --epochs 500 \
    --batch_size 32 \
    --seed "${SEED}" \
    --optimizer adam --lr "${LR}" --lr_min 1e-6 \
    --device cuda \
    --output_dir "${OUT}"
