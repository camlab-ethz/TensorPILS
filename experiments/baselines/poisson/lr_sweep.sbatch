#!/bin/bash
# Learning-rate sweep for the NEW arms, at a reduced budget. Run this BEFORE sweep.sbatch and
# transcribe the winners into its ARMS_LR array.
#
# Why this exists: PINO's relative-Lp reduction and our sum-over-nodes losses differ in gradient
# scale by orders of magnitude, so a single shared lr would silently handicap one side. Reporting
# a baseline that was never tuned is the fastest way to lose a reviewer.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/baselines/poisson/lr_sweep.sbatch
# Read off the winners:
#     python experiments/baselines/poisson/plot_baselines.py \
#         --results_dir output/baselines/poisson/lr --lr_table
#
#SBATCH --job-name=pils-base-lr
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/base_poisson_lr_%A_%a.out
#SBATCH --array=0-19

source ~/env_tensorpils.sh
set -eo pipefail

# --- sweep configuration -----------------------------------------------------
# 4 new arms x 5 learning rates. (The FNO data/galerkin/pls arms keep their published lr.)
ARMS_MODEL=(fno  deeponet   deeponet deeponet)
ARMS_LOSS=( pino pideeponet data     pls)
LRS=(3e-4 1e-3 3e-3 1e-2 3e-2)

N_LR=${#LRS[@]}
ARM_IDX=$(( SLURM_ARRAY_TASK_ID / N_LR ))
LR_IDX=$((  SLURM_ARRAY_TASK_ID % N_LR ))

MODEL=${ARMS_MODEL[$ARM_IDX]}
LOSS=${ARMS_LOSS[$ARM_IDX]}
LR=${LRS[$LR_IDX]}

# Separate output dir per lr, because the run prefix does NOT carry the learning rate --
# without this the five lr runs of an arm would overwrite each other.
OUT=~/TensorPILS/output/baselines/poisson/lr/lr${LR}
# -----------------------------------------------------------------------------

echo "=== poisson lr sweep: model=${MODEL} loss=${LOSS} lr=${LR}"
echo "    (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --pde poisson \
    --model "${MODEL}" \
    --loss "${LOSS}" \
    --grid_resolution 64 \
    --n_train 1024 --n_val 128 --n_test 256 \
    -k 4 \
    --epochs 100 \
    --batch_size 32 \
    --seed 42 \
    --optimizer adam --lr "${LR}" --lr_min 1e-6 \
    --device cuda \
    --output_dir "${OUT}"
