#!/bin/bash
# Allen-Cahn baselines: PINO and PI-DeepONet against supervised / bare / preconditioned FEM
# training, all as the same autoregressive one-step stepper (see README).
#
# One SLURM array task per (arch, loss, seed). All tasks of a seed share one output dir -- the
# run prefix carries the architecture and the loss, so nothing collides.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/baselines/allen_cahn/sweep.sbatch
# Watch:   squeue --me      Logs: logs/base_ac_<jobid>_<task>.out
# Plot:    python experiments/baselines/allen_cahn/plot_ac_baselines.py \
#              --results_dir output/baselines/allen_cahn
#
#SBATCH --job-name=pils-base-ac
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/base_ac_%A_%a.out
#SBATCH --array=0-17

source ~/env_tensorpils.sh
set -eo pipefail

# --- sweep configuration -----------------------------------------------------
# 6 arms x 3 seeds. PRECOND is the extra flag that turns bare LS into preconditioned LS.
ARMS_MODEL=(fno  fno      fno  fno      deeponet deeponet)
ARMS_LOSS=( data galerkin pino galerkin data     pideeponet)
ARMS_PREC=( none none     none multigrid none    none)
# Per-arm learning rate; the two baselines want their own (see the Poisson lr sweep for why).
ARMS_LR=(   1e-3 1e-3     1e-3 1e-3      1e-3     1e-3)
SEEDS=(42 43 44)

N_ARMS=${#ARMS_LOSS[@]}
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_ARMS ))
ARM_IDX=$((  SLURM_ARRAY_TASK_ID % N_ARMS ))

MODEL=${ARMS_MODEL[$ARM_IDX]}
LOSS=${ARMS_LOSS[$ARM_IDX]}
PREC=${ARMS_PREC[$ARM_IDX]}
LR=${ARMS_LR[$ARM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUT=~/TensorPILS/output/baselines/allen_cahn/seed${SEED}
# -----------------------------------------------------------------------------

echo "=== ac baselines: model=${MODEL} loss=${LOSS} precond=${PREC} lr=${LR} seed=${SEED}"
echo "    (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --pde ac \
    --model "${MODEL}" \
    --loss "${LOSS}" \
    --ac_precond "${PREC}" \
    --grid_resolution 64 \
    --n_train 512 --n_val 64 --n_test 128 \
    -k 4 \
    --ac_a 1 --ac_eps 32 --dt 0.01 --n_steps 20 --rollout_steps 4 \
    --bptt_mode pushforward \
    --epochs 300 \
    --batch_size 32 \
    --seed "${SEED}" \
    --optimizer adam --lr "${LR}" --lr_min 1e-6 \
    --device cuda \
    --output_dir "${OUT}"
