#!/bin/bash
# Poisson paper table, STAGE 2: the final runs.
#
# Seven arms x three seeds = 21 tasks. EDIT ARMS_LR BELOW before submitting -- fill in whatever
# lr_sweep.sbatch selected per arm (lowest best validation relative L2). Everything else is the
# repo default, identical across arms, so the comparison differs only in the objective.
#
# Submit from ~/TensorPILS after `mkdir -p logs`, and AFTER the lr sweep:
#     sbatch experiments/poisson/poisson_paper/sweep.sbatch
# Watch:   squeue --me      Logs: logs/pp_final_<jobid>_<task>.out
# Then:    python experiments/poisson/poisson_paper/make_table.py \
#              --root output/poisson/poisson_paper/final
#
#SBATCH --job-name=pils-pp-final
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/pp_final_%A_%a.out
#SBATCH --array=0-20

source ~/env_tensorpils.sh
set -eo pipefail

OMEGA=0.8888888888888888          # 8/9, the optimal Jacobi damping for Q1 in 2D

ARMS_MODEL=(fno fno fno fno fno fno deeponet)
ARMS_TAG=(  data ls pls deepritz pdeepritz pino pideeponet)
ARMS_FLAGS=(
  "--loss data"
  "--loss galerkin"
  "--loss pls --precond_kind multigrid --mg_omega ${OMEGA}"
  "--loss deepritz --bc_mode hard"
  "--loss deepritz --bc_mode hard --precondition --precond_kind multigrid --mg_omega ${OMEGA}"
  "--loss pino"
  "--loss pideeponet"
)
# from stage 1 (lr sweep, 100 epochs)  data  ls    pls   dr    pdr   pino  pideeponet
ARMS_LR=(                              1e-3  3e-3  1e-3  3e-3  1e-3  1e-2  3e-4)
SEEDS=(42 43 44)

N_ARMS=${#ARMS_TAG[@]}
ARM_IDX=$((  SLURM_ARRAY_TASK_ID % N_ARMS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_ARMS ))

MODEL=${ARMS_MODEL[$ARM_IDX]}
TAG=${ARMS_TAG[$ARM_IDX]}
FLAGS=${ARMS_FLAGS[$ARM_IDX]}
LR=${ARMS_LR[$ARM_IDX]}
SEED=${SEEDS[$SEED_IDX]}
LR_MIN=$(python3 -c "print(${LR}/10)")

# Run prefixes do not encode the seed, so each seed needs its own directory.
OUT=~/TensorPILS/output/poisson/poisson_paper/final/seed${SEED}
# -----------------------------------------------------------------------------

echo "=== poisson_paper final: arm=${TAG} lr=${LR} seed=${SEED} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --pde poisson \
    --model "${MODEL}" \
    ${FLAGS} \
    -k 10 \
    --grid_resolution 65 \
    --n_train 1024 --n_val 128 --n_test 256 \
    --epochs 500 \
    --batch_size 32 \
    --optimizer adam \
    --lr "${LR}" --lr_min "${LR_MIN}" \
    --seed "${SEED}" \
    --device cuda \
    --output_dir "${OUT}"
