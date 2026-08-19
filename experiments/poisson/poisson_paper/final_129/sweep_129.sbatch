#!/bin/bash
# Poisson paper table at 129x129 -- the resolution check on the 65^2 finals.
#
# ONE seed (42), seven arms, seven tasks. The 65^2 finals carry the error bars; this run answers
# a different question -- does the ordering of the arms survive a 4x finer grid -- and for that a
# single seed per arm is enough, provided the table reports it as such.
#
# 129 = 2^7 + 1, so the grid is nested-dyadic like 65 = 2^6 + 1 and the multigrid hierarchy is
# exact rather than interpolated.
#
# WHAT CHANGES vs sweep.sbatch, and why:
#   --grid_resolution 129   the point of the run
#   --mg_levels 5           129 -> 65 -> 33 -> 17 -> 9. The 65^2 runs used the default 4 levels and
#                           bottomed out at 9x9; adding a level keeps the SAME coarsest grid, so
#                           the preconditioner is the genuine analogue rather than a shallower one.
#   --time 12:00:00         see README -- set from the 65^2 elapsed times, with headroom
#   --gres=gpumem:20g       activation memory roughly doubles; the default allocation is whatever
#                           GPU the scheduler happens to hand out
#   SEEDS=(42)              one seed, array 0-6
#
# WHAT DELIBERATELY DOES NOT CHANGE: n_modes (16), hidden_dim (64), num_layers (5), batch size,
# epochs, K, and the per-arm learning rates. Holding the model fixed is what makes this a
# resolution study rather than a second, differently-tuned experiment. See the README for the one
# caveat that carries real risk (learning-rate transfer).
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/poisson/poisson_paper/final_129/sweep_129.sbatch
# Watch:   squeue --me      Logs: logs/pp129_<jobid>_<task>.out
# Then:    python experiments/poisson/poisson_paper/plot_final.py \
#              --root output/poisson/poisson_paper/final_129
#
#SBATCH --job-name=pils-pp-129
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=20G
#SBATCH --output=logs/pp129_%A_%a.out
#SBATCH --array=0-6

source ~/env_tensorpils.sh
set -eo pipefail

OMEGA=0.8888888888888888          # 8/9, the optimal Jacobi damping for Q1 in 2D
LEVELS=5                          # 129 -> 65 -> 33 -> 17 -> 9, same coarsest grid as the 65^2 runs

# --- the seven arms of the table (identical to sweep.sbatch) ------------------
ARMS_MODEL=(fno fno fno fno fno fno deeponet)
ARMS_TAG=(  data ls pls deepritz pdeepritz pino pideeponet)
ARMS_FLAGS=(
  "--loss data"
  "--loss galerkin"
  "--loss pls --precond_kind multigrid --mg_omega ${OMEGA} --mg_levels ${LEVELS}"
  "--loss deepritz --bc_mode hard"
  "--loss deepritz --bc_mode hard --precondition --precond_kind multigrid --mg_omega ${OMEGA} --mg_levels ${LEVELS}"
  "--loss pino"
  "--loss pideeponet"
)
# Carried over from the 65^2 stage-1 sweep. NOT re-tuned at 129 -- see README, "learning rates".
#                                     data  ls    pls   dr    pdr   pino  pideeponet
ARMS_LR=(                             1e-3  3e-3  1e-3  3e-3  1e-3  3e-3  3e-4)
SEEDS=(42)

N_ARMS=${#ARMS_TAG[@]}
ARM_IDX=$((  SLURM_ARRAY_TASK_ID % N_ARMS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_ARMS ))

MODEL=${ARMS_MODEL[$ARM_IDX]}
TAG=${ARMS_TAG[$ARM_IDX]}
FLAGS=${ARMS_FLAGS[$ARM_IDX]}
LR=${ARMS_LR[$ARM_IDX]}
SEED=${SEEDS[$SEED_IDX]}
LR_MIN=$(python3 -c "print(${LR}/10)")

OUT=~/TensorPILS/output/poisson/poisson_paper/final_129/seed${SEED}
# -----------------------------------------------------------------------------

echo "=== poisson_paper 129: arm=${TAG} lr=${LR} seed=${SEED}"
echo "    (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd ~/TensorPILS
python -m tensorpils.cli \
    --pde poisson \
    --model "${MODEL}" \
    ${FLAGS} \
    -k 10 \
    --grid_resolution 129 \
    --n_train 1024 --n_val 128 --n_test 256 \
    --epochs 500 \
    --batch_size 32 \
    --optimizer adam \
    --lr "${LR}" --lr_min "${LR_MIN}" \
    --seed "${SEED}" \
    --device cuda \
    --output_dir "${OUT}"
