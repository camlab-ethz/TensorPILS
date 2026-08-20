#!/bin/bash
# Poisson paper table, STAGE 1: learning-rate selection.
#
# Five arms x six learning rates = 30 tasks at a REDUCED 100-epoch budget, to save compute
# (a fifth of the full runs). The cosine schedule still anneals lr -> lr/10 over the run, so the
# schedule has the same shape here as in the 500-epoch finals and the sweep stays self-consistent.
#
# CAVEAT, worth one cheap check: the optimal lr depends on the budget. Tuning the bare
# least-squares arm at 1e4 steps picks ~3e-2, while at 1e5 steps a 30x smaller rate is 25x better
# -- short sweeps systematically prefer rates that are too large. After picking, re-run the winner
# and the runner-up at 500 epochs for one or two arms (2-4 runs) to confirm the ranking holds.
#
# Selection metric is the VALIDATION relative L2 (stats.val_rel_l2_errors). Never the test set.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/poisson/poisson_paper/poisson_benchmark/lr_sweep.sbatch
# Watch:   squeue --me      Logs: logs/pp_lr_<jobid>_<task>.out
# Then:    python experiments/poisson/poisson_paper/poisson_benchmark/plot_lr_sweep.py \
#              --root output/poisson/poisson_paper/poisson_benchmark/lr_sweep
#
#SBATCH --job-name=pils-pp-lr
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/pp_lr_%A_%a.out
#SBATCH --array=0-29

source ~/env_tensorpils.sh
set -eo pipefail

OMEGA=0.8888888888888888          # 8/9, the optimal Jacobi damping for Q1 in 2D

# --- the five arms of the table ---------------------------------------------
ARMS_MODEL=(fno fno fno fno deeponet)
ARMS_TAG=(  data ls pls pino pideeponet)
ARMS_FLAGS=(
  "--loss data"
  "--loss galerkin"
  "--loss pls --precond_kind multigrid --mg_omega ${OMEGA}"
  "--loss pino"
  "--loss pideeponet"
)
# Wide grid: the arms do not share a reduction (ours is a summed square, PINO a relative norm
# ratio), so their gradient scales differ by orders of magnitude.
LRS=(1e-4 3e-4 1e-3 3e-3 1e-2 3e-2)

# Epoch budget. 100 by default (the cheap first pass). Override per submission with
#     sbatch --export=ALL,EPOCHS=500 --array=... lr_sweep.sbatch
# to re-tune a subset of arms at the full budget -- the optimal lr depends on it, and short
# sweeps systematically prefer rates that are too large (see the CAVEAT at the top).
EPOCHS=${EPOCHS:-100}

N_ARMS=${#ARMS_TAG[@]}
ARM_IDX=$(( SLURM_ARRAY_TASK_ID % N_ARMS ))
LR_IDX=$((  SLURM_ARRAY_TASK_ID / N_ARMS ))

MODEL=${ARMS_MODEL[$ARM_IDX]}
TAG=${ARMS_TAG[$ARM_IDX]}
FLAGS=${ARMS_FLAGS[$ARM_IDX]}
LR=${LRS[$LR_IDX]}
# Keep the cosine schedule's shape fixed across the sweep: lr_min = lr/10, as in the defaults.
# A fixed lr_min would mean "no decay" at the small end and a 300x decay at the large end.
LR_MIN=$(python3 -c "print(${LR}/10)")

# Run prefixes do not encode the learning rate, so each lr needs its own directory.
OUT=~/TensorPILS/output/poisson/poisson_paper/poisson_benchmark/lr_sweep/lr${LR}
# -----------------------------------------------------------------------------

echo "=== poisson_paper lr sweep: arm=${TAG} lr=${LR} epochs=${EPOCHS} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --pde poisson \
    --model "${MODEL}" \
    ${FLAGS} \
    -k 10 \
    --grid_resolution 65 \
    --n_train 1024 --n_val 128 --n_test 256 \
    --epochs "${EPOCHS}" \
    --batch_size 32 \
    --optimizer adam \
    --lr "${LR}" --lr_min "${LR_MIN}" \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
