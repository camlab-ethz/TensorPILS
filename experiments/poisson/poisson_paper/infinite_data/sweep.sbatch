#!/bin/bash
# Dataset-size scaling at a FIXED optimization budget: preconditioned least squares against
# supervised training, plus the streaming limit for PLS.
#
# The question: how much labelled data does supervised training need to match label-free
# preconditioned training? Every run gets the same 32,000 optimizer steps (1000 virtual epochs x
# 32 steps, batch 32), so the x-axis is dataset size and nothing else.
#
# 15 tasks:
#   0-13  seven dataset sizes x two arms (data-driven, PLS)
#   14    PLS on an infinite stream -- fresh samples every step, none ever seen twice
#
# There is deliberately NO streaming data-driven run. PLS's infinite-data limit is *reachable*:
# fresh source terms cost nothing because no labels are needed. Infinite labelled data would need
# a PDE solve per sample and does not exist in practice -- it is free here only because this
# dataset is manufactured (closed-form solutions). The missing curve is the point of the figure.
#
# Submit from ~/TensorPILS after `mkdir -p logs`. Throttle to be a good cluster citizen:
#     sbatch --array=0-14%4 experiments/poisson/poisson_paper/infinite_data/sweep.sbatch
# Watch:   squeue --me      Logs: logs/inf_data_<jobid>_<task>.out
# Then:    python experiments/poisson/poisson_paper/infinite_data/plot_scaling.py
#
#SBATCH --job-name=pils-inf-data
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/inf_data_%A_%a.out
#SBATCH --array=0-14

source ~/env_tensorpils.sh
set -eo pipefail

# --- configuration -----------------------------------------------------------
GRID=65                           # 2^6+1: nested dyadic, so the multigrid hierarchy is exact
K=10
OMEGA=0.8888888888888888          # 8/9, the optimal Jacobi damping for Q1 in 2D
EPOCHS=1000                       # virtual epochs; budget = EPOCHS * STEPS = 32,000 steps
STEPS=32                          # optimizer steps per virtual epoch, independent of n_train
SEED=42

SIZES=(64 128 256 512 1024 2048 4096)

# Two arms. Rates are the per-arm optima from the 500-epoch benchmark sweep; lr_min = lr/10 so
# both arms anneal on the same *shape* of cosine schedule and only the rate differs.
#                data      pls
ARMS_TAG=(       data      pls)
ARMS_LR=(        3e-4      3e-4)
ARMS_LR_MIN=(    3e-5      3e-5)
ARMS_FLAGS=(
  "--loss data"
  "--loss pls --precond_kind multigrid --mg_omega ${OMEGA}"
)

N_ARMS=${#ARMS_TAG[@]}
N_SIZES=${#SIZES[@]}
N_FINITE=$(( N_ARMS * N_SIZES ))          # 14

if [ "${SLURM_ARRAY_TASK_ID}" -lt "${N_FINITE}" ]; then
    ARM_IDX=$((  SLURM_ARRAY_TASK_ID % N_ARMS ))
    SIZE_IDX=$(( SLURM_ARRAY_TASK_ID / N_ARMS ))
    CFG="--n_train ${SIZES[$SIZE_IDX]} --fixed_eval"
    WHAT="n_train=${SIZES[$SIZE_IDX]}"
else
    ARM_IDX=1                             # streaming run: PLS only
    CFG="--stream"                        # --stream implies --fixed_eval
    WHAT="streaming (infinite)"
fi

TAG=${ARMS_TAG[$ARM_IDX]}
FLAGS=${ARMS_FLAGS[$ARM_IDX]}
LR=${ARMS_LR[$ARM_IDX]}
LR_MIN=${ARMS_LR_MIN[$ARM_IDX]}

# Run prefixes carry both the loss and the samples-<n> tag (samples-inf when streaming), so all
# 15 runs share one output directory without colliding.
OUT=~/TensorPILS/output/poisson/poisson_paper/infinite_data
# -----------------------------------------------------------------------------

echo "=== infinite_data: arm=${TAG} ${WHAT} lr=${LR} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --pde poisson \
    --model fno \
    ${FLAGS} \
    ${CFG} \
    -k ${K} \
    --grid_resolution ${GRID} \
    --n_val 128 --n_test 256 \
    --steps_per_epoch ${STEPS} \
    --epochs ${EPOCHS} \
    --batch_size 32 \
    --optimizer adam \
    --lr "${LR}" --lr_min "${LR_MIN}" \
    --seed ${SEED} \
    --device cuda \
    --output_dir "${OUT}"
