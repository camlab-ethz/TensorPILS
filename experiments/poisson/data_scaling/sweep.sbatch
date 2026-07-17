#!/bin/bash
# Dataset-size scaling at fixed optimization budget: relative-L2 test error vs n_train,
# including the infinite-data (streaming) limit where no sample is ever seen twice.
# One SLURM array task per dataset size; the LAST task is the streaming run. All tasks
# share one output dir (files are keyed by the samples-<n> tag / samples-inf, no collisions)
# and, via --fixed_eval, the SAME val/test sets. Edit NVALS / EPOCHS / STEPS / PATIENCE below.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/poisson/data_scaling/sweep.sbatch
# Watch:   squeue --me      Logs: logs/data_scaling_<jobid>_<task>.out
#
#SBATCH --job-name=pils-data-scaling
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/data_scaling_%A_%a.out
#SBATCH --array=0-8

# Source the environment first (module/Lmod is not always `set -u`-safe), then strict mode.
source ~/env_tensorpils.sh
set -eo pipefail

# --- sweep configuration (edit freely) ---------------------------------------
GRID=64
EPOCHS=500        # virtual epochs; total budget = EPOCHS * STEPS optimizer steps
STEPS=32          # optimizer steps per virtual epoch (32 steps x batch 32 = 1024 samples/epoch,
                  # so the budget equals the existing 500-epoch runs on n_train=1024)
PATIENCE=50       # early-stop after this many epochs without val improvement
NVALS=(32 64 128 256 512 1024 2048 4096)   # dataset sizes; keep --array as 0-${#NVALS[@]}
# Tasks 0..N-1 sweep n_train; task N (= ${#NVALS[@]}) trains on the infinite stream.
NN=${#NVALS[@]}
if [ "${SLURM_ARRAY_TASK_ID}" -lt "${NN}" ]; then
    CFG="--n_train ${NVALS[${SLURM_ARRAY_TASK_ID}]} --fixed_eval"
else
    CFG="--stream"
fi

OUT=~/TensorPILS/output/poisson/data_scaling
# -----------------------------------------------------------------------------

echo "=== data scaling: ${CFG} (task ${SLURM_ARRAY_TASK_ID}, grid ${GRID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --loss pls \
    --precond_kind multigrid \
    ${CFG} \
    --steps_per_epoch ${STEPS} \
    --patience ${PATIENCE} \
    --grid_resolution ${GRID} \
    --optimizer adam \
    --n_val 128 --n_test 256 \
    -k 4 \
    --epochs ${EPOCHS} \
    --batch_size 32 \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
