#!/bin/bash
# OOD generalization for data-driven losses: train supervised on K=16 with two metrics and
# track relative-L2 each epoch on in-distribution K=16 val plus out-of-distribution K=20.
#   data     - supervised MSE
#   data_h1  - supervised H¹₀ norm (½ eᵀ A e, boundary-projected)
# One SLURM array task per loss; files keyed by the loss tag, shared output dir.
#
# NOTE: n_modes is left at the default 16, so the FNO is band-limited below the K=20 target
# (and at the edge of K=16) — a deliberate extreme-OOD test; expect a truncation floor in
# the K=20 error (cf. ../pls_k16).
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/generalization/data_k16/sweep.sbatch
# Watch:   squeue --me      Logs: logs/gen_data_k16_<jobid>_<task>.out
#
#SBATCH --job-name=pils-gen-data-k16
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/gen_data_k16_%A_%a.out
#SBATCH --array=0-1

# Source the environment first (module/Lmod is not always `set -u`-safe), then
# enable strict mode for the rest of the script.
source ~/env_tensorpils.sh
set -eo pipefail

# --- sweep configuration -----------------------------------------------------
LOSSES=(data data_h1)
LOSS=${LOSSES[$SLURM_ARRAY_TASK_ID]}

OUT=~/TensorPILS/output/generalization/data_k16
# -----------------------------------------------------------------------------

echo "=== generalization (data, K=16): loss=${LOSS} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --loss "${LOSS}" \
    --n_train 1024 --n_val 128 --n_test 256 \
    -k 16 \
    --ood_k 20 \
    --ood_n_val 128 \
    --epochs 5000 \
    --batch_size 32 \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
