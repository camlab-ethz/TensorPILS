#!/bin/bash
# OOD generalization for data-driven losses (in-band regime): train supervised on K=4 with
# two metrics and track relative-L2 each epoch on in-distribution K=4 val plus
# out-of-distribution K=6.
#   data     - supervised MSE
#   data_h1  - supervised H¹₀ norm (½ eᵀ A e, boundary-projected)
# One SLURM array task per loss; files keyed by the loss tag, shared output dir.
# n_modes=16 fully represents K=6, so this is OOD *within* the FNO's representable band
# (unlike ../data_k16, which deliberately exceeds it).
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/generalization/data_k4/sweep.sbatch
# Watch:   squeue --me      Logs: logs/gen_data_k4_<jobid>_<task>.out
#
#SBATCH --job-name=pils-gen-data-k4
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/gen_data_k4_%A_%a.out
#SBATCH --array=0-1

# Source the environment first (module/Lmod is not always `set -u`-safe), then
# enable strict mode for the rest of the script.
source ~/env_tensorpils.sh
set -eo pipefail

# --- sweep configuration -----------------------------------------------------
LOSSES=(data data_h1)
LOSS=${LOSSES[$SLURM_ARRAY_TASK_ID]}

OUT=~/TensorPILS/output/generalization/data_k4
# -----------------------------------------------------------------------------

echo "=== generalization (data, K=4): loss=${LOSS} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --loss "${LOSS}" \
    --n_train 1024 --n_val 128 --n_test 256 \
    -k 4 \
    --ood_k 6 \
    --ood_n_val 128 \
    --epochs 1000 \
    --batch_size 32 \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
