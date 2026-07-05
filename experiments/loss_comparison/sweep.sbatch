#!/bin/bash
# Loss comparison: train the FNO with three supervised/physics losses at default settings
# for 1000 epochs, and compare validation relative-L2 over epochs:
#   data_l2  - supervised true-L2 (mass-weighted, ½ eᵀ M e)  [matches the eval metric]
#   data     - supervised MSE (flat grid average)
#   deepritz - Deep Ritz energy (label-free, no preconditioner)
# One SLURM array task per loss; files keyed by the loss tag, shared output dir.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/loss_comparison/sweep.sbatch
# Watch:   squeue --me      Logs: logs/losscmp_<jobid>_<task>.out
#
#SBATCH --job-name=pils-loss-cmp
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/losscmp_%A_%a.out
#SBATCH --array=0-2

# Source the environment first (module/Lmod is not always `set -u`-safe), then
# enable strict mode for the rest of the script.
source ~/env_tensorpils.sh
set -eo pipefail

# --- experiment configuration ------------------------------------------------
LOSSES=(data_l2 data deepritz)
LOSS=${LOSSES[$SLURM_ARRAY_TASK_ID]}

OUT=~/TensorPILS/output/loss_comparison
# -----------------------------------------------------------------------------

echo "=== loss comparison: loss=${LOSS} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --loss "${LOSS}" \
    --n_train 1024 --n_val 128 --n_test 256 \
    -k 4 \
    --epochs 1000 \
    --batch_size 32 \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
