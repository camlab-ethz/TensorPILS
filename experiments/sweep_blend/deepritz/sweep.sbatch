#!/bin/bash
# Sweep the convex-blend preconditioner strength t for the PRECONDITIONED DEEP RITZ loss
# on Euler. The surrogate descent direction is P·r, interpolating from the plain energy
# gradient (t=0, P=I -> r = Au-b) to the supervised/Newton direction (t=1, P=A^-1 -> u-u*).
# One SLURM array task per t value; files keyed by the precond-blend-<t> tag.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/sweep_blend/deepritz/sweep.sbatch
# Watch:   squeue --me      Logs: logs/blend_deepritz_<jobid>_<task>.out
#
#SBATCH --job-name=pils-blend-deepritz
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/blend_deepritz_%A_%a.out
#SBATCH --array=0-9

# Source the environment first (module/Lmod is not always `set -u`-safe), then
# enable strict mode for the rest of the script.
source ~/env_tensorpils.sh
set -eo pipefail

# --- sweep configuration -----------------------------------------------------
STRENGTHS=(0.0 0.01 0.05 0.1 0.25 0.5 0.75 0.9 0.99 1.0)
T=${STRENGTHS[$SLURM_ARRAY_TASK_ID]}

OUT=~/TensorPILS/output/sweep_blend/deepritz
# -----------------------------------------------------------------------------

echo "=== deepritz blend sweep: strength t=${T} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --loss deepritz --precondition \
    --precond_kind blend \
    --precond_strength "${T}" \
    --optimizer adam \
    --n_train 1024 --n_val 128 --n_test 256 \
    -k 4 \
    --epochs 500 \
    --batch_size 32 \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
