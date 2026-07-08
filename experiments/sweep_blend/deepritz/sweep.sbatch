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
#SBATCH --array=0-10

# Source the environment first (module/Lmod is not always `set -u`-safe), then
# enable strict mode for the rest of the script.
source ~/env_tensorpils.sh
set -eo pipefail

# --- sweep configuration -----------------------------------------------------
# Tasks 0-9 sweep the convex-blend strength t; task 10 is the multigrid (GMG)
# reference in its default setup -- the preconditioner that works in practice.
# To add only the GMG line to an existing sweep: sbatch --array=10 <this file>.
PRECOND=(
    "--precond_kind blend --precond_strength 0.0"
    "--precond_kind blend --precond_strength 0.01"
    "--precond_kind blend --precond_strength 0.05"
    "--precond_kind blend --precond_strength 0.1"
    "--precond_kind blend --precond_strength 0.25"
    "--precond_kind blend --precond_strength 0.5"
    "--precond_kind blend --precond_strength 0.75"
    "--precond_kind blend --precond_strength 0.9"
    "--precond_kind blend --precond_strength 0.99"
    "--precond_kind blend --precond_strength 1.0"
    "--precond_kind multigrid"
)
CFG=${PRECOND[$SLURM_ARRAY_TASK_ID]}     # unquoted below on purpose: splits into flags

OUT=~/TensorPILS/output/sweep_blend/deepritz
# -----------------------------------------------------------------------------

echo "=== deepritz blend sweep: ${CFG} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --loss deepritz --precondition \
    ${CFG} \
    --optimizer adam \
    --n_train 1024 --n_val 128 --n_test 256 \
    -k 4 \
    --epochs 500 \
    --batch_size 32 \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
