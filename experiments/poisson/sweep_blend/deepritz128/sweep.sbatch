#!/bin/bash
# Fine-grid (128²) convex-blend preconditioner-strength sweep for the PRECONDITIONED DEEP RITZ
# loss, using the fast DST preconditioner (--precond_method sine; the dense eigendecomposition is
# infeasible at this size). The surrogate descent direction is P·r, from the plain energy gradient
# (t=0, P=I) to the supervised/Newton direction (t=1, P=A^-1). One SLURM array task per t.
# Edit GRID / EPOCHS / TVALS below to retune.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/poisson/sweep_blend/deepritz128/sweep.sbatch
# Watch:   squeue --me      Logs: logs/blend_dr128_<jobid>_<task>.out
#
#SBATCH --job-name=pils-blend-dr128
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/blend_dr128_%A_%a.out
#SBATCH --array=0-5

# Source the environment first (module/Lmod is not always `set -u`-safe), then strict mode.
source ~/env_tensorpils.sh
set -eo pipefail

# --- sweep configuration (edit freely) ---------------------------------------
GRID=128
EPOCHS=20
TVALS=(1.0 0.9 0.5 0.25 0.01 0.0)      # blend strengths t; keep --array as 0-$((${#TVALS[@]}-1))
# Tasks 0..N-1 sweep t via the fast DST ("sine") preconditioner. Task N (= ${#TVALS[@]}) is the
# multigrid (GMG) baseline; add it with:  sbatch --array=6 <this file>.
NT=${#TVALS[@]}
if [ "${SLURM_ARRAY_TASK_ID}" -lt "${NT}" ]; then
    CFG="--precond_kind blend --precond_strength ${TVALS[${SLURM_ARRAY_TASK_ID}]} --precond_method sine"
else
    CFG="--precond_kind multigrid"
fi

OUT=~/TensorPILS/output/poisson/sweep_blend/deepritz128
# -----------------------------------------------------------------------------

echo "=== deepritz128 blend sweep: ${CFG} (task ${SLURM_ARRAY_TASK_ID}, grid ${GRID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --loss deepritz --precondition \
    ${CFG} \
    --grid_resolution ${GRID} \
    --optimizer adam \
    --n_train 1024 --n_val 128 --n_test 256 \
    -k 4 \
    --epochs ${EPOCHS} \
    --batch_size 32 \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
