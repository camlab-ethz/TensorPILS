#!/bin/bash
# Fine-grid (256²) convex-blend preconditioner-strength sweep for the PLS loss, using the fast
# DST preconditioner (--precond_method sine; the dense eigendecomposition needs a ~33 GB
# eigenvector matrix at this size and will not run). One SLURM array task per blend strength t.
# Preset to the 128 sanity config (EPOCHS=20); raise EPOCHS once the pipeline is confirmed.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/poisson/sweep_blend/pls256/sweep.sbatch
# Watch:   squeue --me      Logs: logs/blend_pls256_<jobid>_<task>.out
#
#SBATCH --job-name=pils-blend-pls256
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/blend_pls256_%A_%a.out
#SBATCH --array=0-5

# Source the environment first (module/Lmod is not always `set -u`-safe), then strict mode.
source ~/env_tensorpils.sh
set -eo pipefail

# --- sweep configuration (edit freely) ---------------------------------------
GRID=256
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

OUT=~/TensorPILS/output/poisson/sweep_blend/pls256
# -----------------------------------------------------------------------------

echo "=== pls256 blend sweep: ${CFG} (task ${SLURM_ARRAY_TASK_ID}, grid ${GRID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --loss pls \
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
