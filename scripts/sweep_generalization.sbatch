#!/bin/bash
# OOD generalization experiment: train the PLS loss on K=4 with a few convex-blend
# strengths t, for 2500 epochs, and track relative-L2 each epoch on the in-distribution
# K=4 val set plus out-of-distribution K=6 and K=8 sources (same grid). One SLURM array
# task per t value; all tasks share the output dir (files keyed by the blend-<t> tag).
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch scripts/sweep_generalization.sbatch
# Watch:   squeue --me      Logs: logs/gen_<jobid>_<task>.out
#
#SBATCH --job-name=pils-gen-sweep
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --array=0-3

# Source the environment first (module/Lmod is not always `set -u`-safe), then
# enable strict mode for the rest of the script.
source ~/env_tensorpils.sh
set -eo pipefail

# --- sweep configuration -----------------------------------------------------
STRENGTHS=(0.05 0.1 0.25 1.0)
T=${STRENGTHS[$SLURM_ARRAY_TASK_ID]}

OUT=~/TensorPILS/output/generalization
# -----------------------------------------------------------------------------

echo "=== generalization sweep: strength t=${T} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --loss pls \
    --precond_kind blend \
    --precond_strength "${T}" \
    --optimizer adam \
    --n_train 1024 --n_val 128 --n_test 256 \
    -k 4 \
    --ood_k 6 8 \
    --ood_n_val 128 \
    --epochs 2500 \
    --batch_size 32 \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
