#!/bin/bash
# Sweep the Stokes blend strength t: P_t = (1-t)*alpha*I + t*P_block.
# t=0 reproduces the bare least-squares loss, t=1 the block-preconditioned one, so one array
# spans both rows of the head-to-head with a continuous conditioning axis in between.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/stokes/blend_sweep/sweep.sbatch          # array 0-9 (the t values)
#     sbatch --array=10-11 experiments/stokes/blend_sweep/sweep.sbatch   # the schur_omega arms
#
#SBATCH --job-name=pils-stokes-blend
#SBATCH --time=01:30:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/stokes_blend_%A_%a.out
#SBATCH --array=0-9

source ~/env_tensorpils.sh
set -eo pipefail

# Tasks 0-9 sweep t (spaced logarithmically in 1-t: the identity term dominates until
# 1-t ~ 1/kappa(P), so a uniform grid in t would bunch every interesting point at the end).
# Tasks 10-11 instead hold t=1 and raise schur_omega to where the measured conditioning is
# minimal -- see experiments/stokes/conditioning/omega_scan.py.
CFG=(
    "--stokes_precond_strength 0.0   --output_dir OUT_BLEND"
    "--stokes_precond_strength 0.5   --output_dir OUT_BLEND"
    "--stokes_precond_strength 0.8   --output_dir OUT_BLEND"
    "--stokes_precond_strength 0.9   --output_dir OUT_BLEND"
    "--stokes_precond_strength 0.95  --output_dir OUT_BLEND"
    "--stokes_precond_strength 0.98  --output_dir OUT_BLEND"
    "--stokes_precond_strength 0.99  --output_dir OUT_BLEND"
    "--stokes_precond_strength 0.997 --output_dir OUT_BLEND"
    "--stokes_precond_strength 0.999 --output_dir OUT_BLEND"
    "--stokes_precond_strength 1.0   --output_dir OUT_BLEND"
    "--schur_omega 4  --output_dir OUT_OMEGA"
    "--schur_omega 16 --output_dir OUT_OMEGA"
)
ARGS=${CFG[$SLURM_ARRAY_TASK_ID]}
ARGS=${ARGS//OUT_BLEND/$HOME/TensorPILS/output/stokes/blend_sweep}
ARGS=${ARGS//OUT_OMEGA/$HOME/TensorPILS/output/stokes/omega}

echo "=== stokes blend sweep: ${ARGS} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --pde stokes --loss pls \
    --grid_resolution 65 \
    --n_train 512 --n_val 64 --n_test 128 \
    -k 4 --epochs 300 --batch_size 32 --seed 42 \
    --optimizer adam --lr 2e-3 --lr_min 1e-4 \
    --device cuda \
    ${ARGS}
