#!/bin/bash
# Monolithic Stokes multigrid (symmetric Uzawa smoother): a genuine P ~ K^-1, which makes the
# APPLIED loss form 0.5*||P r||^2 viable at O(1) conditioning instead of the block form's O(h^-2).
#
# Three arms at one grid, against the block baselines already in output/stokes/h_refinement:
#   0  monolithic P, applied form in the FE metric  -- the one that works (best label-free arm)
#   1  monolithic P, applied form in nodal units     -- same conditioning, ~350x pressure-dominated
#   2  monolithic P, weighted form                   -- diverges: an approximate inverse of an
#                                                       indefinite operator is not a norm
#   3  block P, applied form                         -- the note's documented dead end, as a control
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/stokes/monolithic/sweep.sbatch      # array 0-3
#
#SBATCH --job-name=pils-stokes-mono
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/stokes_mono_%A_%a.out
#SBATCH --array=0-3

source ~/env_tensorpils.sh
set -eo pipefail

# omega=1.0 for the monolithic arms: inside an Uzawa *iteration* the note's omega in (0,1]
# restriction is real (omega=2 diverges), and 1.0 is the measured optimum. The block control
# keeps the 0.5 it was tuned at.
CFG=(
    "--stokes_precond monolithic --stokes_pls_form applied_fe --schur_omega 1.0"
    "--stokes_precond monolithic --stokes_pls_form applied    --schur_omega 1.0"
    "--stokes_precond monolithic --stokes_pls_form weighted   --schur_omega 1.0"
    "--stokes_precond block      --stokes_pls_form applied    --schur_omega 0.5"
)
ARGS=${CFG[$SLURM_ARRAY_TASK_ID]}

OUT=~/TensorPILS/output/stokes/monolithic

echo "=== stokes monolithic: ${ARGS} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --pde stokes --loss pls \
    ${ARGS} \
    --uzawa_pre 4 --uzawa_post 4 --cheb_degree 8 \
    --grid_resolution 33 \
    --n_train 512 --n_val 64 --n_test 128 \
    -k 4 --epochs 300 --batch_size 32 --seed 42 \
    --optimizer adam --lr 2e-3 --lr_min 1e-4 \
    --device cuda \
    --output_dir "${OUT}"
