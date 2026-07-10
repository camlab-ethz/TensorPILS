#!/bin/bash
# Stress-test three Allen-Cahn training losses across increasing reaction strength eps (eps is
# squared in the PDE, so eps=24 => reaction coefficient 576). Reference data is the convex-concave
# (Eyre) FEM integrator, which is unconditionally energy-stable, so it stays well-defined even
# where dt/mesh under-resolve the physics (that is the point of the stress).
#
# 2D SLURM array of 18 tasks = 6 eps values x 3 losses:
#   task t:  eps = EPS[t / 3],  loss = ARMS[t % 3]
#     ARMS: 0 data-driven | 1 Galerkin convex-concave | 2 minimizing-movement
# (backward-Euler dropped: it targets the BE operator, which drifts from the CC reference.)
# 100 epochs for fast turnover. %8 caps concurrency at 8 jobs.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/ac_stresstest/sweep.sbatch                 # all 18
#     sbatch --array=6-8,12-17 experiments/ac_stresstest/sweep.sbatch   # only NEW eps=12,20,24
#       (eps=4,8,16 -> tasks 0-2,3-5,9-11 can be reused from a prior 100-epoch run)
# Watch:   squeue --me      Logs: logs/ac_stress_<jobid>_<task>.out
#
#SBATCH --job-name=pils-ac-stress
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/ac_stress_%A_%a.out
#SBATCH --array=0-17%8

# --ac_ref_chunk 1: at grid 64 the interior system is 3844x3844 and MAGMA's batched LU
# (magma_sgetrf_batched, batch>1) throws an illegal memory access at that size on the cluster
# GPUs -- so the reference solve runs one sample at a time (non-batched path). At large eps the
# Newton solve is stiff (SPD Jacobian M/dt + 3 eps^2 M diag(u^2)); watch the log for convergence.

source ~/env_tensorpils.sh
set -eo pipefail

EPS=(4 8 12 16 20 24)
ARMS=(
    "--loss data"                                          # 0 data-driven
    "--loss galerkin --ac_loss_integrator convex_concave"  # 1 Galerkin convex-concave
    "--loss galerkin --ac_loss_form min_movement"          # 2 minimizing-movement
)
t=$SLURM_ARRAY_TASK_ID
E=${EPS[$((t / 3))]}
ARM=${ARMS[$((t % 3))]}     # unquoted below on purpose: splits into flags

OUT=~/TensorPILS/output/ac_stresstest

echo "=== ac_stresstest: eps=${E}  ${ARM} (task ${t}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli --pde ac \
    ${ARM} \
    --ac_integrator convex_concave \
    --ac_a 1.0 --ac_eps "${E}" -k 4 \
    --grid_resolution 64 \
    --dt 0.0025 --n_steps 10 --rollout_steps 10 \
    --ac_newton_tol 1e-6 \
    --ac_ref_chunk 1 \
    --n_train 1024 --n_val 128 --n_test 256 \
    --epochs 100 \
    --batch_size 32 \
    --optimizer adam \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
