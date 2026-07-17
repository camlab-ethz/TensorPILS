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
# 500 epochs for a converged figure. %8 caps concurrency at 8 jobs.
#
# Submit from ~/TensorPILS after `mkdir -p logs` (run the full array -- 500-epoch runs are not
# comparable to any earlier 100-epoch sweep, so re-run all eps):
#     sbatch experiments/allen_cahn/ac_stresstest/sweep.sbatch      # 18-task array (0-17)
# Watch:   squeue --me      Logs: logs/ac_stress_<jobid>_<task>.out
#
#SBATCH --job-name=pils-ac-stress
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/ac_stress_%A_%a.out
#SBATCH --array=0-17%8

# Reference data is the sparse float64 fem_reference (torch-sla solve). At large eps the Newton
# system is stiff (SPD Jacobian M/dt + 3 eps^2 M diag(u^2)); watch the log for convergence.

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

OUT=~/TensorPILS/output/allen_cahn/ac_stresstest

echo "=== ac_stresstest: eps=${E}  ${ARM} (task ${t}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli --pde ac \
    ${ARM} \
    --ac_integrator convex_concave \
    --ac_a 1.0 --ac_eps "${E}" -k 4 \
    --grid_resolution 64 \
    --dt 0.0025 --n_steps 10 --rollout_steps 10 \
    --n_train 1024 --n_val 128 --n_test 256 \
    --epochs 500 \
    --batch_size 32 \
    --optimizer adam \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
