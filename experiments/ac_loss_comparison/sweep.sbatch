#!/bin/bash
# Compare four FNO training losses for Allen–Cahn on Euler, all on the SAME convex–concave
# reference dataset (deterministic from --seed, so every arm trains on and is evaluated against
# identical data — the rollout-MSE numbers are directly comparable):
#   0  data-driven                (supervised trajectory MSE)
#   1  Galerkin (backward-Euler)  (label-free ½‖R_be‖², BE residual on the CC data)
#   2  Galerkin (convex–concave)  (label-free ½‖R_cc‖²)
#   3  minimizing-movement        (label-free convex–concave JKO objective J)
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/ac_loss_comparison/sweep.sbatch
# Watch:   squeue --me      Logs: logs/ac_losscmp_<jobid>_<task>.out
#
# NOTE: --ac_ref_chunk 1. At grid 64 the interior system is 3844x3844; MAGMA's *batched* LU
# (magma_sgetrf_batched, used by torch.linalg.solve for batch>1) hits an illegal memory access
# at this size, so the reference solve must run one sample at a time (non-batched path). Same
# result, ~serial solves. (Proper fix later: iterative CG on the SPD convex-concave Jacobian.)
#
#SBATCH --job-name=pils-ac-losscmp
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/ac_losscmp_%A_%a.out
#SBATCH --array=0-3

# gpumem:20g -- the 10-step autoregressive rollout does BPTT through 10 FNO forwards, so a
# batch of 32 at grid 64 overruns a small (~10 GB) card. Request >=20 GB. (Fallback if a big
# GPU is scarce: drop --batch_size to 8-16 below, which cuts the unrolled activation memory.)

source ~/env_tensorpils.sh
set -eo pipefail

# --- the four training losses (arm-specific flags) ---------------------------
ARMS=(
    "--loss data"                                          # 0 data-driven
    "--loss galerkin --ac_loss_integrator backward_euler"  # 1 Galerkin backward-Euler
    "--loss galerkin --ac_loss_integrator convex_concave"  # 2 Galerkin convex-concave
    "--loss galerkin --ac_loss_form min_movement"          # 3 minimizing-movement
)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}     # unquoted below on purpose: splits into flags

OUT=~/TensorPILS/output/ac_loss_comparison
# -----------------------------------------------------------------------------

echo "=== ac_loss_comparison: ${ARM} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli --pde ac \
    ${ARM} \
    --ac_integrator convex_concave \
    --ac_a 1.0 --ac_eps 2.0 -k 4 \
    --grid_resolution 64 \
    --dt 0.0025 --n_steps 10 --rollout_steps 10 \
    --ac_newton_tol 1e-6 \
    --ac_ref_chunk 1 \

    --n_train 1024 --n_val 128 --n_test 256 \
    --epochs 1000 \
    --batch_size 32 \
    --optimizer adam \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
