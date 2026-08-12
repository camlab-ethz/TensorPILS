#!/bin/bash
# realistic_ac_K16: harder variant of realistic_ac. IDENTICAL setup -- 128^2 grid, eps=32, a=1,
# dt=0.01, 10-step pushforward rollout, n_train=1024, the same four loss arms -- EXCEPT the
# multi-frequency initial condition uses K=16 modes instead of K=4. Higher-frequency, more complex
# ICs make this a substantially harder learning problem. See ../realistic_ac/README.md for the
# full protocol.
#
# Output goes to a SEPARATE tree (output/allen_cahn/realistic_ac_K16) so K=16 runs never mix with
# the K=4 experiment in the analysis (which keys runs by loss, not by K). The run filenames encode
# K (..._K16_...), so nothing collides even if paths were shared -- but a separate dir keeps the
# analysis clean. Reuses the realistic_ac analysis scripts unchanged (see analyze.sbatch here).
#
# One SLURM array task per TRAINING LOSS; all arms share the SAME deterministic reference dataset.
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/allen_cahn/realistic_ac_K16/sweep.sbatch
# Watch:   squeue --me      Logs: logs/realistic_ac_K16_<jobid>_<task>.out
#
#SBATCH --job-name=pils-realistic-ac-K16
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/realistic_ac_K16_%A_%a.out
#SBATCH --array=0-3

# gpumem:20g -- pushforward detaches each rollout input, so the gradient is ONE step (no BPTT
# through the 10-step rollout) -> single-forward activation memory; ample at 128^2. mem-per-cpu:16G
# covers the fp64 CC reference tensor built on the host; bump if the dataset build OOMs.

source ~/env_tensorpils.sh
set -eo pipefail

# --- the training losses (arm-specific flags); pushforward for ALL (best in every prior study) ---
ARMS=(
    "--loss data"                                                            # 0 data-driven (supervised MSE)
    "--loss galerkin --ac_loss_form min_movement"                            # 1 minimizing-movement (JKO / Deep-Ritz)
    "--loss galerkin --ac_loss_form galerkin --ac_loss_integrator convex_concave"  # 2 bare least-squares  ½‖R_cc‖²
    "--loss galerkin --ac_loss_form galerkin --ac_loss_integrator convex_concave --ac_precond multigrid"  # 3 preconditioned LS  ½‖P R‖²
)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}     # unquoted below on purpose: splits into flags

OUT=~/TensorPILS/output/allen_cahn/realistic_ac_K16
# -----------------------------------------------------------------------------

echo "=== realistic_ac_K16: ${ARM} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli --pde ac \
    ${ARM} \
    --bptt_mode pushforward \
    --ac_integrator convex_concave \
    --ac_a 1.0 --ac_eps 32.0 -k 16 \
    --grid_resolution 128 \
    --dt 0.01 --n_steps 10 --rollout_steps 10 \
    --n_train 1024 --n_val 128 --n_test 256 \
    --epochs 500 \
    --batch_size 16 \
    --optimizer adam \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
