#!/bin/bash
# realistic_ac: a realistic-but-simple Allen–Cahn autoregressive benchmark for FEM.
#   Target regime: grid 256^2, eps=64, a=1, K=4 (tame IC), dt=0.01, n_train=1024 -- see
#   notes/ac_autoregressive/ (§"Numerical soundness") for why these are matched and how the loss
#   geometry follows from the Newton-Jacobian conditioning kappa ~ 1e4 (bare LS sees kappa^2 ~ 1e8).
#
#   *** Currently set to a fast 128^2 / eps=32 SANITY-CHECK on a small dataset. *** Halving BOTH
#   grid and eps leaves the interface resolution ell/h = a(n-1)/eps ~ 4 unchanged -- the same
#   physical regime at a quarter the cost, so it validates the runs end-to-end (incl. the
#   preconditioner, c = 1/dt + 3eps^2 = 3172) before the full run. Restore 256^2 / eps=64 /
#   n_train=1024 / epochs=1000 for production.
#
# One SLURM array task per TRAINING LOSS. All arms share the SAME convex-concave reference
# dataset (deterministic from --seed), so the rollout metrics are directly comparable. Runs are
# auto-tagged by loss/form/integrator/bptt/eps/... so their result & checkpoint files never collide.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/allen_cahn/realistic_ac/sweep.sbatch
# Watch:   squeue --me      Logs: logs/realistic_ac_<jobid>_<task>.out
#
#SBATCH --job-name=pils-realistic-ac
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/realistic_ac_%A_%a.out
#SBATCH --array=0-3

# gpumem:20g -- ample for the 128^2 sanity config, and enough for the 256^2 production grid too:
# pushforward detaches each rollout input, so the gradient is ONE step (no BPTT through the 10-step
# rollout) -- single-forward activation memory. At 256^2 the peak is ~8-10 GB, dominated by the
# fp64 CC reference tensor [n_train+n_val+n_test, n_steps+1, N] built on the GPU (freed before
# training). mem-per-cpu:16G covers that tensor on the host; bump if the dataset build OOMs.

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

OUT=~/TensorPILS/output/allen_cahn/realistic_ac
# -----------------------------------------------------------------------------

echo "=== realistic_ac: ${ARM} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli --pde ac \
    ${ARM} \
    --bptt_mode pushforward \
    --ac_integrator convex_concave \
    --ac_a 1.0 --ac_eps 32.0 -k 4 \
    --grid_resolution 128 \
    --dt 0.01 --n_steps 10 --rollout_steps 10 \
    --n_train 1024 --n_val 128 --n_test 256 \
    --epochs 500 \
    --batch_size 16 \
    --optimizer adam \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
