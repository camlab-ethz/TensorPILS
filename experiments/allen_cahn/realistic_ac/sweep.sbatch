#!/bin/bash
# realistic_ac: a realistic-but-simple Allen–Cahn autoregressive benchmark for FEM.
#   grid 256^2, a=1, eps=64 (stiff reaction), K=4 (tame IC), dt=0.01, train on a 10-step
#   pushforward rollout, n_train=1024. See notes/ac_autoregressive/ (§"Numerical soundness")
#   for why 256^2/eps=64/dt=0.01 are matched and how the loss geometry follows from the
#   Newton-Jacobian conditioning kappa ~ 1e4 (bare LS sees kappa^2 ~ 1e8 -> expected to fail).
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
#SBATCH --array=0-2

# gpumem:40g -- 256^2 is 16x the activation footprint of the earlier 64^2 studies. Pushforward
# keeps it manageable: it detaches each rollout input, so the gradient is ONE step (no BPTT
# through the 10-step rollout), i.e. single-forward activation memory. If a 40 GB card is scarce,
# drop --batch_size to 8 (halves activations) before requesting less.
# mem-per-cpu:16G -- the CC reference tensor is [n_train+n_val+n_test, n_steps+1, 256^2] fp64
# (~8 GB); bump if the dataset build OOMs on the host.

source ~/env_tensorpils.sh
set -eo pipefail

# --- the training losses (arm-specific flags); pushforward for ALL (best in every prior study) ---
ARMS=(
    "--loss data"                                                            # 0 data-driven (supervised MSE)
    "--loss galerkin --ac_loss_form min_movement"                            # 1 minimizing-movement (JKO / Deep-Ritz)
    "--loss galerkin --ac_loss_form galerkin --ac_loss_integrator convex_concave"  # 2 bare least-squares  ½‖R_cc‖²
    # 3 PRECONDITIONED least-squares -- add once the AC preconditioned LS loss lands (see README
    #   "Adding the preconditioned-LS arm"); it slots in here as one more array task, e.g.:
    # "--loss galerkin --ac_loss_integrator convex_concave --ac_precond multigrid"
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
    --ac_a 1.0 --ac_eps 64.0 -k 4 \
    --grid_resolution 256 \
    --dt 0.01 --n_steps 10 --rollout_steps 10 \
    --n_train 10 --n_val 128 --n_test 256 \
    --epochs 10 \
    --batch_size 16 \
    --optimizer adam \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
