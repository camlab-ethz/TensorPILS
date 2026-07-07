#!/bin/bash
# Loss comparison: train the FNO with several supervised/physics losses at default settings
# for 1000 epochs, and compare validation relative-L2 over epochs (plot flag in brackets):
#   data                              - supervised MSE                              [mse]
#   data_l2                           - supervised true-L2 (½ eᵀ M e)               [l2]
#   data_h1                           - supervised H¹₀ (½ eᵀ A e, boundary-proj.)   [h1]
#   deepritz (penalty BC)             - Deep Ritz energy, penalty BC               [dr_penalty]
#   deepritz (hard BC)                - Deep Ritz energy, hard BC (= precond t=0)  [dr_t0]
#   deepritz precond blend t=1.0      - preconditioned Deep Ritz, t=1              [dr_t1]
#   pls blend t=1.0                   - preconditioned least-squares, t=1          [pls_t1]
#   pls blend t=0.5                   - preconditioned least-squares, t=0.5        [pls_t0.5]
# One SLURM array task per config; files keyed by the loss/precond tag, shared output dir.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/loss_comparison/sweep.sbatch
# Watch:   squeue --me      Logs: logs/losscmp_<jobid>_<task>.out
#
#SBATCH --job-name=pils-loss-cmp
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/losscmp_%A_%a.out
#SBATCH --array=0-7

# Source the environment first (module/Lmod is not always `set -u`-safe), then
# enable strict mode for the rest of the script.
source ~/env_tensorpils.sh
set -eo pipefail

# --- experiment configuration ------------------------------------------------
# One entry per task; full CLI flags so the two Deep Ritz BC variants can differ.
CONFIGS=(
    "--loss data"                                                                 # mse
    "--loss data_l2"                                                              # l2
    "--loss data_h1"                                                              # h1
    "--loss deepritz"                                                             # dr_penalty
    "--loss deepritz --bc_mode hard"                                              # dr_t0
    "--loss deepritz --precondition --precond_kind blend --precond_strength 1.0"  # dr_t1
    "--loss pls --precond_kind blend --precond_strength 1.0"                      # pls_t1
    "--loss pls --precond_kind blend --precond_strength 0.5"                      # pls_t0.5
)
CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

OUT=~/TensorPILS/output/loss_comparison
# -----------------------------------------------------------------------------

echo "=== loss comparison: [${CFG}] (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    ${CFG} \
    --n_train 1024 --n_val 128 --n_test 256 \
    -k 4 \
    --epochs 1000 \
    --batch_size 32 \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
