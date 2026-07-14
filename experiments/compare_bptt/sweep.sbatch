#!/bin/bash
# Compare autodiff/BPTT modes for Allen-Cahn rollout training, across reaction strength eps.
# Three modes (see --bptt_mode):
#   full_bptt    : no detach; backprop through the whole rollout
#   detach_prev  : detach the previous-frame coupling in the loss (MM proximal centre / Galerkin R),
#                  keep full BPTT  -- this is the *current* minimizing-movement default
#   pushforward  : also detach each rollout input, so gradients are one-step (Brandstetter et al.)
#
# One shared sweep over three losses x their modes x eps (48 tasks); plot per-loss afterwards.
# Config-major layout: task t -> CONFIGS[t/6], EPS[t%6].
#   CONFIGS 0-2  minimizing-movement  x {full_bptt, detach_prev, pushforward}
#   CONFIGS 3-5  Galerkin (LS, CC)    x {full_bptt, detach_prev, pushforward}
#   CONFIGS 6-7  data-driven          x {full_bptt, pushforward}   (data has no coupling knob)
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/compare_bptt/sweep.sbatch          # full 48-task array
#     sbatch --array=0-17 experiments/compare_bptt/sweep.sbatch   # e.g. minimizing-movement only
# Watch:   squeue --me      Logs: logs/cmp_bptt_<jobid>_<task>.out
#
#SBATCH --job-name=pils-cmp-bptt
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/cmp_bptt_%A_%a.out
#SBATCH --array=0-47%8

source ~/env_tensorpils.sh
set -eo pipefail

CONFIGS=(
    "--loss galerkin --ac_loss_form min_movement --bptt_mode full_bptt"    # 0 MM full_bptt
    "--loss galerkin --ac_loss_form min_movement --bptt_mode detach_prev"  # 1 MM detach_prev (current)
    "--loss galerkin --ac_loss_form min_movement --bptt_mode pushforward"  # 2 MM pushforward
    "--loss galerkin --ac_loss_form galerkin --bptt_mode full_bptt"        # 3 Galerkin full_bptt (current)
    "--loss galerkin --ac_loss_form galerkin --bptt_mode detach_prev"      # 4 Galerkin detach_prev
    "--loss galerkin --ac_loss_form galerkin --bptt_mode pushforward"      # 5 Galerkin pushforward
    "--loss data --bptt_mode full_bptt"                                    # 6 data full_bptt (current)
    "--loss data --bptt_mode pushforward"                                  # 7 data pushforward
)
EPS=(4 8 12 16 20 24)

t=$SLURM_ARRAY_TASK_ID
CFG=${CONFIGS[$((t / 6))]}     # unquoted below on purpose: splits into flags
E=${EPS[$((t % 6))]}

OUT=~/TensorPILS/output/compare_bptt

echo "=== compare_bptt: eps=${E}  ${CFG} (task ${t}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli --pde ac \
    ${CFG} \
    --ac_integrator convex_concave \
    --ac_a 1.0 --ac_eps "${E}" -k 4 \
    --grid_resolution 64 \
    --dt 0.0025 --n_steps 10 --rollout_steps 10 \
    --n_train 1024 --n_val 128 --n_test 256 \
    --epochs 100 \
    --batch_size 32 \
    --optimizer adam \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
