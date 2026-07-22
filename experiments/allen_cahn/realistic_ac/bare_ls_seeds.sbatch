#!/bin/bash
# realistic_ac -- bare (UNpreconditioned) least-squares, 3 seeds, NO nan-guard.
# Purpose: check whether the bare-LS physics loss ever survives training on its own (finishes
# finite, collapsing to ~100% rel-L2) or whether every seed diverges to NaN. This is the diagnostic
# for the observed bare-LS failure -- exact production config (128^2, eps=32, dt=0.01, n_train=1024,
# pushforward), only the seed varies.
#
# One array task per seed. --seed varies the ICs, model init and data order. The AC run filename
# does NOT encode the seed, so each task writes to its OWN output dir (else they collide); the
# parent is separate from the main realistic_ac/ dir so a crash leaves no artifacts there.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/allen_cahn/realistic_ac/bare_ls_seeds.sbatch
# Watch:   squeue --me      Logs: logs/bare_ls_seeds_<jobid>_<task>.out
#
#SBATCH --job-name=pils-bare-ls-seeds
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/bare_ls_seeds_%A_%a.out
#SBATCH --array=0-2

source ~/env_tensorpils.sh
set -eo pipefail

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=~/TensorPILS/output/allen_cahn/realistic_ac_bare_ls/seed${SEED}

echo "=== bare-LS (no guard) seed ${SEED} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli --pde ac \
    --loss galerkin --ac_loss_form galerkin --ac_loss_integrator convex_concave \
    --bptt_mode pushforward \
    --ac_integrator convex_concave \
    --ac_a 1.0 --ac_eps 32.0 -k 4 \
    --grid_resolution 128 \
    --dt 0.01 --n_steps 10 --rollout_steps 10 \
    --n_train 1024 --n_val 128 --n_test 256 \
    --epochs 500 \
    --batch_size 16 \
    --optimizer adam \
    --seed ${SEED} \
    --device cuda \
    --output_dir "${OUT}"
