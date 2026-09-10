#!/bin/bash
# Allen-Cahn benchmark, STAGE 1: learning-rate selection.
#
# Six arms x six rates = 36 tasks, each a FULL 500-epoch run. The full budget is deliberate: the
# Poisson sweep showed the optimum depends on it, and short sweeps systematically prefer rates that
# are too large -- the data arm moved 1e-3 -> 3e-4 and PI-DeepONet's ranking inverted outright when
# the budget went from 100 to 500 epochs.
#
# --no_checkpoint: these runs are never reloaded, and a checkpoint carries the optimizer state as
# well (~3x the parameter count, ~36 MB each). Model selection is unaffected -- the best state is
# held in memory and restored at the end either way.
#
# Physical regime is inherited verbatim from experiments/allen_cahn/realistic_ac: 128^2, eps=32,
# K=4, a=1, dt=0.01, T=10, R=10, pushforward, convex-concave. Fixed, not under investigation.
#
# *** TASK INDICES CHANGED when the minimizing-movement arm was added. *** ARM_IDX = task % N_ARMS,
# so growing N_ARMS from 5 to 6 REMAPS every index: task 5 used to be data@3e-4, it is now mm@1e-4.
# Appending a LEARNING RATE is index-preserving (LR_IDX = task / N_ARMS, N_ARMS unchanged);
# appending an ARM is not. Runs already on disk are unaffected -- they are written under per-arm
# run prefixes -- but do not resubmit an old index list expecting the old meaning.
#
# NOT INCLUDED -- the boundary-penalised PI-DeepONet (lambda_bc=100). PIDeepONetACLoss has no
# lambda_bc and PIDeepONetACTrainer refuses mollify=False; see README.md.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch --array=0-35%15 experiments/allen_cahn/ac_paper/lr_sweep.sbatch          # all six arms
#     sbatch --array=5,11,17,23,29,35%6 experiments/allen_cahn/ac_paper/lr_sweep.sbatch # mm only
# Watch:   squeue --me      Logs: logs/ac_lr_<jobid>_<task>.out
#
#SBATCH --job-name=pils-ac-lr
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/ac_lr_%A_%a.out
#SBATCH --array=0-35

source ~/env_tensorpils.sh
set -eo pipefail

# --- physical regime (fixed; matches realistic_ac exactly) ---------------------
GRID=128
EPS=32.0
K=4
A=1.0
DT=0.01
NSTEPS=10
ROLLOUT=10
EPOCHS=500
BATCH=16
SEED=42

# --- the six arms -------------------------------------------------------------
ARMS_MODEL=(fno  fno  fno  fno  deeponet       fno)
ARMS_TAG=(  data ls   pls  pino pideeponet-mol mm)
ARMS_FLAGS=(
  "--loss data"
  "--loss galerkin --ac_loss_form galerkin --ac_loss_integrator convex_concave"
  "--loss galerkin --ac_loss_form galerkin --ac_loss_integrator convex_concave --ac_precond multigrid"
  "--loss pino"
  "--loss pideeponet"
  "--loss galerkin --ac_loss_form min_movement"
)
# Same grid as the Poisson sweep. Widen if a winner lands on an edge -- append, never reorder.
LRS=(1e-4 3e-4 1e-3 3e-3 1e-2 3e-2)

N_ARMS=${#ARMS_TAG[@]}
ARM_IDX=$(( SLURM_ARRAY_TASK_ID % N_ARMS ))
LR_IDX=$((  SLURM_ARRAY_TASK_ID / N_ARMS ))

MODEL=${ARMS_MODEL[$ARM_IDX]}
TAG=${ARMS_TAG[$ARM_IDX]}
FLAGS=${ARMS_FLAGS[$ARM_IDX]}
LR=${LRS[$LR_IDX]}
# lr_min = lr/10 keeps the cosine schedule's SHAPE fixed across the sweep; a fixed floor would mean
# no decay at the small end and a 300x decay at the large end.
LR_MIN=$(python3 -c "print(${LR}/10)")

# Run prefixes do not encode the learning rate, so each rate needs its own directory.
OUT=~/TensorPILS/output/allen_cahn/ac_paper/lr_sweep/lr${LR}
# ------------------------------------------------------------------------------

echo "=== ac_paper lr sweep: arm=${TAG} lr=${LR} epochs=${EPOCHS} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli --pde ac \
    --model "${MODEL}" \
    ${FLAGS} \
    --bptt_mode pushforward \
    --ac_integrator convex_concave \
    --ac_a ${A} --ac_eps ${EPS} -k ${K} \
    --grid_resolution ${GRID} \
    --dt ${DT} --n_steps ${NSTEPS} --rollout_steps ${ROLLOUT} \
    --n_train 1024 --n_val 128 --n_test 256 \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH} \
    --optimizer adam \
    --lr "${LR}" --lr_min "${LR_MIN}" \
    --seed ${SEED} \
    --no_checkpoint \
    --device cuda \
    --output_dir "${OUT}"
