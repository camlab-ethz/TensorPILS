#!/bin/bash
# Allen-Cahn benchmark, STAGE 2 (finals) -- SCAFFOLD, NOT YET RUNNABLE.
#
# *** DO NOT SUBMIT AS IS. *** Two things are unresolved and are marked <<< DECIDE >>> below:
#   - the per-arm learning rates (placeholders; run lr_sweep.sbatch first),
#   - whether the minimizing-movement arm is included.
# The grid/eps regime is settled: 128^2 / eps=32, matching realistic_ac.
# See README.md in this directory. The physical settings are inherited from
# experiments/allen_cahn/realistic_ac/sweep.sbatch, the arm list from
# experiments/baselines/allen_cahn/sweep.sbatch.
#
#SBATCH --job-name=pils-ac-bench
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/ac_bench_%A_%a.out
#SBATCH --array=0-14

source ~/env_tensorpils.sh
set -eo pipefail

# --- physical regime -----------------------------------------------------------------
# Fixed at 128/32, matching realistic_ac. Halving both grid and eps holds the interface
# resolution l/h = a(n-1)/eps ~ 4, so this is the same physical regime at a quarter the cost.
GRID=128
EPS=32.0
EPOCHS=500
K=4
A=1.0
DT=0.01
NSTEPS=10
ROLLOUT=10
BATCH=16

# --- the five arms -------------------------------------------------------------------
# <<< DECIDE >>> minimizing movement ("--loss galerkin --ac_loss_form min_movement") is the
# best arm in the preliminary study and is omitted here only because the paper table lists five.
ARMS_MODEL=(fno  fno      fno      fno  deeponet)
ARMS_TAG=(  data ls       pls      pino pideeponet)
ARMS_FLAGS=(
  "--loss data"
  "--loss galerkin --ac_loss_form galerkin --ac_loss_integrator convex_concave"
  "--loss galerkin --ac_loss_form galerkin --ac_loss_integrator convex_concave --ac_precond multigrid"
  "--loss pino"
  "--loss pideeponet"
)
# <<< DECIDE >>> placeholders. Run lr_sweep.sbatch first; the Poisson sweep showed the optimum
# moves with the epoch budget and differs per arm by up to 30x.
#            data  ls    pls   pino  pideeponet
ARMS_LR=(    1e-3  1e-3  1e-3  1e-3  1e-3)
SEEDS=(42 43 44)

N_ARMS=${#ARMS_TAG[@]}
ARM_IDX=$((  SLURM_ARRAY_TASK_ID % N_ARMS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_ARMS ))

MODEL=${ARMS_MODEL[$ARM_IDX]}
TAG=${ARMS_TAG[$ARM_IDX]}
FLAGS=${ARMS_FLAGS[$ARM_IDX]}
LR=${ARMS_LR[$ARM_IDX]}
LR_MIN=$(python3 -c "print(${LR}/10)")
SEED=${SEEDS[$SEED_IDX]}

OUT=~/TensorPILS/output/allen_cahn/ac_paper/final/seed${SEED}
# --------------------------------------------------------------------------------------

echo "=== ac_benchmark: arm=${TAG} lr=${LR} seed=${SEED} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

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
    --seed "${SEED}" \
    --device cuda \
    --output_dir "${OUT}"
