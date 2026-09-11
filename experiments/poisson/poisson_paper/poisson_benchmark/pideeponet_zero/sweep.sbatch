#!/bin/bash
# PI-DeepONet, zero-BC mode -- STAGE 2: the table row (3 seeds), plus the mollified control.
#
# Arm 0 is the table row: --pideeponet_bc zero at the (lr, lambda_bc) that lr_sweep.sbatch
# selected on validation. Arm 1 is the reference implementation's own BC treatment (the
# sin(pi x)sin(pi y) mollifier, no penalty) at its previously selected lr, kept as the labelled
# control the paper reports alongside -- it is the number the table carried before this rerun.
# Edit ZERO_LR / ZERO_LAM if the sweep selects differently.
#
# Submit from ~/TensorPILS after `mkdir -p logs`, and AFTER the lr sweep:
#     sbatch experiments/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/sweep.sbatch
# Then:    python experiments/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/summarize.py
#
#SBATCH --job-name=pils-pidon-zero-final
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/pidon_zero_final_%A_%a.out
#SBATCH --array=0-5

source ~/env_tensorpils.sh
set -eo pipefail

ZERO_LR=1e-4      # selected by lr_sweep.sbatch (best validation rel-L2, seed 42, 500 epochs);
ZERO_LAM=0.03     # summarize.py prints the current selection -- keep these in sync with it
MOLL_LR=1e-4      # the mollified arm's rate from the main benchmark's sweep

ARMS_TAG=(  zero mollifier )
ARMS_FLAGS=(
  "--pideeponet_bc zero --pi_lambda_bc ${ZERO_LAM}"
  "--pideeponet_bc mollifier"
)
ARMS_LR=(   ${ZERO_LR} ${MOLL_LR} )
ARMS_DIR=(  final/lr${ZERO_LR}_lam${ZERO_LAM} control_mollified )
SEEDS=(42 43 44)

N_ARMS=${#ARMS_TAG[@]}
ARM_IDX=$((  SLURM_ARRAY_TASK_ID % N_ARMS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_ARMS ))

TAG=${ARMS_TAG[$ARM_IDX]}
FLAGS=${ARMS_FLAGS[$ARM_IDX]}
LR=${ARMS_LR[$ARM_IDX]}
SEED=${SEEDS[$SEED_IDX]}
LR_MIN=$(python3 -c "print(${LR}/10)")
OUT=~/TensorPILS/output/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/${ARMS_DIR[$ARM_IDX]}/seed${SEED}

echo "=== pideeponet ${TAG}: lr=${LR} seed=${SEED} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --pde poisson \
    --model deeponet --loss pideeponet ${FLAGS} \
    -k 10 \
    --grid_resolution 65 \
    --n_train 1024 --n_val 128 --n_test 256 \
    --epochs 500 \
    --batch_size 32 \
    --optimizer adam \
    --lr "${LR}" --lr_min "${LR_MIN}" \
    --seed "${SEED}" \
    --device cuda \
    --output_dir "${OUT}"
