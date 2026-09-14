#!/bin/bash
# Stokes Table-1 benchmark, STAGE 2: the table rows (four arms x three seeds).
#
# EDIT ARMS_LR / ARMS_W / ARMS_LAM below with what stage 1 selected -- `summarize.py` prints the
# three lines to paste. Output goes to config-named directories, final/<arm>/<cell>/seed<S>, so a
# re-selection never mixes cells. Seed 42 is re-run here for the sbatch route (run_local.sh copies
# the stage-1 run instead).
#
# Submit from ~/TensorPILS after `mkdir -p logs`, AFTER the stage-1 sweep:
#     sbatch experiments/stokes/stokes_paper/stokes_benchmark/sweep.sbatch
# Then:    python experiments/stokes/stokes_paper/stokes_benchmark/summarize.py
#
#SBATCH --job-name=pils-stokes-final
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/stokes_bench_final_%A_%a.out
#SBATCH --array=0-11

source ~/env_tensorpils.sh
set -eo pipefail
cd ~/TensorPILS
source experiments/stokes/stokes_paper/stokes_benchmark/common.sh

ARMS_TAG=( pls  data pino pideeponet )
# --- paste from summarize.py (order pls data pino pideeponet) -----------------------------
ARMS_LR=(  1e-3 3e-4 3e-3 1e-4 )
ARMS_W=(   256  x    300  100  )
ARMS_LAM=( x    x    x    0.1  )
# -----------------------------------------------------------------------------------------

N_ARMS=${#ARMS_TAG[@]}
ARM_IDX=$((  SLURM_ARRAY_TASK_ID % N_ARMS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_ARMS ))
ARM=${ARMS_TAG[$ARM_IDX]}; LR=${ARMS_LR[$ARM_IDX]}; W=${ARMS_W[$ARM_IDX]}; LAM=${ARMS_LAM[$ARM_IDX]}
SEED=${SEEDS[$SEED_IDX]}
FLAGS=$(arm_flags "$ARM" "$W" "$LAM")
LR_MIN=$(python3 -c "print(${LR}/10)")
OUT=~/TensorPILS/output/stokes/stokes_paper/stokes_benchmark/final/${ARM}/$(cell_name "$ARM" "$LR" "$W" "$LAM")/seed${SEED}

echo "=== stokes benchmark stage 2: arm=${ARM} lr=${LR} w=${W} lam=${LAM} seed=${SEED} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="
python -m tensorpils.cli ${COMMON} ${FLAGS} \
    --lr "${LR}" --lr_min "${LR_MIN}" --seed "${SEED}" \
    --output_dir "${OUT}"
