#!/bin/bash
# Stokes Table-1 benchmark, STAGE 1: per-arm hyperparameter selection at the FULL budget.
#
# One task per cell of STAGE1_CELLS in common.sh (arm, lr, continuity weight, boundary-penalty
# weight), seed 42, 500 epochs -- the same protocol the Poisson rows' rates came from. Selection
# is on the VALIDATION mean of the velocity and pressure relative L2 (summarize.py). Never the test.
#
# After the sweep, `python .../summarize.py` prints the ranking, the selection and EDGE warnings;
# extend STAGE1_CELLS in common.sh (append only) and submit the new indices with --array=N-M.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/stokes/stokes_paper/stokes_benchmark/lr_sweep.sbatch
#
#SBATCH --job-name=pils-stokes-lr
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/stokes_bench_lr_%A_%a.out
#SBATCH --array=0-57

source ~/env_tensorpils.sh
set -eo pipefail
cd ~/TensorPILS
source experiments/stokes/stokes_paper/stokes_benchmark/common.sh

read -r ARM LR W LAM <<< "${STAGE1_CELLS[$SLURM_ARRAY_TASK_ID]}"
FLAGS=$(arm_flags "$ARM" "$W" "$LAM")
LR_MIN=$(python3 -c "print(${LR}/10)")
OUT=~/TensorPILS/output/stokes/stokes_paper/stokes_benchmark/lr_sweep/${ARM}/$(cell_name "$ARM" "$LR" "$W" "$LAM")

echo "=== stokes benchmark stage 1: arm=${ARM} lr=${LR} w=${W} lam=${LAM} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="
python -m tensorpils.cli ${COMMON} ${FLAGS} \
    --lr "${LR}" --lr_min "${LR_MIN}" --seed 42 \
    --output_dir "${OUT}"
