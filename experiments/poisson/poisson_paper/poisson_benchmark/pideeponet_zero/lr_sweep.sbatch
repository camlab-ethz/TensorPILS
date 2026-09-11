#!/bin/bash
# PI-DeepONet, zero-BC mode -- STAGE 1: joint (lr, lambda_bc) selection.
#
# --pideeponet_bc zero is the counterpart of the table's PINO arm (--pino_bc zero): the boundary
# nodes of the grid output are zeroed exactly as our own arms do, and the residual carries the
# original PI-DeepONet soft boundary penalty weighted by --pi_lambda_bc (an autodiff Laplacian at
# interior points cannot see a mask, so the penalty is what couples the residual to the BC).
# The penalty weight is a new hyperparameter, so it is tuned JOINTLY with the learning rate, at
# the FULL 500-epoch budget on seed 42 -- the same protocol the other arms' final rates came from.
#
# Grid: lr in {3e-5, 1e-4, 3e-4, 1e-3} x lambda in {0.3, 1, 3}  (12 cells)
#       + lambda in {0.1, 0.03, 0} at lr in {3e-5, 1e-4}          (6 cells: the first grid put the
#         optimum on its lower lambda edge, so it was extended; lambda=0 is the no-BC control)
#       + lambda in {0.01, 0.003, 0.001} at lr 1e-4, and lr 3e-4 at lambda 0.03   (4 cells: the
#         optimum was on the lower edge again, and one lr cross-check at the small-lambda end)
# Selection metric is the VALIDATION relative L2 (stats.val_rel_l2_errors). Never the test set.
#
# Submit from ~/TensorPILS after `mkdir -p logs`:
#     sbatch experiments/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/lr_sweep.sbatch
# Then:    python experiments/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/summarize.py
#
#SBATCH --job-name=pils-pidon-zero-lr
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/pidon_zero_lr_%A_%a.out
#SBATCH --array=0-21

source ~/env_tensorpils.sh
set -eo pipefail

CELLS_LR=( 3e-5 3e-5 3e-5  1e-4 1e-4 1e-4  3e-4 3e-4 3e-4  1e-3 1e-3 1e-3  3e-5 3e-5 3e-5  1e-4 1e-4 1e-4  1e-4 1e-4  1e-4  3e-4 )
CELLS_LAM=(0.3  1    3     0.3  1    3     0.3  1    3     0.3  1    3     0.1  0.03 0     0.1  0.03 0     0.01 0.003 0.001 0.03 )

LR=${CELLS_LR[$SLURM_ARRAY_TASK_ID]}
LAM=${CELLS_LAM[$SLURM_ARRAY_TASK_ID]}
LR_MIN=$(python3 -c "print(${LR}/10)")
OUT=~/TensorPILS/output/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/lr_sweep/lr${LR}_lam${LAM}

echo "=== pideeponet zero-BC lr sweep: lr=${LR} lambda_bc=${LAM} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="

cd ~/TensorPILS
python -m tensorpils.cli \
    --pde poisson \
    --model deeponet --loss pideeponet --pideeponet_bc zero --pi_lambda_bc "${LAM}" \
    -k 10 \
    --grid_resolution 65 \
    --n_train 1024 --n_val 128 --n_test 256 \
    --epochs 500 \
    --batch_size 32 \
    --optimizer adam \
    --lr "${LR}" --lr_min "${LR_MIN}" \
    --seed 42 \
    --device cuda \
    --output_dir "${OUT}"
