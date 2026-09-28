#!/usr/bin/env bash
# Poisson: the Poisson rows of Table 1. Five methods x three seeds (15 runs) at the learning rates
# selected by lr_sweep.sh. The checkpoints are kept: the resolution figure of the appendix
# evaluates these models on other grids.
#
# Run from the repository root:  bash experiments/poisson/benchmark/run.sh [task|list]
# (see experiments/common.sh), then  python experiments/poisson/benchmark/summarize.py
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"

OUT=${OUT:-output/poisson/benchmark}
COMMON="--pde poisson -k 10 --grid_resolution 65 --n_train 1024 --n_val 128 --n_test 256 \
--epochs 500 --batch_size 32 --optimizer adam --device cuda"

TASKS=()
for seed in 42 43 44; do
    run="$PY -m tensorpils.cli $COMMON --seed $seed --output_dir $OUT/final/seed$seed"
    TASKS+=("$run --model fno --loss pls --mg_omega 0.8888888888888888 \
--lr 3e-4 --lr_min $(lr_div10 3e-4)")
    TASKS+=("$run --model fno --loss data --lr 3e-4 --lr_min $(lr_div10 3e-4)")
    TASKS+=("$run --model fno --loss galerkin --lr 3e-3 --lr_min $(lr_div10 3e-3)")
    TASKS+=("$run --model fno --loss pino --lr 3e-3 --lr_min $(lr_div10 3e-3)")
    TASKS+=("$run --model deeponet --loss pideeponet --pideeponet_bc zero --pi_lambda_bc 0.03 \
--lr 1e-4 --lr_min $(lr_div10 1e-4)")
done

run_tasks "$@"
