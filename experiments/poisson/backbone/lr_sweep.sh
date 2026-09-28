#!/usr/bin/env bash
# Learning rate of the GAOT runs in the backbone table: the three losses at
# lr in {1e-4, 3e-4, 1e-3, 3e-3}, seed 42, 150 epochs (12 runs). The FNO runs use 1e-3.
#
# Run from the repository root:  bash experiments/poisson/backbone/lr_sweep.sh [task|list]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"

OUT=${OUT:-output/poisson/backbone}
COMMON="--pde poisson --model gaot -k 4 --grid_resolution 64 --n_train 1024 --n_val 128 \
--n_test 256 --epochs 150 --batch_size 32 --optimizer adam --lr_min 1e-6 --seed 42 \
--device cuda --no_checkpoint"

TASKS=()
for loss in data galerkin pls; do
    for lr in 1e-4 3e-4 1e-3 3e-3; do
        TASKS+=("$PY -m tensorpils.cli $COMMON --lr $lr --output_dir $OUT/lr_sweep/lr$lr \
--loss $loss")
    done
done

run_tasks "$@"
