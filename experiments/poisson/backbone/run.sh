#!/usr/bin/env bash
# The backbone table of the appendix: the three Poisson losses with an FNO and with GAOT, on the
# structured K = 4 problem (64 x 64 grid), seeds 42, 43 and 44 (18 runs).
#
# Run from the repository root:  bash experiments/poisson/backbone/run.sh [task|list]
# (see experiments/common.sh), then  python experiments/poisson/backbone/summarize.py
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"

OUT=${OUT:-output/poisson/backbone}
COMMON="--pde poisson -k 4 --grid_resolution 64 --n_train 1024 --n_val 128 --n_test 256 \
--epochs 500 --batch_size 32 --optimizer adam --lr 1e-3 --lr_min 1e-6 --device cuda \
--no_checkpoint"

TASKS=()
for seed in 42 43 44; do
    for model in fno gaot; do
        for loss in data pls galerkin; do
            TASKS+=("$PY -m tensorpils.cli $COMMON --seed $seed --output_dir $OUT/seed$seed \
--model $model --loss $loss")
        done
    done
done

run_tasks "$@"
