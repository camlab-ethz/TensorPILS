#!/usr/bin/env bash
# Figure 3: test error against training-set size at a fixed optimisation budget of 32,000 steps
# (1000 virtual epochs of 32 steps). L_PLS and L_data at six sizes (tasks 0-11), and L_PLS on an
# infinite stream of fresh samples (task 12). Poisson with K = 10 on the 65 x 65 grid, FEM labels,
# validation and test sets shared by all runs.
#
# Run from the repository root:  bash experiments/poisson/infinite_data/run.sh [task|list]
# (see experiments/common.sh), then  python experiments/poisson/infinite_data/plot_scaling.py
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"

OUT=${OUT:-output/poisson/infinite_data}
run="$PY -m tensorpils.cli --pde poisson --model fno -k 10 --grid_resolution 65 \
--dataset_solution fem --n_val 128 --n_test 256 --steps_per_epoch 32 --epochs 1000 \
--batch_size 32 --optimizer adam --lr 3e-4 --lr_min 3e-5 --seed 42 --device cuda \
--no_checkpoint --output_dir $OUT"
PLS="--loss pls --mg_omega 0.8888888888888888"

TASKS=()
for n in 64 128 256 512 1024 2048; do
    TASKS+=("$run --n_train $n --fixed_eval --loss data")
    TASKS+=("$run --n_train $n --fixed_eval $PLS")
done
TASKS+=("$run --stream $PLS")

run_tasks "$@"
