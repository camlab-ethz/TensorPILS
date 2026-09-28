#!/usr/bin/env bash
# Poisson learning-rate selection, seed 42 at the full 500-epoch budget (lr_min = lr / 10).
#   tasks  0-23  L_PLS, L_data, L_LS and PINO at lr in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}
#   tasks 24-45  PI-DeepONet, learning rate x boundary weight: {3e-5, 1e-4, 3e-4, 1e-3} x
#                {0.3, 1, 3}, extended towards small weights where the optimum sat on the edge
# Selection is on the validation relative L2 (summarize.py prints it).
#
# Run from the repository root:  bash experiments/poisson/benchmark/lr_sweep.sh [task|list]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"

OUT=${OUT:-output/poisson/benchmark}
COMMON="--pde poisson -k 10 --grid_resolution 65 --n_train 1024 --n_val 128 --n_test 256 \
--epochs 500 --batch_size 32 --optimizer adam --device cuda --seed 42 --no_checkpoint"

TASKS=()
for lr in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
    run="$PY -m tensorpils.cli $COMMON --lr $lr --lr_min $(lr_div10 $lr) \
--output_dir $OUT/lr_sweep/lr$lr --model fno"
    TASKS+=("$run --loss pls --mg_omega 0.8888888888888888")
    TASKS+=("$run --loss data")
    TASKS+=("$run --loss galerkin")
    TASKS+=("$run --loss pino")
done
for cell in "3e-5 0.3" "3e-5 1" "3e-5 3" "1e-4 0.3" "1e-4 1" "1e-4 3" "3e-4 0.3" "3e-4 1" \
            "3e-4 3" "1e-3 0.3" "1e-3 1" "1e-3 3" "3e-5 0.1" "3e-5 0.03" "3e-5 0" "1e-4 0.1" \
            "1e-4 0.03" "1e-4 0" "1e-4 0.01" "1e-4 0.003" "1e-4 0.001" "3e-4 0.03"; do
    read -r lr lam <<< "$cell"
    TASKS+=("$PY -m tensorpils.cli $COMMON --lr $lr --lr_min $(lr_div10 $lr) \
--output_dir $OUT/lr_sweep_pideeponet/lr${lr}_lam$lam --model deeponet --loss pideeponet \
--pideeponet_bc zero --pi_lambda_bc $lam")
done

run_tasks "$@"
