#!/usr/bin/env bash
# Allen-Cahn learning-rate selection: five methods x lr in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}
# (30 runs), seed 42 at the full 500-epoch budget, lr_min = lr / 10. Selection is on the
# validation space-time relative L2 (plot_lr_sweep.py prints it). The paper's sweep ran on the
# 128 x 128 grid; the finals (run.sh) moved to 129 x 129 for a nested multigrid hierarchy.
#
# Run from the repository root:  bash experiments/allen_cahn/benchmark/lr_sweep.sh [task|list]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"

OUT=${OUT:-output/allen_cahn/benchmark}
COMMON="--pde ac --ac_a 1.0 --ac_eps 32.0 -k 4 --grid_resolution 128 --dt 0.01 --n_steps 10 \
--rollout_steps 10 --n_train 1024 --n_val 128 --n_test 256 --epochs 500 --batch_size 16 \
--optimizer adam --device cuda --seed 42 --no_checkpoint"

TASKS=()
for lr in 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
    run="$PY -m tensorpils.cli $COMMON --lr $lr --lr_min $(lr_div10 $lr) \
--output_dir $OUT/lr_sweep/lr$lr"
    TASKS+=("$run --model fno --loss galerkin --ac_precond multigrid")
    TASKS+=("$run --model fno --loss data")
    TASKS+=("$run --model fno --loss galerkin")
    TASKS+=("$run --model fno --loss pino")
    TASKS+=("$run --model deeponet --loss pideeponet")
done

run_tasks "$@"
