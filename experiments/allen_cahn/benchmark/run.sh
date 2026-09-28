#!/usr/bin/env bash
# Allen-Cahn: the Allen-Cahn rows of Table 1 and the runs behind Figure 4. Five methods x three
# seeds (15 runs) at the learning rates selected by lr_sweep.sh, on the 129 x 129 grid (whose
# multigrid hierarchy 129 -> 65 -> 33 -> 17 is nested). The checkpoints are kept for
# evaluate_test.py.
#
# Run from the repository root:  bash experiments/allen_cahn/benchmark/run.sh [task|list]
# (see experiments/common.sh), then evaluate_test.py and plot_paper.py (README.md).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"

OUT=${OUT:-output/allen_cahn/benchmark}
COMMON="--pde ac --ac_a 1.0 --ac_eps 32.0 -k 4 --grid_resolution 129 --dt 0.01 --n_steps 10 \
--rollout_steps 10 --n_train 1024 --n_val 128 --n_test 256 --epochs 500 --batch_size 16 \
--optimizer adam --device cuda"

TASKS=()
for seed in 42 43 44; do
    run="$PY -m tensorpils.cli $COMMON --seed $seed --output_dir $OUT/final/seed$seed"
    TASKS+=("$run --model fno --loss galerkin --ac_precond multigrid --lr 3e-4 --lr_min $(lr_div10 3e-4)")
    TASKS+=("$run --model fno --loss data --lr 3e-4 --lr_min $(lr_div10 3e-4)")
    TASKS+=("$run --model fno --loss galerkin --lr 1e-3 --lr_min $(lr_div10 1e-3)")
    TASKS+=("$run --model fno --loss pino --lr 1e-3 --lr_min $(lr_div10 1e-3)")
    TASKS+=("$run --model deeponet --loss pideeponet --lr 1e-3 --lr_min $(lr_div10 1e-3)")
done

run_tasks "$@"
