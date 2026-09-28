#!/usr/bin/env bash
# Figure 2: L_PLS with the spectral blend P_t = (1-t) I + t A^-1, for the t shown in the figure,
# and with the geometric multigrid V-cycle as the practical reference (8 runs). Poisson with
# K = 4 on the 64 x 64 grid, where the dense eigendecomposition behind P_t is cheap.
#
# Run from the repository root:  bash experiments/poisson/blend_sweep/run.sh [task|list]
# (see experiments/common.sh), then  python experiments/poisson/blend_sweep/plot_paper.py
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"

OUT=${OUT:-output/poisson/blend_sweep}
run="$PY -m tensorpils.cli --pde poisson --model fno --loss pls -k 4 --grid_resolution 64 \
--n_train 1024 --n_val 128 --n_test 256 --epochs 500 --batch_size 32 --optimizer adam \
--lr 1e-3 --lr_min 1e-4 --seed 42 --device cuda --no_checkpoint --output_dir $OUT"

TASKS=()
for t in 0.0 0.01 0.05 0.1 0.25 0.75 1.0; do
    TASKS+=("$run --precond_kind blend --precond_strength $t")
done
TASKS+=("$run --precond_kind multigrid")

run_tasks "$@"
