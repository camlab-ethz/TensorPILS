#!/usr/bin/env bash
# Stokes past an obstacle: the Stokes rows of Table 1 and the runs behind Figure 5 and the
# appendix sample figure. Four methods x three seeds (tasks 0-11), plus PI-DeepONet with 1024
# random collocation points per step, quoted in the Baseline paragraph (tasks 12-14).
#
# Run from the repository root:  bash experiments/stokes/benchmark/run.sh [task|list]
# (see experiments/common.sh), then  python experiments/stokes/benchmark/summarize.py
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"

OUT=${OUT:-output/stokes/benchmark}
COMMON="--pde stokes --mesh_h 0.035 --mesh_cache_dir output/meshes -k 10 \
--n_train 1024 --n_val 128 --n_test 256 --epochs 500 --batch_size 32 --optimizer adam \
--lr_min 1e-6 --device cuda"
PIDON="--model deeponet --loss pideeponet --pideeponet_bc zero --pi_lambda_bc 0.1 \
--pi_div_weight 100 --lr 1e-4"

TASKS=()
for seed in 42 43 44; do
    run="$PY -m tensorpils.cli $COMMON --seed $seed --output_dir $OUT/final/seed$seed"
    TASKS+=("$run --model gaot --loss pls --schur_omega 16 --lr 1e-3")     # L_PLS
    TASKS+=("$run --model gaot --loss data --lr 1e-3")                     # L_data
    TASKS+=("$run --model gaot --loss galerkin --lr 1e-3")                 # L_LS (no preconditioner)
    TASKS+=("$run $PIDON")                                                 # PI-DeepONet, all nodes
done
for seed in 42 43 44; do
    TASKS+=("$PY -m tensorpils.cli $COMMON --seed $seed --output_dir $OUT/colloc1024/seed$seed \
$PIDON --pi_n_colloc 1024 --no_checkpoint")
done

run_tasks "$@"
