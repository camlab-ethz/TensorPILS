#!/usr/bin/env bash
# Stokes hyperparameter search, seed 42 at the full 500-epoch budget: the learning rate of
# L_data, the Schur weight omega_S of L_PLS (the preconditioner ablation table), and PI-DeepONet's
# learning rate and continuity weight at 1024 random collocation points per step. Selection is on
# the validation mean of the velocity and pressure relative errors.
#
# Run from the repository root:  bash experiments/stokes/benchmark/grid_search.sh [task|list]
# (see experiments/common.sh), then  python experiments/stokes/benchmark/summarize.py
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"

OUT=${OUT:-output/stokes/benchmark}
COMMON="--pde stokes --mesh_h 0.035 --mesh_cache_dir output/meshes -k 10 \
--n_train 1024 --n_val 128 --n_test 256 --epochs 500 --batch_size 32 --optimizer adam \
--lr_min 1e-6 --device cuda --seed 42 --no_checkpoint"

TASKS=()
for lr in 3e-4 1e-3; do
    TASKS+=("$PY -m tensorpils.cli $COMMON --output_dir $OUT/grid_search/data_lr$lr \
--model gaot --loss data --lr $lr")
done
for omega in 0.5 4 16 32 64 256; do
    TASKS+=("$PY -m tensorpils.cli $COMMON --output_dir $OUT/grid_search/pls_omega$omega \
--model gaot --loss pls --schur_omega $omega --lr 1e-3")
done
for cfg in "1e-4 100" "3e-4 100" "1e-4 1" "1e-4 1000"; do
    read -r lr w <<< "$cfg"
    TASKS+=("$PY -m tensorpils.cli $COMMON --output_dir $OUT/grid_search/pideeponet_lr${lr}_w$w \
--model deeponet --loss pideeponet --pideeponet_bc zero --pi_lambda_bc 0.1 --pi_n_colloc 1024 \
--lr $lr --pi_div_weight $w")
done

run_tasks "$@"
