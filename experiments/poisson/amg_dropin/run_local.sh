#!/usr/bin/env bash
# Drop-in AMG arm, 3 seeds. Byte-identical to the `pls` row of
# experiments/baselines/poisson/sweep.sbatch except for --precond_kind amg.
#
# Runs the seeds CONCURRENTLY on one GPU: a Poisson run needs ~1.9 GB, and the AmgX applies are
# host-launch-bound rather than compute-bound, so they overlap well. Each seed is its own
# process, which also keeps it to one live AmgX solver per process (two abort at exit).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PYTHON:-$HOME/venvs/tensorgalerkin/bin/python}"
OUT_ROOT="${OUT_ROOT:-$REPO/output/poisson/amg_dropin}"
LOG_DIR="${LOG_DIR:-$REPO/logs/amg_dropin}"
SEEDS=(${SEEDS:-42 43 44})
LR="${LR:-1e-3}"          # the rate the GMG arm was tuned at -- reused on purpose, see README

mkdir -p "$LOG_DIR"
cd "$REPO"

for SEED in "${SEEDS[@]}"; do
    echo "launching seed ${SEED} -> ${LOG_DIR}/seed${SEED}.log"
    PYTHONPATH="$REPO" "$PY" -m tensorpils.cli \
        --pde poisson \
        --model fno \
        --loss pls \
        --precond_kind amg \
        --grid_resolution 64 \
        --n_train 1024 --n_val 128 --n_test 256 \
        -k 4 \
        --epochs 500 \
        --batch_size 32 \
        --seed "${SEED}" \
        --optimizer adam --lr "${LR}" --lr_min 1e-6 \
        --device cuda \
        --no_checkpoint \
        --output_dir "${OUT_ROOT}/seed${SEED}" \
        > "${LOG_DIR}/seed${SEED}.log" 2>&1 &
done

wait
echo "all seeds done"
for SEED in "${SEEDS[@]}"; do
    echo "--- seed ${SEED} ---"
    grep -E 'Test MSE|Test rel-L2|Restored best' "${LOG_DIR}/seed${SEED}.log" || tail -3 "${LOG_DIR}/seed${SEED}.log"
done
