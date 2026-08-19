#!/bin/bash
# Initialization probe: one 1-epoch run per arm at a learning rate small enough to be a no-op.
#
# WHY. The trainer records validation only *after* each epoch (trainer.py: train_epoch() then
# validate()), so no run ever logs its error at initialization and the sweep curves all start
# 32 optimizer steps in -- at six different places, one per learning rate. This probe recovers
# the missing epoch-0 point without touching the trainer or the CLI: with lr = 1e-12, Adam moves
# the parameters by ~1e-12 per step, which over 32 steps is invisible in float32, so the epoch-1
# validation number IS the initialization error.
#
# It is a measurement, not a cosmetic: "all six lr runs share an init" is currently an inference
# from the seed being 42 everywhere. This checks it.
#
# Cheap enough to run on a laptop -- 7 arms x 1 epoch. CPU rather than MPS because the
# preconditioned arms use sparse operators, which MPS does not fully support.
#
# Usage (from the repo root):
#     bash experiments/poisson/poisson_paper/init_probe.sh
# Then plot_lr_sweep.py picks it up automatically and prepends it as epoch 0.

set -eo pipefail

OMEGA=0.8888888888888888          # 8/9, the optimal Jacobi damping for Q1 in 2D
LR=1e-12                          # effectively zero: no parameter moves in float32
DEVICE=${DEVICE:-cpu}
OUT=${OUT:-output/poisson/poisson_paper/init_probe}
# Prefer the repo venv: the other local env on this machine carries a stale tensormesh
# (0.1.1, no `Field`) and fails at import, the same way Euler did.
REPO_VENV="$(cd "$(dirname "$0")/../../.." && pwd)/.venv/bin/python"
PYTHON=${PYTHON:-$([ -x "$REPO_VENV" ] && echo "$REPO_VENV" || echo python)}

# Must match lr_sweep.sbatch exactly, or the probe measures a different model.
ARMS_MODEL=(fno fno fno fno fno fno deeponet)
ARMS_TAG=(  data ls pls deepritz pdeepritz pino pideeponet)
ARMS_FLAGS=(
  "--loss data"
  "--loss galerkin"
  "--loss pls --precond_kind multigrid --mg_omega ${OMEGA}"
  "--loss deepritz --bc_mode hard"
  "--loss deepritz --bc_mode hard --precondition --precond_kind multigrid --mg_omega ${OMEGA}"
  "--loss pino"
  "--loss pideeponet"
)

for i in "${!ARMS_TAG[@]}"; do
  echo "=== init probe: arm=${ARMS_TAG[$i]} (${DEVICE}) ==="
  "${PYTHON}" -m tensorpils.cli \
      --pde poisson \
      --model "${ARMS_MODEL[$i]}" \
      ${ARMS_FLAGS[$i]} \
      -k 10 \
      --grid_resolution 65 \
      --n_train 1024 --n_val 128 --n_test 256 \
      --epochs 1 \
      --batch_size 32 \
      --optimizer adam \
      --lr "${LR}" --lr_min "${LR}" \
      --seed 42 \
      --device "${DEVICE}" \
      --output_dir "${OUT}"
done

echo
echo "done -> ${OUT}/results/"
