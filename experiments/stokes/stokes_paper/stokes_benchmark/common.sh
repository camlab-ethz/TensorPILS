#!/usr/bin/env bash
# Shared definitions for the Stokes Table-1 benchmark: the common CLI line, the four arms, the
# cell naming and the stage-1 grid. Sourced by run_local.sh, lr_sweep.sbatch and sweep.sbatch so
# the three routes cannot drift apart. Edit grids HERE only, and only by APPENDING cells (sbatch
# array indices map onto STAGE1_CELLS positions).

# Identical to the Poisson Table-1 protocol (K, grid size, splits, budget, optimiser, schedule).
COMMON="--pde stokes -k 10 --grid_resolution 65 --n_train 1024 --n_val 128 --n_test 256 \
  --epochs 500 --batch_size 32 --optimizer adam --device cuda"

# arm_flags ARM W LAM   (W = continuity weight, LAM = boundary-penalty weight; 'x' = not used)
arm_flags() {
  case "$1" in
    data)       echo "--model fno --loss data" ;;
    pls)        echo "--model fno --loss pls --stokes_precond block --schur_omega ${2/x/0.5}" ;;   # W slot = Schur weight omega (x = 0.5)
    pino)       echo "--model fno --loss pino --pi_div_weight $2" ;;
    pideeponet) echo "--model deeponet --loss pideeponet --pideeponet_bc zero --pi_lambda_bc $3 --pi_div_weight $2 --pi_n_colloc 1024" ;;
    *) echo "unknown arm: $1" >&2; return 1 ;;
  esac
}

# cell_name ARM LR W LAM  ->  lr<lr>[_dw<w>|_om<w>][_lam<lam>]   (the directory key summarize.py parses)
# For pls the W slot is the Schur weight omega and is tagged _om; for the baselines it is the
# continuity weight, tagged _dw. 'x' adds no tag (pls: omega 0.5, the draft's default).
cell_name() {
  local s="lr$2"
  if [ "$3" != "x" ]; then if [ "$1" = "pls" ]; then s="${s}_om$3"; else s="${s}_dw$3"; fi; fi
  [ "$4" != "x" ] && s="${s}_lam$4"
  echo "$s"
}

# Stage-1 grid, one "ARM LR W LAM" per entry, seed 42, full budget. Rationale in README.md.
# APPEND only: lr_sweep.sbatch task i runs STAGE1_CELLS[i].
STAGE1_CELLS=()
for lr in 1e-4 3e-4 1e-3 3e-3; do STAGE1_CELLS+=("data $lr x x"); done            #  4
for lr in 1e-4 3e-4 1e-3 3e-3; do STAGE1_CELLS+=("pls $lr x x"); done             #  4
for lr in 3e-4 1e-3 3e-3 1e-2; do for w in 0.3 1 3 10; do
  STAGE1_CELLS+=("pino $lr $w x"); done; done                                     # 16
for lr in 3e-5 1e-4 3e-4; do for lam in 0.01 0.03 0.1; do
  STAGE1_CELLS+=("pideeponet $lr 1 $lam"); done; done                             #  9
# --- edge-rule extensions go below this line (append; note the reason) --------------------
# 2026-09-10: lambda=0.1 was the largest tried and won at every lr (best at lr 1e-4, 3e-5 close),
# so extend upward by one and two half-decades at those two rates.
STAGE1_CELLS+=("pideeponet 1e-4 1 0.3" "pideeponet 1e-4 1 1" "pideeponet 3e-5 1 0.3")   #  3
# 2026-09-10: the continuity weight is the decisive knob for PINO -- w = 0.3/1/3/10 gave 75/59/36/11 %
# validation error at lr 3e-4, monotone and still on the grid's upper edge. Mechanism: the relative
# reduction normalises by ||f|| ~ 193 while the divergence residual of a unit-velocity field is
# O(10), so at small w an irrotational velocity error grad(phi) costs almost nothing (the momentum
# equation absorbs it into the pressure). Extend upward by three half-decades at the three best rates.
for lr in 3e-4 1e-3 3e-3; do for w in 30 100 300; do STAGE1_CELLS+=("pino $lr $w x"); done; done   #  9
# 2026-09-10 (evening): w=300 at lr 3e-3 came out on top (1.5 %) with w on the upper edge again, and
# PI-DeepONet's stage-1b pass put its optimum on the edge too (w=100). One more half-decade each.
STAGE1_CELLS+=("pino 1e-3 1000 x" "pino 3e-3 1000 x")                                             #  2
STAGE1_CELLS+=("pideeponet 1e-4 300 0.1" "pideeponet 1e-4 1000 0.1")                              #  2
# 2026-09-10 (late): the baselines' continuity weight turned out decisive, and PLS has the same knob
# in its Schur block (omega, fixed at the draft's 0.5 so far; the conditioning study puts the optimum
# near 16). Tune it like the baselines' weight: omega in {4, 16} at the two best rates.
for lr in 1e-3 3e-3; do for om in 4 16; do STAGE1_CELLS+=("pls $lr $om x"); done; done            #  4
# omega=16 won at lr 1e-3 (2.14 %) and is the largest tried: one more half-decade, plus the lr
# cross-check below the selected rate at that omega.
STAGE1_CELLS+=("pls 1e-3 64 x" "pls 3e-4 16 x" "pls 3e-4 64 x")                                   #  3
# omega=64 won again (2.05 %, gains flattening: 16 -> 2.14 %); one more step to close the edge.
STAGE1_CELLS+=("pls 1e-3 256 x")                                                                  #  1
# omega=256 -> 1.94 % (from 2.05 %): last step of the edge rule.
STAGE1_CELLS+=("pls 1e-3 1024 x")                                                                 #  1

# Stage 1b: continuity-weight check for PI-DeepONet at its selected (lr, lam).
STAGE1B_W=(3 10 30 100)     # same lesson as PINO: the weight must reach the units of ||f||/||grad u||

# Seeds of the table runs (stage 2). Seed 42 is the stage-1 run itself.
SEEDS=(42 43 44)
