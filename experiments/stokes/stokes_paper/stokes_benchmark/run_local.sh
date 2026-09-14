#!/usr/bin/env bash
# Compute-node driver for the Stokes Table-1 benchmark (the sbatch pair is the login-node route;
# this is for when you are already inside an allocation, e.g. an Euler eu-g* node). Same protocol,
# same output paths, so the routes are interchangeable:
#
#   STAGE=1    seed 42, full 500-epoch budget, every cell of STAGE1_CELLS (common.sh);
#   STAGE=1b   PI-DeepONet continuity weight w in STAGE1B_W at its selected (lr, lam);
#   STAGE=2    seeds 43, 44 at each arm's selected cell (seed 42 copied from stage 1)
#              -> final/<arm>/<cell>/seed<S>/ (config-named, so a re-selection never mixes cells);
#   STAGE=all  1 -> 1b -> 2 (default).
#
# Selection is done by summarize.py --select (lowest best VALIDATION mean(u, p) per arm). After a
# stage, run summarize.py and look for EDGE lines: append the suggested cells to STAGE1_CELLS in
# common.sh and re-run STAGE=1 -- finished cells are skipped (a cell is skipped iff its results
# JSON exists, so a failed cell is simply re-run), which makes the whole script restartable.
#
# Runs PAR cells at a time on one GPU. Measured alone: FNO arms ~2 GB and 1.1-1.6 s/epoch,
# PI-DeepONet at 1024 collocation nodes 4.2 GB and 1.6 s/epoch, so PAR=4 fits a 24 GB card easily
# (the GPU is the bottleneck, not memory).
#
#   PAR=3 bash experiments/stokes/stokes_paper/stokes_benchmark/run_local.sh
#   DRY_RUN=1 bash .../run_local.sh          # print what would run, run nothing
#   STAGE=2 bash .../run_local.sh            # only the seeds
#   python experiments/stokes/stokes_paper/stokes_benchmark/summarize.py
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
cd "$ROOT"
source ~/venvs/tensorgalerkin/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DRY_RUN=${DRY_RUN:-0}
source "$HERE/common.sh"
export COMMON
OUT=$ROOT/output/stokes/stokes_paper/stokes_benchmark
PAR=${PAR:-4}
STAGE=${STAGE:-all}
mkdir -p "$OUT" logs

run_cell() {  # ARM LR W LAM SEED DIR
  local arm=$1 lr=$2 w=$3 lam=$4 seed=$5 dir=$6
  if ls "$dir"/results/*.json >/dev/null 2>&1; then echo "skip  $dir"; return 0; fi
  local flags lr_min
  flags=$(arm_flags "$arm" "$w" "$lam") || return 1
  lr_min=$(python -c "print($lr/10)")
  if [ "$DRY_RUN" = "1" ]; then
    echo "would run: python -m tensorpils.cli $COMMON $flags --lr $lr --lr_min $lr_min --seed $seed --output_dir $dir"; return 0
  fi
  mkdir -p "$dir"
  echo "start $dir  ($(date +%H:%M))"
  python -m tensorpils.cli $COMMON $flags --lr "$lr" --lr_min "$lr_min" \
      --seed "$seed" --output_dir "$dir" > "$dir/train.log" 2>&1 \
    && echo "done  $dir  ($(date +%H:%M))" || echo "FAIL  $dir  (see $dir/train.log)"
}
export -f run_cell arm_flags cell_name
queue() { xargs -P "$PAR" -L 1 bash -c 'run_cell "$0" "$1" "$2" "$3" "$4" "$5"'; }

# selected ARM -> "lr w lam" from summarize.py --select (grid centre under DRY_RUN with no runs)
selected() {
  local line
  line=$(python "$HERE/summarize.py" --root "$OUT" --select 2>/dev/null | awk -v a="$1" '$1==a {print $2, $3, $4}')
  if [ -z "$line" ]; then
    case "$1" in data|pls) line="3e-4 x x" ;; pino) line="3e-3 3 x" ;; pideeponet) line="1e-4 1 0.03" ;; esac
    echo "(no stage-1 result for $1 yet; using the grid centre $line)" >&2
  fi
  echo "$line"
}

if [ "$STAGE" = "1" ] || [ "$STAGE" = "all" ]; then
  echo "=== stage 1: ${#STAGE1_CELLS[@]} cells, seed 42, full budget ==="
  for c in "${STAGE1_CELLS[@]}"; do
    read -r arm lr w lam <<< "$c"
    echo "$arm $lr $w $lam 42 $OUT/lr_sweep/$arm/$(cell_name "$arm" "$lr" "$w" "$lam")"
  done | queue
fi

if [ "$STAGE" = "1b" ] || [ "$STAGE" = "all" ]; then
  read -r lr w lam <<< "$(selected pideeponet)"
  echo "=== stage 1b: pideeponet continuity weight in {${STAGE1B_W[*]}} at lr=$lr lam=$lam ==="
  for w2 in "${STAGE1B_W[@]}"; do
    echo "pideeponet $lr $w2 $lam 42 $OUT/lr_sweep/pideeponet/$(cell_name pideeponet "$lr" "$w2" "$lam")"
  done | queue
fi

if [ "$STAGE" = "2" ] || [ "$STAGE" = "all" ]; then
  echo "=== selection (best validation mean(u, p), seed 42) ==="
  python "$HERE/summarize.py" --root "$OUT" --select || true
  echo "=== stage 2: seeds 43, 44 at the selected cell of each arm -> final/ ==="
  {
    for arm in pideeponet pino pls data; do          # longest first
      read -r lr w lam <<< "$(selected "$arm")"
      cell=$(cell_name "$arm" "$lr" "$w" "$lam")
      if [ "$DRY_RUN" != "1" ]; then
        mkdir -p "$OUT/final/$arm/$cell/seed42"
        cp -rn "$OUT/lr_sweep/$arm/$cell/." "$OUT/final/$arm/$cell/seed42/" 2>/dev/null || true
      fi
      for seed in "${SEEDS[@]}"; do
        [ "$seed" = "42" ] && continue
        echo "$arm $lr $w $lam $seed $OUT/final/$arm/$cell/seed$seed"
      done
    done
  } | queue
fi
echo "=== all done ($STAGE): $(date) ==="
