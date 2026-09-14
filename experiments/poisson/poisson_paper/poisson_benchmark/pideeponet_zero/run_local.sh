#!/usr/bin/env bash
# Compute-node driver for the whole PI-DeepONet zero-BC study (the sbatch files are the
# login-node route; this one is for when you are already inside an allocation, e.g. an Euler
# eu-g* node). Same protocol, same output paths, so the two routes are interchangeable:
#
#   stage 1  seed 42, full 500-epoch budget, the 22-cell (lr, lambda_bc) grid of lr_sweep.sbatch;
#            selection on best VALIDATION rel-L2;
#   stage 2  seeds 43, 44 at the selected cell (seed 42 is copied from stage 1)
#            -> final/lr<lr>_lam<lambda>/ (config-named, so re-selection never mixes configs);
#   control  the reference's mollifier at its own selected lr, seeds 42-44 -> control_mollified/.
#
# Runs PAR cells at a time on one GPU. Each PI-DeepONet process needs ~7 GB (the double-backward
# graph over all interior nodes), so PAR=3 is the most a 24 GB card takes -- a fourth one OOMs.
# Every cell skips itself if its results JSON exists, so the script is restartable; a cell that
# failed (no JSON) is simply re-run.
#
#   PAR=3 bash experiments/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/run_local.sh
#   DRY_RUN=1 bash .../run_local.sh        # print what would run, run nothing
#   python experiments/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/summarize.py
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)
cd "$ROOT"
source ~/venvs/tensorgalerkin/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export DRY_RUN=${DRY_RUN:-0}
OUT=$ROOT/output/poisson/poisson_paper/poisson_benchmark/pideeponet_zero
PAR=${PAR:-3}
MOLL_LR=1e-4                      # the mollified arm's rate from the main benchmark's sweep
mkdir -p "$OUT" logs

export COMMON="--pde poisson --model deeponet --loss pideeponet \
  -k 10 --grid_resolution 65 --n_train 1024 --n_val 128 --n_test 256 \
  --epochs 500 --batch_size 32 --optimizer adam --device cuda"

run_cell() {  # bc-flags lr seed outdir
  local flags=$1 lr=$2 seed=$3 dir=$4
  if ls "$dir"/results/*.json >/dev/null 2>&1; then echo "skip  $dir"; return 0; fi
  local lr_min; lr_min=$(python -c "print($lr/10)")
  if [ "$DRY_RUN" = "1" ]; then
    echo "would run: python -m tensorpils.cli $COMMON $flags --lr $lr --lr_min $lr_min --seed $seed --output_dir $dir"; return 0
  fi
  mkdir -p "$dir"
  echo "start $dir  ($(date +%H:%M))"
  python -m tensorpils.cli $COMMON $flags --lr "$lr" --lr_min "$lr_min" \
      --seed "$seed" --output_dir "$dir" > "$dir/train.log" 2>&1 \
    && echo "done  $dir  ($(date +%H:%M))" || echo "FAIL  $dir  (see $dir/train.log)"
}
export -f run_cell
# xargs splits each queued line into: lr lambda seed dir  (flags are rebuilt from lambda)
queue() { xargs -P "$PAR" -L 1 bash -c 'run_cell "--pideeponet_bc zero --pi_lambda_bc $1" "$0" "$2" "$3"'; }

echo "=== stage 1: (lr, lambda_bc) grid, seed 42 ==="
{
  for lr in 3e-5 1e-4 3e-4 1e-3; do for lam in 0.3 1 3; do echo "$lr $lam 42 $OUT/lr_sweep/lr${lr}_lam${lam}"; done; done
  for lr in 3e-5 1e-4;           do for lam in 0.1 0.03 0; do echo "$lr $lam 42 $OUT/lr_sweep/lr${lr}_lam${lam}"; done; done
  # second extension: 0.03 was again the lower edge (and still far better than 0), so keep
  # halving the decade down to where the BC is lost; one cross-check of lr at the small-lambda end
  for lam in 0.01 0.003 0.001;   do echo "1e-4 $lam 42 $OUT/lr_sweep/lr1e-4_lam${lam}"; done
  echo "3e-4 0.03 42 $OUT/lr_sweep/lr3e-4_lam0.03"
} | queue

echo "=== selecting on best validation rel-L2 ==="
read -r BEST_LR BEST_LAM < <(python - "$OUT/lr_sweep" <<'PY'
import glob, json, os, sys
root = sys.argv[1]; rows = []
for path in glob.glob(os.path.join(root, "lr*_lam*", "results", "*.json")):
    d = json.load(open(path)); cell = os.path.basename(os.path.dirname(os.path.dirname(path)))
    lr, lam = cell[2:].split("_lam"); rows.append((min(d["stats"]["val_rel_l2_errors"]), lr, lam))
rows.sort(); print(rows[0][1], rows[0][2]) if rows else print("1e-4 0.3")
PY
)
echo "selected lr=$BEST_LR lambda_bc=$BEST_LAM"

FINAL=$OUT/final/lr${BEST_LR}_lam${BEST_LAM}
echo "=== stage 2: seeds 43, 44 at the selected cell -> $FINAL ==="
if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$FINAL/seed42"; cp -rn "$OUT/lr_sweep/lr${BEST_LR}_lam${BEST_LAM}/." "$FINAL/seed42/"
fi
for seed in 43 44; do echo "$BEST_LR $BEST_LAM $seed $FINAL/seed$seed"; done | queue

echo "=== control: mollifier (reference BC), seeds 42-44 -> control_mollified/ ==="
for seed in 42 43 44; do echo "$MOLL_LR x $seed $OUT/control_mollified/seed$seed"; done \
  | xargs -P "$PAR" -L 1 bash -c 'run_cell "--pideeponet_bc mollifier" "$0" "$2" "$3"'
echo "=== all done: $(date) ==="
