"""Aggregate the PI-DeepONet zero-BC study and print the Table-1 row.

Reads ``<root>/lr_sweep/lr<lr>_lam<lambda>/results/*.json`` (stage 1),
``<root>/final/lr<lr>_lam<lambda>/seed*/`` (stage 2, one config-named folder per selected cell; the
one matching the stage-1 selection is the table row) and ``<root>/control_mollified/seed*/`` (the
reference's mollifier, the labelled control), and prints

* the stage-1 ranking by best VALIDATION relative L2 (the selection criterion; test is shown
  for information only and played no part in the choice);
* per-seed and aggregated numbers for the two stage-2 arms, in the paper's convention: test
  relative L2 at the best-validation checkpoint, mean +/- half-range over the seeds;
* the LaTeX cell to paste into ``tab:summary``.

Usage:
    python experiments/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/summarize.py \
        [--root output/poisson/poisson_paper/poisson_benchmark/pideeponet_zero]
"""

import argparse
import glob
import json
import os


def _load(path):
    d = json.load(open(path))
    v = d["stats"]["val_rel_l2_errors"]
    return dict(best_val=min(v), best_epoch=int(d["stats"].get("best_epoch", -1)),
                test=float(d["test_rl2"]), n_epochs=len(v))


def stage1(root):
    rows = []
    for path in glob.glob(os.path.join(root, "lr_sweep", "lr*_lam*", "results", "*.json")):
        cell = os.path.basename(os.path.dirname(os.path.dirname(path)))
        lr, lam = cell[2:].split("_lam")
        rows.append((lr, lam, _load(path)))
    rows.sort(key=lambda r: r[2]["best_val"])
    print("stage 1 -- seed 42, 500 epochs, ranked by best validation rel-L2 (selection criterion):")
    print(f"  {'lr':>6} {'lambda_bc':>9} {'best-val':>9} {'@epoch':>7} {'test':>8}")
    for lr, lam, r in rows:
        print(f"  {lr:>6} {lam:>9} {100*r['best_val']:8.2f}% {r['best_epoch']:>7} {100*r['test']:7.2f}%")
    if rows:
        print(f"  selected: lr={rows[0][0]}  lambda_bc={rows[0][1]}")
    return rows


def stage2(root, sub, label):
    runs = []
    for path in sorted(glob.glob(os.path.join(root, sub, "seed*", "results", "*.json"))):
        seed = os.path.basename(os.path.dirname(os.path.dirname(path)))
        runs.append((seed, _load(path)))
    print(f"\n{label}  ({sub}/):")
    if not runs:
        print("  (no runs)"); return None
    for seed, r in runs:
        print(f"  {seed}: test {100*r['test']:6.2f}%   best-val {100*r['best_val']:6.2f}% @ epoch {r['best_epoch']}")
    t = [r["test"] for _, r in runs]; b = [r["best_val"] for _, r in runs]
    mean, half = sum(t) / len(t), 0.5 * (max(t) - min(t))
    bmean, bhalf = sum(b) / len(b), 0.5 * (max(b) - min(b))
    print(f"  -> test rel-L2 {mean:.3f} +/- {half:.3f} (half-range over {len(t)} seeds);"
          f"  best-val {bmean:.3f} +/- {bhalf:.3f}")
    if len(t) < 3:
        print("  WARNING: fewer than 3 seeds")
    still = [s for s, r in runs if r["best_epoch"] >= r["n_epochs"] - 5]
    if still:
        print(f"  note: best epoch within 5 of the budget for {', '.join(still)} -- still descending, budget-limited")
    return mean, half


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/poisson/poisson_paper/poisson_benchmark/pideeponet_zero")
    args = ap.parse_args()
    rows = stage1(args.root)
    selected = f"lr{rows[0][0]}_lam{rows[0][1]}" if rows else None
    zero = None
    for cfg in sorted(glob.glob(os.path.join(args.root, "final", "lr*_lam*"))):
        name = os.path.basename(cfg)
        tag = "  <-- selected on validation (the table row)" if name == selected else "  (not selected)"
        res = stage2(args.root, os.path.join("final", name), f"stage 2 -- PI-DeepONet, zero-BC, {name}{tag}")
        if name == selected:
            zero = res
    moll = stage2(args.root, "control_mollified", "control -- PI-DeepONet, sin(pi x)sin(pi y) mollifier")
    if zero:
        print(f"\nLaTeX (tab:summary, Poisson / PI-DeepONet, {selected}):  ${zero[0]:.3f} \\pm {zero[1]:.3f}$")
    elif selected:
        print(f"\nno stage-2 runs yet for the selected cell {selected}")
    if moll:
        print(f"LaTeX (mollified control):                   ${moll[0]:.3f} \\pm {moll[1]:.3f}$")


if __name__ == "__main__":
    main()
