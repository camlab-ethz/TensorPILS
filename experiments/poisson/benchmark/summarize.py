"""Tables of the Poisson benchmark, from the results JSONs of ``lr_sweep.sh`` and ``run.sh``.

Prints

1. the learning-rate selection: the best validation relative L2 of every method at every rate
   (seed 42, 500 epochs), and the rate each method selects;
2. the same for PI-DeepONet over learning rate x boundary weight;
3. the Poisson rows of Table 1: test relative L2 at the best-validation checkpoint, mean +/-
   half-range over seeds 42/43/44, and seconds per epoch.

    python experiments/poisson/benchmark/summarize.py [--root output/poisson/benchmark]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

# (key, row label, results-file prefix)
ARMS = [
    ("pls",        "L_PLS",       "fno_pls_"),
    ("data",       "L_data",      "fno_data_"),
    ("galerkin",   "L_LS",        "fno_galerkin_"),
    ("pino",       "PINO",        "fno_pino"),
    ("pideeponet", "PI-DeepONet", "deeponet_pi-"),
]


def records(pattern):
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            rec = json.load(fh)
        rec["_path"] = path
        out.append(rec)
    return out


def arm_of(rec):
    return next((k for k, _, pre in ARMS if rec["prefix"].startswith(pre)), None)


def best_val(rec):
    return float(np.nanmin(rec["stats"]["val_rel_l2_errors"]))


def lr_table(root):
    runs = {}
    for rec in records(os.path.join(root, "lr_sweep", "lr*", "results", "*.json")):
        lr = re.search(r"lr_sweep[/\\]lr([^/\\]+)[/\\]", rec["_path"]).group(1)
        runs.setdefault(arm_of(rec), {})[lr] = best_val(rec)
    if not runs:
        print(f"learning-rate sweep: no results under {root}/lr_sweep")
        return
    lrs = sorted({lr for d in runs.values() for lr in d}, key=float)
    print("learning-rate sweep, seed 42: best validation relative L2 [%]")
    print(f"  {'method':<12}" + "".join(f"{lr:>9}" for lr in lrs) + "   selected")
    for key, label, _ in ARMS:
        if key not in runs:
            continue
        row = runs[key]
        pick = min(row, key=row.get)
        print(f"  {label:<12}" + "".join(f"{100 * row[lr]:9.2f}" if lr in row else f"{'':>9}"
                                         for lr in lrs) + f"   {pick}")


def pideeponet_table(root):
    cells = {}
    for rec in records(os.path.join(root, "lr_sweep_pideeponet", "*", "results", "*.json")):
        m = re.search(r"lr([^_/\\]+)_lam([^/\\]+)[/\\]results", rec["_path"])
        cells[(m.group(1), m.group(2))] = best_val(rec)
    if not cells:
        print(f"\nPI-DeepONet sweep: no results under {root}/lr_sweep_pideeponet")
        return
    lrs = sorted({lr for lr, _ in cells}, key=float)
    lams = sorted({lam for _, lam in cells}, key=float)
    print("\nPI-DeepONet sweep, seed 42: best validation relative L2 [%]  (rows: boundary weight)")
    print(f"  {'lambda_bc':<10}" + "".join(f"{lr:>9}" for lr in lrs))
    for lam in lams:
        print(f"  {lam:<10}" + "".join(f"{100 * cells[(lr, lam)]:9.2f}" if (lr, lam) in cells
                                       else f"{'':>9}" for lr in lrs))
    lr, lam = min(cells, key=cells.get)
    print(f"  selected: lr {lr}, lambda_bc {lam}")


def final_table(root):
    by_arm = {}
    for rec in records(os.path.join(root, "final", "seed*", "results", "*.json")):
        seed = int(re.search(r"seed(\d+)", rec["_path"]).group(1))
        by_arm.setdefault(arm_of(rec), {})[seed] = rec
    if not by_arm:
        print(f"\nTable 1: no results under {root}/final")
        return
    print("\nTable 1, Poisson: test relative L2 [%], mean +/- half-range over seeds")
    print(f"  {'method':<12} {'seeds':<10} {'test [%]':>16} {'s/epoch':>8}")
    for key, label, _ in ARMS:
        runs = by_arm.get(key)
        if not runs:
            continue
        rs = [runs[s] for s in sorted(runs)]
        v = 100.0 * np.array([r["test_rl2"] for r in rs])
        t = [np.mean(r["stats"]["epoch_times"]) for r in rs if r["stats"].get("epoch_times")]
        print(f"  {label:<12} {','.join(map(str, sorted(runs))):<10} "
              f"{v.mean():7.2f} +/- {0.5 * (v.max() - v.min()):5.2f} "
              f"{np.mean(t) if t else float('nan'):8.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/poisson/benchmark")
    args = ap.parse_args()
    lr_table(args.root)
    pideeponet_table(args.root)
    final_table(args.root)


if __name__ == "__main__":
    main()
