"""Tables of the Stokes experiment, from the results JSONs of ``run.sh`` and ``grid_search.sh``.

Prints

1. the Stokes rows of Table 1: test relative FE-L2 of velocity and pressure at the
   best-validation checkpoint, mean +/- half-range over seeds 42/43/44, and seconds per epoch;
2. the PI-DeepONet run at 1024 random collocation points per step (Baseline paragraph);
3. the grid search on seed 42, with the configuration each method selects on validation;
4. the preconditioner ablation table: the unpreconditioned L_LS against L_PLS for every Schur
   weight omega_S, test errors on seed 42.

    python experiments/stokes/benchmark/summarize.py [--root output/stokes/benchmark]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

# (key, row label, results-file prefix)
ARMS = [
    ("pls",        "L_PLS",       "gaot_stokes_pls_"),
    ("data",       "L_data",      "gaot_stokes_data_"),
    ("galerkin",   "L_LS",        "gaot_stokes_galerkin_"),
    ("pideeponet", "PI-DeepONet", "deeponet_stokes_pi-"),
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


def seconds_per_epoch(rec):
    t = rec["stats"].get("epoch_times") or []
    return float(np.mean(t)) if t else float("nan")


def cell(values):
    v = 100.0 * np.asarray(values, dtype=float)
    return f"{v.mean():6.2f} +/- {0.5 * (v.max() - v.min()):4.2f}"


def seed_table(root, title):
    by_arm = {}
    for rec in records(os.path.join(root, "seed*", "results", "*.json")):
        seed = int(re.search(r"seed(\d+)", rec["_path"]).group(1))
        by_arm.setdefault(arm_of(rec), {})[seed] = rec
    if not by_arm:
        print(f"\n{title}: no results under {root}")
        return
    print(f"\n{title}  ({root})")
    print(f"  {'method':<12} {'seeds':<10} {'velocity [%]':>16} {'pressure [%]':>16} {'s/epoch':>8}")
    for key, label, _ in ARMS:
        runs = by_arm.get(key)
        if not runs:
            continue
        rs = [runs[s] for s in sorted(runs)]
        print(f"  {label:<12} {','.join(map(str, sorted(runs))):<10} "
              f"{cell([r['test_rel_l2_u'] for r in rs]):>16} "
              f"{cell([r['test_rel_l2_p'] for r in rs]):>16} "
              f"{np.mean([seconds_per_epoch(r) for r in rs]):8.2f}")


def grid_table(root):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "grid_search", "*"))):
        recs = records(os.path.join(d, "results", "*.json"))
        if recs:
            runs.append((os.path.basename(d), recs[0]))
    if not runs:
        print(f"\ngrid search: no results under {root}/grid_search")
        return {}
    print("\ngrid search, seed 42 (selection: lowest best validation mean of the two errors)")
    print(f"  {'configuration':<26} {'best val [%]':>12} {'test u [%]':>10} {'test p [%]':>10}")
    best = {}
    for name, rec in runs:
        group = name.split("_")[0]
        if rec["best_val_error"] < best.get(group, (np.inf,))[0]:
            best[group] = (rec["best_val_error"], name)
    for name, rec in runs:
        mark = "  <- selected" if best[name.split("_")[0]][1] == name else ""
        print(f"  {name:<26} {100 * rec['best_val_error']:12.2f} {100 * rec['test_rel_l2_u']:10.2f} "
              f"{100 * rec['test_rel_l2_p']:10.2f}{mark}")
    return dict(runs)


def ablation_table(root, grid):
    ls = [r for r in records(os.path.join(root, "final", "seed42", "results", "*.json"))
          if arm_of(r) == "galerkin"]
    pls = sorted(((float(n[len("pls_omega"):]), r) for n, r in grid.items()
                  if n.startswith("pls_omega")), key=lambda t: t[0])
    if not (ls and pls):
        return
    cols = [("no P", ls[0])] + [(f"{w:g}", r) for w, r in pls]
    print("\npreconditioner ablation, test errors on seed 42 (L_LS, then L_PLS by omega_S)")
    print("  " + "".join(f"{c:>9}" for c, _ in [("", None)] + cols))
    print("  " + "".join(f"{v:>9}" for v in ["u [%]"] + [f"{100 * r['test_rel_l2_u']:.2f}"
                                                          for _, r in cols]))
    print("  " + "".join(f"{v:>9}" for v in ["p [%]"] + [f"{100 * r['test_rel_l2_p']:.2f}"
                                                          for _, r in cols]))
    print("  (the condition-number row comes from conditioning.py)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/stokes/benchmark")
    args = ap.parse_args()
    seed_table(os.path.join(args.root, "final"), "Table 1, Stokes")
    seed_table(os.path.join(args.root, "colloc1024"), "PI-DeepONet, 1024 collocation points")
    grid = grid_table(args.root)
    ablation_table(args.root, grid)


if __name__ == "__main__":
    main()
