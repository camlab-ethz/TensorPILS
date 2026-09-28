"""The backbone table of the appendix: FNO against GAOT on structured Poisson (K = 4, 64 x 64).

Prints GAOT's learning-rate selection (``lr_sweep.sh``: seed 42, 150 epochs, best validation
relative L2) and the table itself (``run.sh``): test relative L2 at the best-validation checkpoint,
mean +/- half-range over seeds 42/43/44, per loss and architecture.

    python experiments/poisson/backbone/summarize.py [--root output/poisson/backbone]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

# (row label, results-file stem after the architecture tag)
LOSSES = [
    ("L_data (supervised)",     "data_K4_"),
    ("L_PLS (preconditioned)",  "pls_mg-"),
    ("L_LS (bare residual)",    "galerkin_K4_"),
]
ARCHES = ["fno", "gaot"]


def records(pattern):
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            rec = json.load(fh)
        rec["_path"] = path
        out.append(rec)
    return out


def lr_table(root):
    recs = records(os.path.join(root, "lr_sweep", "lr*", "results", "gaot_*.json"))
    if not recs:
        print(f"GAOT learning-rate sweep: no results under {root}/lr_sweep")
        return
    print("GAOT learning-rate sweep, seed 42, 150 epochs: best validation relative L2 [%]")
    for label, stem in LOSSES:
        rows = []
        for rec in recs:
            if os.path.basename(rec["_path"]).startswith("gaot_" + stem):
                lr = re.search(r"lr_sweep[/\\]lr([^/\\]+)[/\\]", rec["_path"]).group(1)
                rows.append((float(lr), lr, 100 * np.nanmin(rec["stats"]["val_rel_l2_errors"])))
        if rows:
            pick = min(rows, key=lambda r: r[2])[1]
            cells = "  ".join(f"{lr}: {v:.3f}" for _, lr, v in sorted(rows))
            print(f"  {label:<24} {cells}   -> {pick}")


def final_table(root):
    print("\nbackbone table: test relative L2 [%], mean +/- half-range over seeds")
    print(f"  {'loss':<24}" + "".join(f"{a.upper():>21}" for a in ARCHES))
    for label, stem in LOSSES:
        cells = []
        for arch in ARCHES:
            recs = records(os.path.join(root, "seed*", "results", f"{arch}_{stem}*.json"))
            if recs:
                e = 100 * np.array([r["test_rl2"] for r in recs])
                cells.append(f"{e.mean():.2f} +/- {0.5 * (e.max() - e.min()):.2f} ({len(e)})")
            else:
                cells.append("--")
        print(f"  {label:<24}" + "".join(f"{c:>21}" for c in cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/poisson/backbone")
    args = ap.parse_args()
    lr_table(args.root)
    final_table(args.root)


if __name__ == "__main__":
    main()
