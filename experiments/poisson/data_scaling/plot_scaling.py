"""Aggregate the data-scaling sweep into the "test error vs dataset size" figure.

Reads the per-run ``results/*.json`` written by ``PoissonTrainer._save_results_json`` and
plots the final test relative-L2 against ``n_train`` (log axis). The streaming run — the
infinite-data limit, in which no sample is ever seen twice — has no finite abscissa and is
drawn as a dashed horizontal reference line (the floor the finite-``n`` curve saturates at;
see ``notes/preconditioner_notes/infinite_data_limit.tex``). When multiple seeds per ``n``
are present they are aggregated to mean with a min/max band.

Usage (after pulling the sweep output back from the cluster):
    python experiments/poisson/data_scaling/plot_scaling.py \
        --results_dir output/poisson/data_scaling/results
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt


def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as fh:
            runs.append(json.load(fh))
    return runs


def run_metric(run, metric):
    if metric == "best_val":
        rl2 = run["stats"]["val_rel_l2_errors"]
        return min(rl2) if rl2 else float("nan")
    return run.get("test_rl2", float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="output/poisson/data_scaling/results")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write the figure (default: alongside results_dir).")
    ap.add_argument("--metric", choices=["test_rl2", "best_val"], default="test_rl2",
                    help="y-axis: final test relative-L2 (default; the honest number) or the "
                         "best validation relative-L2 (diagnostic).")
    ap.add_argument("--figsize", type=float, nargs=2, default=[7.0, 5.0], metavar=("W", "H"),
                    help="Figure size in inches (default: 7 5). A squat aspect (e.g. 6 2.8) is "
                         "what fits a LaTeX wrapfigure, which can only wrap as many lines as the "
                         "adjacent paragraph provides.")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No results found in {args.results_dir}")
    print(f"Loaded {len(runs)} runs.")

    finite, stream_vals, stopped = defaultdict(list), [], {}
    for r in runs:
        y = run_metric(r, args.metric)
        if r.get("stream"):
            stream_vals.append(y)
        elif r.get("n_train"):
            finite[r["n_train"]].append(y)
            if r["stats"].get("stopped_epoch", -1) >= 0:
                stopped[r["n_train"]] = r["stats"]["stopped_epoch"]

    ns = sorted(finite)
    mean = np.array([np.mean(finite[n]) for n in ns])
    lo = np.array([np.min(finite[n]) for n in ns])
    hi = np.array([np.max(finite[n]) for n in ns])
    multi_seed = any(len(v) > 1 for v in finite.values())

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    ax.plot(ns, mean, "o-", lw=1.8, label="finite dataset")
    if multi_seed:
        ax.fill_between(ns, lo, hi, alpha=0.2)
    else:
        # Single seed: mark runs that early-stopped (they used less than the full budget).
        for n, ep in stopped.items():
            ax.annotate(f"stop@{ep}", (n, np.mean(finite[n])), fontsize=7,
                        textcoords="offset points", xytext=(4, -10))
    if stream_vals:
        y = float(np.mean(stream_vals))
        ax.axhline(y, color="black", ls="--", lw=1.8,
                   label=f"streaming (infinite data): {y:.2%}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("training-set size $n$")
    ylabel = ("test relative $L^2$" if args.metric == "test_rl2"
              else "best validation relative $L^2$")
    ax.set_ylabel(ylabel)
    ax.set_title("Fixed optimization budget: error vs dataset size")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results_dir))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"scaling_{args.metric}.png")
    # Raster for quick viewing and the working notes; vector for the paper. Note that
    # *.png is git-ignored repo-wide, so only the .pdf can be committed under paper/figures/.
    fig.savefig(out_path, dpi=150)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path)
    print(f"scaling -> {out_path}\n        -> {pdf_path}")


if __name__ == "__main__":
    main()
