"""Aggregate a preconditioner-strength sweep into two figures.

Reads the per-run ``results/*.json`` written by ``Trainer._save_results_json`` and produces:

  1. overlay  — validation relative-L2 vs epoch, one curve per strength (the
                "physics-informed -> supervised transition" figure);
  2. collapse — final relative-L2 vs the conditioning number kappa(H_P).

Usage (after pulling the sweep output back from the cluster):
    python experiments/sweep_blend/plot_sweep.py \
        --results_dir output/sweep_blend/pls/results   # or .../deepritz/results
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt


def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as fh:
            runs.append(json.load(fh))
    # sort by preconditioner strength when available
    runs.sort(key=lambda r: (r.get("precond_strength") if r.get("precond_strength") == r.get("precond_strength") else -1.0))
    return runs


def plot_overlay(runs, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in runs:
        rl2 = r["stats"]["val_rel_l2_errors"]
        if not rl2:
            continue
        label = f"t={r['precond_strength']:.2f} (κ(H)={r['stats']['precond_cond_h']:.1e})"
        ax.plot(range(1, len(rl2) + 1), rl2, label=label, lw=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation relative $L^2$")
    ax.set_yscale("log")
    ax.set_title("PLS: convergence vs preconditioner strength $t$")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"overlay  -> {out_path}")


def plot_collapse(runs, out_path):
    kappa, final_rl2, strengths = [], [], []
    for r in runs:
        rl2 = r["stats"]["val_rel_l2_errors"]
        kh = r["stats"]["precond_cond_h"]
        if not rl2 or kh != kh:            # skip empty / NaN-conditioning runs
            continue
        kappa.append(kh)
        final_rl2.append(min(rl2))          # best achieved
        strengths.append(r["precond_strength"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(kappa, final_rl2, "o-", lw=1.6)
    for k, e, t in zip(kappa, final_rl2, strengths):
        ax.annotate(f"t={t:.2f}", (k, e), fontsize=8,
                    textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel(r"conditioning $\kappa(H_P)$")
    ax.set_ylabel("best validation relative $L^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Collapse: final error vs conditioning")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"collapse -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="output/sweep_blend/pls/results")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write the figures (default: alongside results_dir).")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No results found in {args.results_dir}")
    print(f"Loaded {len(runs)} runs.")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results_dir))
    os.makedirs(out_dir, exist_ok=True)
    plot_overlay(runs, os.path.join(out_dir, "sweep_overlay.png"))
    plot_collapse(runs, os.path.join(out_dir, "sweep_collapse.png"))


if __name__ == "__main__":
    main()
