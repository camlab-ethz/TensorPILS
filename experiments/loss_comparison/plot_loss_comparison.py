"""Overlay validation relative-L2 vs epoch for the loss-comparison experiment.

Reads the per-run ``results/*.json`` written by ``Trainer._save_results_json`` and draws one
curve per loss (identified by each run's ``loss_type``).

Usage (after pulling the output back from the cluster):
    python experiments/loss_comparison/plot_loss_comparison.py \
        --results_dir output/loss_comparison/results
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt

DATA_LABELS = {
    "data": "MSE (data)",
    "data_l2": r"true $L^2$ (data)",
    "data_h1": r"$H^1_0$ (data)",
}

# order for a stable legend / colour assignment
ORDER = ["data", "data_l2", "data_h1", "deepritz-penalty", "deepritz-hard"]


def label_for(run):
    lt = run.get("loss_type", "?")
    if lt == "deepritz":
        return f"Deep Ritz ({run.get('bc_mode', 'penalty')} BC)"
    return DATA_LABELS.get(lt, lt)


def sort_key(run):
    lt = run.get("loss_type", "?")
    key = f"deepritz-{run.get('bc_mode', 'penalty')}" if lt == "deepritz" else lt
    return ORDER.index(key) if key in ORDER else len(ORDER)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="output/loss_comparison/results")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write the figure (default: alongside results_dir).")
    args = ap.parse_args()

    runs = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        with open(path) as fh:
            runs.append(json.load(fh))
    if not runs:
        raise SystemExit(f"No results found in {args.results_dir}")
    print(f"Loaded {len(runs)} runs.")

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in sorted(runs, key=sort_key):
        rl2 = r["stats"]["val_rel_l2_errors"]
        if not rl2:
            continue
        ax.plot(range(1, len(rl2) + 1), rl2, lw=1.6, label=label_for(r))
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation relative $L^2$")
    ax.set_yscale("log")
    ax.set_title("Loss comparison: validation relative $L^2$")
    ax.legend(framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results_dir))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "loss_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"loss_comparison -> {out_path}")


if __name__ == "__main__":
    main()
