"""Overlay validation relative-L2 vs epoch for the loss-comparison experiment.

Reads the per-run ``results/*.json`` written by ``Trainer._save_results_json``, classifies
each run into one of the known curve keys, and draws one curve per key. Which curves are
shown is configurable with per-curve flags (default: all present are shown):

    mse       supervised MSE (--loss data)
    l2        supervised true-L2 (--loss data_l2)
    h1        supervised H^1_0 (--loss data_h1)
    dr_penalty  Deep Ritz, penalty BC
    dr_t0       Deep Ritz, hard BC  (== preconditioned Deep Ritz at t=0)
    dr_t1       preconditioned Deep Ritz, blend t=1
    pls_t1      preconditioned least-squares, blend t=1
    pls_t0.5    preconditioned least-squares, blend t=0.5

Usage (after pulling the output back from the cluster):
    python experiments/poisson/loss_comparison/plot_loss_comparison.py \
        --results_dir output/poisson/loss_comparison/results            # all curves
    python experiments/poisson/loss_comparison/plot_loss_comparison.py \
        --results_dir output/poisson/loss_comparison/results --mse --h1 --pls_t1   # subset
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt

# curve keys, in stable legend / colour order
KEYS = ["mse", "l2", "h1", "dr_penalty", "dr_t0", "dr_t1", "pls_t1", "pls_t0.5"]

LABELS = {
    "mse": "MSE (data)",
    "l2": r"true $L^2$ (data)",
    "h1": r"$H^1_0$ (data)",
    "dr_penalty": "Deep Ritz (penalty BC)",
    "dr_t0": "Deep Ritz (hard BC, $t=0$)",
    "dr_t1": "Deep Ritz precond ($t=1$)",
    "pls_t1": "PLS ($t=1$)",
    "pls_t0.5": "PLS ($t=0.5$)",
}


def _is(t, value):
    return t == t and abs(t - value) < 1e-6      # NaN-safe float match


def classify(run):
    """Map a run's metadata to a curve key (or None if it is not one of the known curves)."""
    lt = run.get("loss_type")
    t = run.get("precond_strength", float("nan"))
    if lt == "data":
        return "mse"
    if lt == "data_l2":
        return "l2"
    if lt == "data_h1":
        return "h1"
    if lt == "pls":
        if _is(t, 1.0):
            return "pls_t1"
        if _is(t, 0.5):
            return "pls_t0.5"
        return None
    if lt == "deepritz":
        if run.get("precondition"):
            if _is(t, 1.0):
                return "dr_t1"
            if _is(t, 0.0):
                return "dr_t0"
            return None
        return "dr_t0" if run.get("bc_mode") == "hard" else "dr_penalty"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="output/poisson/loss_comparison/results")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write the figure (default: alongside results_dir).")
    for k in KEYS:
        ap.add_argument(f"--{k}", dest=k, action="store_true", help=f"show the '{k}' curve")
    args = ap.parse_args()

    selected = [k for k in KEYS if getattr(args, k)] or KEYS   # no flags -> show all

    by_key = {}
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        with open(path) as fh:
            run = json.load(fh)
        key = classify(run)
        if key is not None:
            by_key[key] = run
    if not by_key:
        raise SystemExit(f"No recognized runs found in {args.results_dir}")
    print(f"Found curves: {', '.join(k for k in KEYS if k in by_key)}")
    print(f"Showing: {', '.join(k for k in selected if k in by_key)}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for k in KEYS:
        if k not in selected or k not in by_key:
            continue
        rl2 = by_key[k]["stats"]["val_rel_l2_errors"]
        if not rl2:
            continue
        ax.plot(range(1, len(rl2) + 1), rl2, lw=1.6, label=LABELS[k])
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
