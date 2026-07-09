"""Overlay Allen–Cahn rollout-MSE convergence for four training losses.

Reads the per-run ``results/*.json`` written by ``RolloutTrainer._save_results_json`` and plots
per-epoch validation rollout-MSE (mean over batch, rollout steps, and grid — see
``RolloutTrainer._eval_loader``), one curve per training loss:

  1. data-driven                 (supervised trajectory MSE)
  2. Galerkin (backward-Euler)   (label-free ½‖R_be‖²)
  3. Galerkin (convex–concave)   (label-free ½‖R_cc‖²)
  4. minimizing-movement         (label-free convex–concave JKO objective J)

All four are trained on and evaluated against the *same* convex–concave reference dataset, so
the MSE numbers are directly comparable.

Usage (after pulling the output back from the cluster):
    python experiments/ac_loss_comparison/plot_ac_loss_comparison.py \
        --results_dir output/ac_loss_comparison/results   # optional: --smooth
"""

import argparse
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt

# Fixed arm order -> stable colours/legend across replots.
ARM_ORDER = ["data-driven", "Galerkin (backward-Euler)", "Galerkin (convex–concave)",
             "minimizing-movement"]


def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as fh:
            runs.append(json.load(fh))
    return runs


def arm_label(run):
    """Human-readable training-loss label from the run's config."""
    if run.get("loss_type") == "data" or (run.get("lambda_data") and not run.get("lambda_galerkin")):
        return "data-driven"
    if run.get("ac_loss_form") == "min_movement":
        return "minimizing-movement"
    pretty = {"backward_euler": "backward-Euler", "convex_concave": "convex–concave"}
    return f"Galerkin ({pretty.get(run.get('ac_integrator'), run.get('ac_integrator'))})"


def smooth_curve(y, window):
    """Centered moving average (numpy only), edge-padded to preserve length."""
    y = np.asarray(y, dtype=float)
    if window <= 1 or y.size < 2:
        return y
    w = min(int(window), y.size)
    if w % 2 == 0:
        w -= 1
    if w <= 1:
        return y
    pad = w // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(w) / w, mode="valid")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="output/ac_loss_comparison/results")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write the figure (default: alongside results_dir).")
    ap.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=False,
                    help="Smooth curves with a centered moving average (default off).")
    ap.add_argument("--smooth_window", type=int, default=15)
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No results found in {args.results_dir}")
    by_label = {arm_label(r): r for r in runs}
    print(f"Loaded {len(runs)} runs: {', '.join(by_label)}")

    ordered = [(lab, by_label[lab]) for lab in ARM_ORDER if lab in by_label]
    ordered += [(lab, r) for lab, r in by_label.items() if lab not in ARM_ORDER]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for lab, r in ordered:
        mse = r["stats"]["val_errors"]
        if not mse:
            continue
        x = range(1, len(mse) + 1)
        if args.smooth:
            base, = ax.plot(x, mse, lw=0.8, alpha=0.2)
            ax.plot(x, smooth_curve(mse, args.smooth_window), lw=1.8, label=lab,
                    color=base.get_color())
        else:
            ax.plot(x, mse, lw=1.6, label=lab)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation rollout MSE")
    ax.set_yscale("log")
    ax.set_title("Allen–Cahn: rollout-MSE convergence by training loss")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results_dir))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ac_loss_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"overlay -> {out_path}")


if __name__ == "__main__":
    main()
