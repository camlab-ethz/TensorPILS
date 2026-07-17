"""Overlay Allen–Cahn rollout convergence for four training losses, and rollout error growth.

Reads the per-run ``results/*.json`` written by ``RolloutTrainer._save_results_json`` and makes:

  1. ``ac_loss_comparison.png``  — per-epoch validation metric vs epoch, one curve per loss;
  2. ``ac_error_growth.png``     — test-set relative FEM-L2 vs rollout step (error accumulation
                                    of the best model), one curve per loss (if L2 fields present).

Metrics (``--metric``): ``mse`` (grid rollout MSE), ``st_rel_l2`` (space-time relative FEM-L2),
``final_rel_l2`` (final-time relative FEM-L2). Default ``auto`` prefers the space-time L2 when
present (older runs without it fall back to MSE). All four arms train on and evaluate against the
same convex–concave reference, so the numbers are directly comparable.

The four training losses:
  1. data-driven                 (supervised trajectory MSE)
  2. Galerkin (backward-Euler)   (label-free ½‖R_be‖²)
  3. Galerkin (convex–concave)   (label-free ½‖R_cc‖²)
  4. minimizing-movement         (label-free convex–concave JKO objective J)

Usage:
    python experiments/allen_cahn/ac_loss_comparison/plot_ac_loss_comparison.py \
        --results_dir output/allen_cahn/ac_loss_comparison/results   # optional: --metric st_rel_l2 --smooth
"""

import argparse
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt

ARM_ORDER = ["data-driven", "Galerkin (backward-Euler)", "Galerkin (convex–concave)",
             "minimizing-movement"]

# metric key -> (stats field, y-axis label)
METRICS = {
    "mse": ("val_errors", "validation rollout MSE"),
    "st_rel_l2": ("val_st_rel_l2", "validation space-time relative $L^2$"),
    "final_rel_l2": ("val_final_rel_l2", "validation final-time relative $L^2$"),
}


def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as fh:
            runs.append(json.load(fh))
    return runs


def arm_label(run):
    if run.get("loss_type") == "data" or (run.get("lambda_data") and not run.get("lambda_galerkin")):
        return "data-driven"
    if run.get("ac_loss_form") == "min_movement":
        return "minimizing-movement"
    pretty = {"backward_euler": "backward-Euler", "convex_concave": "convex–concave"}
    return f"Galerkin ({pretty.get(run.get('ac_integrator'), run.get('ac_integrator'))})"


def smooth_curve(y, window):
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


def ordered_arms(runs):
    by_label = {arm_label(r): r for r in runs}
    out = [(lab, by_label[lab]) for lab in ARM_ORDER if lab in by_label]
    out += [(lab, r) for lab, r in by_label.items() if lab not in ARM_ORDER]
    return out


def plot_convergence(ordered, metric, out_path, smooth, window):
    field, ylabel = METRICS[metric]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for lab, r in ordered:
        y = r["stats"].get(field) or []
        if not y:
            continue
        x = range(1, len(y) + 1)
        if smooth:
            base, = ax.plot(x, y, lw=0.8, alpha=0.2)
            ax.plot(x, smooth_curve(y, window), lw=1.8, label=lab, color=base.get_color())
        else:
            ax.plot(x, y, lw=1.6, label=lab)
    ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.set_yscale("log")
    ax.set_title(f"Allen–Cahn: {metric} convergence by training loss")
    ax.legend(fontsize=9, framealpha=0.9); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"convergence -> {out_path}")


def plot_error_growth(ordered, out_path):
    """Test-set relative FEM-L2 vs rollout step (best model) — how error accumulates in time."""
    have = [(lab, r["test_rel_l2_steps"]) for lab, r in ordered if r.get("test_rel_l2_steps")]
    if not have:
        print("error-growth: no test_rel_l2_steps in results (older runs) — skipped")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for lab, steps in have:
        ax.plot(range(1, len(steps) + 1), steps, "o-", lw=1.6, ms=4, label=lab)
    ax.set_xlabel("rollout step $k$"); ax.set_ylabel("test relative $L^2$ at step $k$")
    ax.set_yscale("log")
    ax.set_title("Allen–Cahn: rollout error growth (best model)")
    ax.legend(fontsize=9, framealpha=0.9); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"error-growth -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="output/allen_cahn/ac_loss_comparison/results")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--metric", choices=["auto", "mse", "st_rel_l2", "final_rel_l2"], default="auto")
    ap.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--smooth_window", type=int, default=15)
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No results found in {args.results_dir}")
    ordered = ordered_arms(runs)
    print(f"Loaded {len(runs)} runs: {', '.join(lab for lab, _ in ordered)}")

    metric = args.metric
    if metric == "auto":
        metric = "st_rel_l2" if (ordered and ordered[0][1]["stats"].get("val_st_rel_l2")) else "mse"
    print(f"Metric: {metric}")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results_dir))
    os.makedirs(out_dir, exist_ok=True)
    plot_convergence(ordered, metric, os.path.join(out_dir, "ac_loss_comparison.png"),
                     args.smooth, args.smooth_window)
    plot_error_growth(ordered, os.path.join(out_dir, "ac_error_growth.png"))


if __name__ == "__main__":
    main()
