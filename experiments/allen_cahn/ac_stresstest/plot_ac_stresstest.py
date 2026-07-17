"""Allen-Cahn loss stress-test: test error vs reaction strength eps, one curve per training loss.

Reads the per-run ``results/*.json`` (best-model test metrics written by
``RolloutTrainer._save_results_json``) and plots the chosen metric against ``eps`` for each of the
three losses, so you can see how each degrades as the problem stiffens:

  1. data-driven                 (supervised trajectory MSE)
  2. Galerkin (convex-concave)   (label-free least-squares residual)
  3. minimizing-movement         (label-free convex-concave JKO objective; the "correct" loss)

Story: physics-informed learning with the *correct* physics loss (minimizing-movement) stays
accurate as the reaction stiffens, while the data-driven and least-squares losses fall apart.

Metric (``--metric``): ``st_rel_l2`` (space-time relative FEM-L2, default), ``final_rel_l2``, or
``mse`` -- all best-model test-set scalars. The reference data is convex-concave for every eps, so
these are comparable across losses at a fixed eps. Use ``--eps_min/--eps_max`` to restrict the
range (e.g. drop reusable-but-off-grid eps values from an earlier sweep).

Usage (after pulling the output back from the cluster):
    python experiments/allen_cahn/ac_stresstest/plot_ac_stresstest.py \
        --results_dir output/allen_cahn/ac_stresstest/results       # optional: --metric final_rel_l2
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt

ARM_ORDER = ["data-driven", "Galerkin (backward-Euler)", "Galerkin (convex-concave)",
             "minimizing-movement"]
METRICS = {
    "st_rel_l2": ("test_st_rel_l2", "test space-time relative $L^2$"),
    "final_rel_l2": ("test_final_rel_l2", "test final-time relative $L^2$"),
    "mse": ("test_mse", "test rollout MSE"),
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
    pretty = {"backward_euler": "backward-Euler", "convex_concave": "convex-concave"}
    return f"Galerkin ({pretty.get(run.get('ac_integrator'), run.get('ac_integrator'))})"


def get_eps(run):
    if run.get("eps") is not None:
        return float(run["eps"])
    m = re.search(r"_eps([0-9.]+)_", run.get("prefix", ""))   # fallback: parse from the filename tag
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="output/allen_cahn/ac_stresstest/results")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--metric", choices=list(METRICS), default="st_rel_l2")
    ap.add_argument("--eps_min", type=float, default=None, help="drop runs with eps below this")
    ap.add_argument("--eps_max", type=float, default=None, help="drop runs with eps above this")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No results found in {args.results_dir}")
    field, ylabel = METRICS[args.metric]

    # arm -> {eps: metric}
    series = {}
    for r in runs:
        eps = get_eps(r)
        val = r.get(field)
        if eps is None or val is None:
            continue
        if (args.eps_min is not None and eps < args.eps_min) or \
           (args.eps_max is not None and eps > args.eps_max):
            continue
        series.setdefault(arm_label(r), {})[eps] = val

    ordered = [a for a in ARM_ORDER if a in series] + [a for a in series if a not in ARM_ORDER]
    print(f"Loaded {len(runs)} runs across {len(series)} losses; metric={args.metric}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in ordered:
        pts = sorted(series[arm].items())                    # [(eps, val), ...]
        xs = [e for e, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, "o-", lw=1.7, ms=5, label=arm)
    ax.set_yscale("log")                              # errors span orders; eps axis stays linear
    ax.set_xlabel(r"reaction strength $\epsilon$   (reaction coefficient $\epsilon^2$)")
    ax.set_ylabel(ylabel)
    ax.axvline(16, color="0.6", ls=":", lw=1)
    ax.text(16, ax.get_ylim()[1], " resolution limit", color="0.5",
            va="top", ha="left", fontsize=8)
    ax.set_title("Physics-informed learning helps for Allen-Cahn")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results_dir))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ac_stresstest_{args.metric}.png")
    fig.savefig(out_path, dpi=150)
    print(f"stress-test -> {out_path}")


if __name__ == "__main__":
    main()
