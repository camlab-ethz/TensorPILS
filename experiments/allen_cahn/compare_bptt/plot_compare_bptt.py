"""Compare autodiff/BPTT modes for Allen-Cahn rollout training: one figure per loss.

Reads the per-run ``results/*.json`` and, for each loss (minimizing-movement, Galerkin-LS,
data-driven), plots the chosen metric vs reaction strength ``eps`` with one curve per BPTT mode
(full_bptt / detach_prev / pushforward). So each figure answers, for that loss, "which autodiff
mode is most robust as the problem stiffens?"

Metric (``--metric``): ``st_rel_l2`` (space-time relative FEM-L2, default), ``final_rel_l2``,
``mse`` -- best-model test scalars.

Usage (after pulling the output back from the cluster):
    python experiments/allen_cahn/compare_bptt/plot_compare_bptt.py \
        --results_dir output/allen_cahn/compare_bptt/results        # optional: --metric final_rel_l2
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")            # headless
import matplotlib.pyplot as plt

METRICS = {
    "st_rel_l2": ("test_st_rel_l2", "test space-time relative $L^2$"),
    "final_rel_l2": ("test_final_rel_l2", "test final-time relative $L^2$"),
    "mse": ("test_mse", "test rollout MSE"),
}
MODE_ORDER = ["full_bptt", "detach_prev", "pushforward"]
MODE_PRETTY = {"full_bptt": "full BPTT", "detach_prev": "detach prev (+BPTT)",
               "pushforward": "pushforward"}
# loss key -> (pretty title, filename slug)
LOSS_PRETTY = {"mm": ("minimizing-movement", "mm"),
               "ls": ("Galerkin (least-squares)", "galerkin"),
               "data": ("data-driven", "data")}


def load_runs(results_dir):
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as fh:
            out.append(json.load(fh))
    return out


def loss_key(run):
    if run.get("loss_type") == "data" or (run.get("lambda_data") and not run.get("lambda_galerkin")):
        return "data"
    return "mm" if run.get("ac_loss_form") == "min_movement" else "ls"


def get_eps(run):
    if run.get("eps") is not None:
        return float(run["eps"])
    m = re.search(r"_eps([0-9.]+)_", run.get("prefix", ""))
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="output/allen_cahn/compare_bptt/results")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--metric", choices=list(METRICS), default="st_rel_l2")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No results found in {args.results_dir}")
    field, ylabel = METRICS[args.metric]

    # loss -> mode -> {eps: value}
    data = {}
    for r in runs:
        eps, val, mode = get_eps(r), r.get(field), r.get("bptt_mode")
        if eps is None or val is None or mode is None:
            continue
        data.setdefault(loss_key(r), {}).setdefault(mode, {})[eps] = val

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results_dir))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loaded {len(runs)} runs; losses: {', '.join(data)}; metric={args.metric}")

    for lk, per_mode in data.items():
        title, slug = LOSS_PRETTY.get(lk, (lk, lk))
        fig, ax = plt.subplots(figsize=(7.5, 5))
        modes = [m for m in MODE_ORDER if m in per_mode] + [m for m in per_mode if m not in MODE_ORDER]
        for mode in modes:
            pts = sorted(per_mode[mode].items())
            ax.plot([e for e, _ in pts], [v for _, v in pts], "o-", lw=1.7, ms=5,
                    label=MODE_PRETTY.get(mode, mode))
        ax.set_yscale("log")
        ax.set_xlabel(r"reaction strength $\epsilon$")
        ax.set_ylabel(ylabel)
        ax.axvline(16, color="0.6", ls=":", lw=1)
        ax.set_title(f"BPTT modes for the {title} loss")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"compare_bptt_{slug}_{args.metric}.png")
        fig.savefig(path, dpi=150)
        print(f"  {title:26s} -> {path}")


if __name__ == "__main__":
    main()
