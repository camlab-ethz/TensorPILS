"""Plot the OOD generalization experiment: one figure per eval-K, four t-curves each.

Reads the per-run ``results/*.json`` written by ``Trainer._save_results_json``. Each figure
overlays the relative-L2-vs-epoch curves for every preconditioner strength ``t``:

  * in-distribution K=4  -> from ``stats.val_rel_l2_errors``;
  * out-of-distribution  -> from ``stats.ood_rel_l2["K6"]``, ``["K8"]``, ...

Usage (after pulling the sweep output back from the cluster):
    python experiments/generalization/plot_generalization.py \
        --results_dir output/generalization/pls_k4/results   # or .../pls_k16/results
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt


def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as fh:
            runs.append(json.load(fh))
    runs.sort(key=lambda r: r.get("precond_strength", 0.0))
    return runs


def curve_for(run, target_k):
    """Relative-L2-vs-epoch list for eval target ``target_k`` (the training K uses val)."""
    if target_k == run["K"]:
        return run["stats"]["val_rel_l2_errors"]
    return run["stats"].get("ood_rel_l2", {}).get(f"K{target_k}", [])


def plot_for_k(runs, target_k, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    train_k = runs[0]["K"]
    for r in runs:
        rl2 = curve_for(r, target_k)
        if not rl2:
            continue
        ax.plot(range(1, len(rl2) + 1), rl2, lw=1.6, label=f"t={r['precond_strength']:.2f}")
    tag = "in-distribution" if target_k == train_k else "out-of-distribution"
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative $L^2$")
    ax.set_yscale("log")
    ax.set_title(f"Eval on K={target_k} ({tag}) — trained on K={train_k}")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"K={target_k} -> {out_path}")


def eval_ks(runs):
    """The training K plus every OOD K present, sorted."""
    ks = {runs[0]["K"]}
    for r in runs:
        for key in r["stats"].get("ood_rel_l2", {}):
            ks.add(int(key.lstrip("K")))
    return sorted(ks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="output/generalization/pls_k4/results")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write the figures (default: alongside results_dir).")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No results found in {args.results_dir}")
    print(f"Loaded {len(runs)} runs.")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results_dir))
    os.makedirs(out_dir, exist_ok=True)
    for k in eval_ks(runs):
        plot_for_k(runs, k, os.path.join(out_dir, f"generalization_K{k}.png"))


if __name__ == "__main__":
    main()
