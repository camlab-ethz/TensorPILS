"""Long-rollout analysis (stage 2: plot). Reads metrics.json from compute_long_rollout.py and makes,
per eps, two figures:

  * energy vs rollout step -- 8 model curves + the convex-concave reference energy (dashed black);
  * error  vs rollout step -- 8 model curves (relative FEM-L2 vs the CC reference), log-y.

Colour encodes the loss (min-movement / Galerkin-LS / data-driven), line style the BPTT mode
(full_bptt / detach_prev / pushforward). Diverged rollouts stop where they went non-finite and are
marked with a red x (and flagged in the legend), so a blow-up is visible rather than dropped.

Usage:
    python experiments/long_rollout/plot_long_rollout.py --metrics output/long_rollout/metrics.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOSS_COLOR = {"mm": "#1f77b4", "ls": "#ff7f0e", "data": "#2ca02c"}
LOSS_NAME = {"mm": "min-movement", "ls": "Galerkin-LS", "data": "data-driven"}
MODE_STYLE = {"full_bptt": "-", "detach_prev": "--", "pushforward": ":"}
MODE_NAME = {"full_bptt": "full BPTT", "detach_prev": "detach-prev", "pushforward": "pushforward"}
ORDER = [("mm", "full_bptt"), ("mm", "detach_prev"), ("mm", "pushforward"),
         ("ls", "full_bptt"), ("ls", "detach_prev"), ("ls", "pushforward"),
         ("data", "full_bptt"), ("data", "pushforward")]


def nan_clean(vec):
    return [float("nan") if v is None else v for v in vec]


def curve(ax, run):
    y = nan_clean(run["energy"] if ax._is_energy else run["error"])
    x = list(range(len(y)))
    lab = f"{LOSS_NAME[run['loss']]} · {MODE_NAME.get(run['mode'], run['mode'])}"
    if run["diverged"]:
        lab += f"  (⊗{run['diverge_step']})"
    ax.plot(x, y, color=LOSS_COLOR.get(run["loss"], "gray"),
            ls=MODE_STYLE.get(run["mode"], "-"), lw=1.7, label=lab)
    if run["diverged"] and run["diverge_step"] > 0:
        j = run["diverge_step"] - 1
        ax.plot(j, y[j], "x", color="red", ms=8, mew=2, zorder=6)


def sort_runs(runs):
    idx = {k: i for i, k in enumerate(ORDER)}
    return sorted(runs, key=lambda r: idx.get((r["loss"], r["mode"]), 99))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", default="output/long_rollout/metrics.json")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    data = json.load(open(args.metrics))
    steps = data["steps"]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.metrics))
    os.makedirs(out_dir, exist_ok=True)

    for eps, blk in sorted(data["by_eps"].items(), key=lambda kv: float(kv[0])):
        runs = sort_runs(blk["runs"])

        # --- energy ---
        fig, ax = plt.subplots(figsize=(7.5, 5)); ax._is_energy = True
        for r in runs:
            curve(ax, r)
        ax.plot(range(len(blk["ref_energy"])), blk["ref_energy"], "k--", lw=2.2,
                label="convex-concave reference", zorder=8)
        e0 = blk["ref_energy"][0]
        ax.set_ylim(0, 2.5 * e0)                       # cap so a blow-up clips (its red x still shows)
        ax.axvline(10, color="0.7", ls=":", lw=1)      # training horizon
        ax.set_xlabel("rollout step"); ax.set_ylabel("Ginzburg-Landau energy")
        ax.set_title(f"Long rollout energy  (eps={eps}, {steps} steps)")
        ax.legend(fontsize=7.5, framealpha=0.9, ncol=2); ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(out_dir, f"long_rollout_eps{eps}_energy.png")
        fig.savefig(p, dpi=150); plt.close(fig); print(f"  {p}")

        # --- error ---
        fig, ax = plt.subplots(figsize=(7.5, 5)); ax._is_energy = False
        for r in runs:
            curve(ax, r)
        ax.set_yscale("log")
        ax.axvline(10, color="0.7", ls=":", lw=1)
        ax.text(10, ax.get_ylim()[1], " train horizon", color="0.5", va="top", ha="left", fontsize=8)
        ax.set_xlabel("rollout step"); ax.set_ylabel("relative $L^2$ vs convex-concave")
        ax.set_title(f"Long rollout error  (eps={eps}, {steps} steps)")
        ax.legend(fontsize=7.5, framealpha=0.9, ncol=2); ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        p = os.path.join(out_dir, f"long_rollout_eps{eps}_error.png")
        fig.savefig(p, dpi=150); plt.close(fig); print(f"  {p}")


if __name__ == "__main__":
    main()
