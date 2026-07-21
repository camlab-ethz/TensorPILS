"""realistic_ac rollout analysis (stage 2: plot). Reads metrics.json from compute_rollout.py and
makes two figures, each overlaying the **test (solid) and train (dashed)** splits so the
generalization gap is visible at a glance:

  * rollout_error.png   -- relative FEM-L2 vs the convex-concave reference, per rollout step (log-y).
  * rollout_energy.png  -- Ginzburg-Landau energy per rollout step, with the CC reference energy
                           (light dashed = test ref, light dotted = train ref).

Colour encodes the training loss (min-movement / bare least-squares / data-driven). A dotted
vertical line marks the 10-step training horizon; everything to its right is time-extrapolation.
Diverged rollouts stop where they went non-finite and are marked with a red x.

Usage:
    python experiments/allen_cahn/realistic_ac/plot_rollout.py \
        --metrics output/allen_cahn/realistic_ac/metrics.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOSS_COLOR = {"mm": "#1f77b4", "ls": "#ff7f0e", "data": "#2ca02c"}
LOSS_NAME = {"mm": "min-movement", "ls": "bare least-squares", "data": "data-driven"}
SPLIT_STYLE = {"test": "-", "train": "--"}
ORDER = ["data", "mm", "ls"]


def nan_clean(vec):
    return [float("nan") if v is None else v for v in vec]


def sort_runs(runs):
    idx = {k: i for i, k in enumerate(ORDER)}
    return sorted(runs, key=lambda r: idx.get(r["loss"], 99))


def draw_split(ax, blk, split, field):
    """Plot every run's `field` (energy|error) for one split, styled solid=test / dashed=train."""
    for run in sort_runs(blk["runs"]):
        y = nan_clean(run[field])
        x = list(range(len(y)))
        lab = f"{LOSS_NAME.get(run['loss'], run['loss'])} · {split}"
        if run["diverged"]:
            lab += f"  (⊗{run['diverge_step']})"
        ax.plot(x, y, color=LOSS_COLOR.get(run["loss"], "gray"),
                ls=SPLIT_STYLE.get(split, "-"), lw=1.7, label=lab)
        if run["diverged"] and run["diverge_step"] > 0:
            j = run["diverge_step"] - 1
            ax.plot(j, y[j], "x", color="red", ms=8, mew=2, zorder=6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", default="output/allen_cahn/realistic_ac/metrics.json")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    data = json.load(open(args.metrics))
    steps, eps = data["steps"], data["eps"]
    horizon = data.get("train_horizon", 10)
    splits = data["splits"]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.metrics))
    os.makedirs(out_dir, exist_ok=True)
    suptitle = f"realistic_ac  (eps={eps:g}, {data.get('grid', 256)}², {steps} steps)"

    # --- error ---
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for split, blk in splits.items():
        draw_split(ax, blk, split, "error")
    ax.set_yscale("log")
    ax.axvline(horizon, color="0.7", ls=":", lw=1)
    ax.text(horizon, ax.get_ylim()[1], " train horizon", color="0.5", va="top", ha="left", fontsize=8)
    ax.set_xlabel("rollout step"); ax.set_ylabel("relative $L^2$ vs convex-concave")
    ax.set_title(f"Rollout error — {suptitle}")
    ax.legend(fontsize=8, framealpha=0.9, ncol=2); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = os.path.join(out_dir, "rollout_error.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  {p}")

    # --- energy ---
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for split, blk in splits.items():
        draw_split(ax, blk, split, "energy")
    for split, refls in (("test", "--"), ("train", ":")):
        if split in splits:
            ref = splits[split]["ref_energy"]
            ax.plot(range(len(ref)), ref, color="0.4", ls=refls, lw=1.4, zorder=8,
                    label=f"CC reference · {split}")
    e0 = splits[next(iter(splits))]["ref_energy"][0]
    ax.set_ylim(0, 2.5 * e0)                          # cap so a blow-up clips (its red x still shows)
    ax.axvline(horizon, color="0.7", ls=":", lw=1)
    ax.set_xlabel("rollout step"); ax.set_ylabel("Ginzburg–Landau energy")
    ax.set_title(f"Rollout energy — {suptitle}")
    ax.legend(fontsize=8, framealpha=0.9, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "rollout_energy.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  {p}")


if __name__ == "__main__":
    main()
