"""Validation curves for the Poisson paper table, STAGE 2 (the 500-epoch finals).

The stage-1 companion, ``plot_lr_sweep.py``, keys on *learning rate* -- one curve per lr inside
each arm -- because that is what stage 1 varies. The finals hold the lr fixed at the selected
value and vary the *seed*, so three near-identical lines per panel would say very little. Here
each arm gets one mean curve with a min-max band across seeds, which is the honest summary for
n = 3: a standard deviation over three samples is not a meaningful spread estimate.

Two figures:

``final_panels.{png,pdf}``
    One panel per arm, mean +/- min-max band, shared log y-axis. Use this to check that the seeds
    agree and that nothing diverged.
``final_overlay.{png,pdf}``
    All seven arms' mean curves on one axis. This is the comparison figure -- it is where the
    preconditioned arms separating from the bare ones is visible.

Colours in the overlay are the Okabe-Ito colourblind-safe set, and each arm additionally carries
its own dash pattern, so identity never rests on hue alone.

Usage:
    python experiments/poisson/poisson_paper/poisson_benchmark/plot_final.py \
        --root output/poisson/poisson_paper/poisson_benchmark/final
"""

import argparse
import glob
import json
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt

# Same arm table and matching rules as plot_lr_sweep.py -- order fixes panels, colours, and the
# order of any table generated downstream.
ARMS = [
    ("data",       "data-driven",         lambda n: n.startswith("fno_data_")),
    ("ls",         "least squares",       lambda n: n.startswith("fno_galerkin_")),
    ("pls",        "LS + multigrid",      lambda n: n.startswith("fno_pls_")),
    ("pino",       "PINO",                lambda n: n.startswith("fno_pino")),
    ("pideeponet", "PI-DeepONet",         lambda n: n.startswith("deeponet_pi-")),
]

# Okabe-Ito, minus the yellow (illegible on white). Seven arms, seven hues.
OKABE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
DASHES = [(None, None), (4, 1.5), (1, 1.5), (6, 1.5, 1, 1.5), (3, 1, 3, 1), (8, 2), (2, 1, 5, 1)]
MUTED, TEXT = "#6b6b6b", "#1a1a1a"


def load(root):
    """{arm_key: {seed: val_rel_l2_history}} from ``<root>/seed<S>/results/*.json``."""
    out = {k: {} for k, _, _ in ARMS}
    for path in sorted(glob.glob(os.path.join(root, "seed*", "results", "*.json"))):
        m = re.search(r"seed(\d+)[/\\]results", path)
        if not m:
            continue
        seed = int(m.group(1))
        name = os.path.basename(path)
        with open(path) as fh:
            run = json.load(fh)
        hist = run.get("stats", {}).get("val_rel_l2_errors") or []
        if not hist:
            continue
        for key, _, match in ARMS:
            if match(name):
                out[key][seed] = np.asarray(hist, dtype=float)
                break
    return out


def stack(runs):
    """Seed histories -> (epochs, mean, lo, hi). Truncates to the shortest, so a run that died
    early shortens the curve rather than silently contributing NaNs to the band."""
    n = min(len(h) for h in runs.values())
    arr = np.vstack([h[:n] for _, h in sorted(runs.items())])
    return np.arange(1, n + 1), arr.mean(axis=0), arr.min(axis=0), arr.max(axis=0)


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.edgecolor": MUTED, "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    })


def despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, which="major", color=MUTED, alpha=0.2, lw=0.5)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/poisson/poisson_paper/poisson_benchmark/final")
    ap.add_argument("--out_dir", default=None, help="default: alongside --root")
    args = ap.parse_args()
    out_dir = args.out_dir or args.root

    data = load(args.root)
    n_found = sum(len(v) for v in data.values())
    if not n_found:
        raise SystemExit(f"no runs with a validation history under {args.root}")
    print(f"loaded {n_found} runs")
    style()

    # ---------------------------------------------------------------- per-arm panels
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    summary = {}
    for i, (ax, (key, title, _)) in enumerate(zip(axes.ravel(), ARMS)):
        runs = data[key]
        if not runs:
            ax.set_title(f"{title}\n(no runs)", fontsize=10); ax.set_axis_off(); continue
        ep, mean, lo, hi = stack(runs)
        c = OKABE[i]
        ax.fill_between(ep, lo, hi, color=c, alpha=0.18, lw=0)
        ax.plot(ep, mean, lw=1.5, color=c)
        # Per-seed best, so the panel title reports the spread that the table will have to carry.
        bests = np.array([h.min() for h in runs.values()])
        summary[key] = (title, bests, sorted(runs))
        ax.set_yscale("log"); ax.set_xlabel("epoch")
        ax.set_title(f"{title}\nbest {bests.mean():.4f}  [{bests.min():.4f}, {bests.max():.4f}]",
                     fontsize=9)
        despine(ax)
    axes[0, 0].set_ylabel(r"validation relative $L^2$")
    axes[1, 0].set_ylabel(r"validation relative $L^2$")
    axes[-1, -1].set_axis_off()                       # 5 arms in a 6-panel grid
    fig.tight_layout(pad=0.6)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "final_panels")
    fig.savefig(p + ".png", dpi=200); fig.savefig(p + ".pdf")
    plt.close(fig)
    print(f"  {p}.png\n  {p}.pdf")

    # ------------------------------------------------------------------- overlay
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for i, (key, title, _) in enumerate(ARMS):
        runs = data[key]
        if not runs:
            continue
        ep, mean, lo, hi = stack(runs)
        c = OKABE[i]
        ax.fill_between(ep, lo, hi, color=c, alpha=0.12, lw=0)
        line, = ax.plot(ep, mean, lw=1.6, color=c, label=f"{title}  ({mean.min():.3f})")
        if DASHES[i][0] is not None:
            line.set_dashes(DASHES[i])
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"validation relative $L^2$")
    ax.legend(fontsize=8, frameon=False, ncol=2, title="arm (best mean)", title_fontsize=8)
    despine(ax)
    fig.tight_layout(pad=0.6)
    p = os.path.join(out_dir, "final_overlay")
    fig.savefig(p + ".png", dpi=200); fig.savefig(p + ".pdf")
    plt.close(fig)
    print(f"  {p}.png\n  {p}.pdf")

    # ------------------------------------------------------------------- summary
    print("\nbest validation relative L2 (per seed, then mean +/- half-range):")
    for key, _, _ in ARMS:
        if key not in summary:
            continue
        title, bests, seeds = summary[key]
        per = "  ".join(f"s{s}={b:.4f}" for s, b in zip(seeds, bests))
        half = 0.5 * (bests.max() - bests.min())
        print(f"  {title:<22s} {bests.mean():.4f} +/- {half:.4f}   ({per})")
    missing = [k for k, _, _ in ARMS if len(data[k]) < 3]
    if missing:
        print(f"\nWARNING: fewer than 3 seeds for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
