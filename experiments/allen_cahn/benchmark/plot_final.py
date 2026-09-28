"""Validation curves and the paper-table numbers for the Allen-Cahn benchmark, STAGE 2 (finals).

The finals hold each arm's learning rate fixed at the value ``plot_lr_sweep.py`` selected and vary
the seed, so each arm gets one mean curve with a min-max band across seeds (the honest summary for
n = 3). Two points worth knowing:

* **The metric** is ``stats.val_st_rel_l2``, the space-time relative L2 over the rollout window,
  and the table numbers are the *test* space-time relative L2 at the best-validation checkpoint
  (``test_st_rel_l2`` in the results JSON), which is what the paper's summary table reports.
* **Divergence is flagged**, with the same rules as the lr sweep (``plot_lr_sweep.diagnose``): a
  seed that goes NaN or collapses to the trivial state (>= COLLAPSE) is reported by name. It stays
  in the band -- a failure is a result, not an outlier to drop -- but it cannot pass silently.

Two figures, written alongside ``--root``:

``ac_final_panels.{png,pdf}``
    One panel per arm, mean curve with min-max band, shared log y-axis.
``ac_final_overlay.{png,pdf}``
    All arms' mean curves on one axis; the legend gives each arm's per-seed best validation error
    as mean [min, max].

Spread over seeds is reported as mean [min, max] throughout, not mean +/- half-range: with n = 3
no inferential statistic is reliable, and mean +/- half-range reads as an interval that coincides
with [min, max] only when the mean sits at the midpoint.

Usage:
    python experiments/allen_cahn/benchmark/plot_final.py
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

from plot_lr_sweep import ARMS, COLLAPSE, diagnose

# Okabe-Ito and dash patterns exactly as in the Poisson plot_final.py, so data/ls/pls/pino/
# PI-DeepONet carry the same identity in both papers' figures; min. movement takes the sixth.
OKABE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
DASHES = [(None, None), (4, 1.5), (1, 1.5), (6, 1.5, 1, 1.5), (3, 1, 3, 1), (8, 2), (2, 1, 5, 1)]
MUTED, TEXT = "#6b6b6b", "#1a1a1a"
TEST_KEYS = ("test_st_rel_l2", "test_final_rel_l2")


def load(root):
    """{arm: {seed: run dict}} from ``<root>/seed<S>/results/*.json``; runs without a history are skipped."""
    out = {k: {} for k, _, _ in ARMS}
    for path in sorted(glob.glob(os.path.join(root, "seed*", "results", "*.json"))):
        m = re.search(r"seed(\d+)[/\\]results", path)
        if not m:
            continue
        with open(path) as fh:
            run = json.load(fh)
        hist = run.get("stats", {}).get("val_st_rel_l2") or []
        if not hist:
            continue
        run["hist"] = np.asarray(hist, dtype=float)
        name = os.path.basename(path)
        for key, _, match in ARMS:
            if match(name):
                out[key][int(m.group(1))] = run
                break
    return out


def stack(runs):
    """Seed histories -> (epochs, mean, lo, hi), truncated to the shortest run. NaN-aware, so a
    seed that diverged mid-run does not erase the other seeds' band; diagnose() reports it."""
    n = min(len(r["hist"]) for r in runs.values())
    arr = np.vstack([r["hist"][:n] for _, r in sorted(runs.items())])
    with np.errstate(all="ignore"):
        return (np.arange(1, n + 1), np.nanmean(arr, axis=0),
                np.nanmin(arr, axis=0), np.nanmax(arr, axis=0))


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


def mean_min_max(x):
    """'mean [min, max]' over the finite seeds, annotated when some seed is NaN (diverged)."""
    x = np.asarray(x, dtype=float)
    f = x[np.isfinite(x)]
    if f.size == 0:
        return "all NaN"
    s = f"{f.mean():.4f} [{f.min():.4f}, {f.max():.4f}]"
    return s + (f" ({f.size}/{x.size} finite)" if f.size < x.size else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/allen_cahn/benchmark/final")
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
    flags = []
    for i, (ax, (key, title, _)) in enumerate(zip(axes.ravel(), ARMS)):
        runs = data[key]
        if not runs:
            ax.set_title(f"{title}\n(no runs)", fontsize=10); ax.set_axis_off(); continue
        ep, mean, lo, hi = stack(runs)
        c = OKABE[i]
        ax.fill_between(ep, lo, hi, color=c, alpha=0.18, lw=0)
        ax.plot(ep, mean, lw=1.5, color=c)
        bests = []
        for seed, r in sorted(runs.items()):
            best, note, _ = diagnose(r["hist"])
            bests.append(best)
            if note:
                flags.append(f"  {title:<16s} seed {seed}  {note}")
        bests = np.array(bests)
        ax.set_yscale("log"); ax.set_xlabel("epoch")
        ax.set_title(f"{title}\nbest val {mean_min_max(bests)}", fontsize=9)
        despine(ax)
    axes[0, 0].set_ylabel(r"validation space-time relative $L^2$")
    axes[1, 0].set_ylabel(r"validation space-time relative $L^2$")
    fig.tight_layout(pad=0.6)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "ac_final_panels")
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
        with np.errstate(all="ignore"):
            bests = np.array([np.nanmin(r["hist"]) for r in runs.values()])
        label = f"{title}  {np.nanmean(bests):.3f} [{np.nanmin(bests):.3f}, {np.nanmax(bests):.3f}]"
        line, = ax.plot(ep, mean, lw=1.6, color=c, label=label)
        if DASHES[i][0] is not None:
            line.set_dashes(DASHES[i])
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"validation space-time relative $L^2$")
    ax.legend(fontsize=8, frameon=False, ncol=2, title="arm: best val, mean [min, max] over seeds",
              title_fontsize=8)
    despine(ax)
    fig.tight_layout(pad=0.6)
    p = os.path.join(out_dir, "ac_final_overlay")
    fig.savefig(p + ".png", dpi=200); fig.savefig(p + ".pdf")
    plt.close(fig)
    print(f"  {p}.png\n  {p}.pdf")

    # ------------------------------------------------------------------- summary
    if flags:
        print(f"\ndiverged or collapsed seeds (>= {COLLAPSE}; kept in the band and the table):")
        print("\n".join(flags))
    print("\nper arm, mean [min, max] over seeds (test = best-validation checkpoint):")
    for key, title, _ in ARMS:
        runs = data[key]
        if not runs:
            continue
        print(f"  {title}  (seeds {','.join(map(str, sorted(runs)))})")
        print(f"    best val st-rel   {mean_min_max([r['best_val_st_rel_l2'] for r in runs.values()])}")
        for k, lab in zip(TEST_KEYS, ("test st-rel", "test final-step")):
            print(f"    {lab:<17s} {mean_min_max([r[k] for r in runs.values()])}")
    missing = [t for k, t, _ in ARMS if 0 < len(data[k]) < 3]
    if missing:
        print(f"\nWARNING: fewer than 3 seeds for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
