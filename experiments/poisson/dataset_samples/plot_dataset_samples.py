"""Appendix figure ``poisson_dataset_samples``: sample (source, solution) pairs from the Poisson
dataset.

A 2x2 grid: one ``(f, u)`` pair at ``K=4`` above one at ``K=10``, the frequency content the paper's
Poisson experiments actually use. Field styling follows
``tensorpils/viz.py`` (``RdBu_r``, ``origin="lower"``); the panels are drawn straight from
:class:`PoissonDataset`, so no trained model is involved.

Authored at the text width so ``\\includegraphics[width=\\linewidth]`` does not rescale it and the
point sizes below are what reaches the page.

Usage (CPU, seconds):
    python experiments/poisson/dataset_samples/plot_dataset_samples.py
"""

import argparse
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt

from tensorpils.data import PoissonDataset

MUTED = "#6b6b6b"
TEXT = "#1a1a1a"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Ks", type=int, nargs=2, default=[4, 10],
                    help="the two frequency cutoffs to show (default 4 10)")
    # 65, not 64: nested-dyadic, and what every Poisson experiment in the paper runs on.
    ap.add_argument("--grid", type=int, default=65)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample", type=int, default=0, help="index of the drawn sample")
    ap.add_argument("--out_dir", default="output/poisson/dataset_samples")
    ap.add_argument("--name", default="poisson_dataset_samples")
    ap.add_argument("--layout", choices=["1x4", "2x2"], default="1x4",
                    help="all four panels in a row (default) or stacked by K")
    ap.add_argument("--width", type=float, default=5.5, help="inches; the text width")
    ap.add_argument("--height", type=float, default=None,
                    help="default: 1.55 in for 1x4, 4.9 in for 2x2")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    FS, FS_TICK = 9.0, 7.0
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    })

    row_major = args.layout == "2x2"
    nr, nc = (2, 2) if row_major else (1, 4)
    height = args.height if args.height else (4.9 if row_major else 1.55)
    fig, axes = plt.subplots(nr, nc, figsize=(args.width, height), dpi=args.dpi)
    axes = np.atleast_2d(axes)
    for row, K in enumerate(args.Ks):
        ds = PoissonDataset(num_samples=args.sample + 1, K=K, seed=args.seed,
                            grid_resolution=args.grid)
        (f_grid, _, _, _), u_grid = ds[args.sample]
        f = f_grid.squeeze(0).numpy()
        u = u_grid.numpy()
        for col, (field, sym) in enumerate(((f, "f"), (u, "u"))):
            ax = axes[row, col] if row_major else axes[0, 2 * row + col]
            v = float(np.abs(field).max())
            im = ax.imshow(field, cmap="RdBu_r", origin="lower", extent=(0, 1, 0, 1),
                           vmin=-v, vmax=v)                     # symmetric: these fields are signed
            ax.set_title(rf"${sym}$,  $K={K}$", fontsize=FS, pad=3)
            ax.set_xticks([0, 1])
            # In a row of four the y ticks repeat for no gain; keep them on the first panel only.
            ax.set_yticks([0, 1] if (row_major or (row == 0 and col == 0)) else [])
            ax.tick_params(labelsize=FS_TICK, length=2, width=0.5, pad=1)
            # Three explicit ticks in plain decimals. The default scientific formatter parks a
            # floating "1e-2" above the bar, which collides with the panel title and reads as a
            # stray label; a rounded tick value carries the same information in place.
            # Round *down* to one significant figure: "%.1g" rounds to nearest, which can push
            # the tick past the colour range (0.015 -> 0.02 > vmax) so it is silently not drawn.
            e = math.floor(math.log10(0.8 * v)) if v > 0 else 0
            t = math.floor(0.8 * v / 10 ** e) * 10 ** e
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[-t, 0.0, t])
            cb.ax.set_yticklabels([f"{x:g}" for x in (-t, 0.0, t)])
            cb.ax.tick_params(labelsize=FS_TICK, length=2, width=0.5, pad=1)
            cb.outline.set_linewidth(0.5)

    fig.tight_layout(pad=0.5)
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, args.name)
    fig.savefig(p + ".png", dpi=args.dpi)
    fig.savefig(p + ".pdf")
    plt.close(fig)
    print(f"  {p}.png\n  {p}.pdf")


if __name__ == "__main__":
    main()
