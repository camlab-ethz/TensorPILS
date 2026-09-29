"""Assemble the graphical abstract (Figure 1): one 1x3 figure, sized for a paper.

Everything here is derived from one requirement -- it must look right as a full-width 1x3 figure
in the ICLR template:

* **Authored at the final size.** The canvas is exactly the 5.5 in text width, so
  ``\\includegraphics[width=\\linewidth]`` does not rescale it and the point sizes below are the
  ones that reach the page. (The inspection renders were ~23 in wide with 26 pt type, which lands
  at ~6 pt once shrunk to a column -- smaller than body text.)
* **Type near body size.** Body text is 10 pt; labels here are 9 pt and ticks 7.5 pt, the usual
  "slightly smaller than body" convention.
* **No equations.** Panels are named in words, so the figure reads without the caption.
* **Downsampled.** The matrices are 4000 x 3969 (every one of the 4000 steps, the 63^2 interior
  modes of the 65 x 65 grid); at 5.5 in they are aggregated to the pixel grid
  by RMS over log-spaced bins rather than subsampled, so nothing is aliased away.

Usage (see run.sh):
    python experiments/graphical_abstract/plot_abstract.py \\
        --bare output/graphical_abstract/bare.npz \\
        --mg   output/graphical_abstract/mg.npz \\
        --out  output/graphical_abstract/graphical_abstract.pdf
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# The paper's three entities, anchored on a magma sub-path so any ramp between them stays
# saturated. Orange and blue (the previous L_PLS) are near-opposite hues, so a ramp holding both
# had to cross the achromatic axis -- measured minimum chroma 5.4, i.e. two muddy grey-brown steps.
# On this path the minimum is 14.5. Pairwise dE: LS-PLS 20.5, LS-data 41.0, PLS-data 22.8.
C_LS   = "#e07a1f"   # L_LS   -- unpreconditioned least squares
C_PLS  = "#bf3a77"   # L_PLS  -- preconditioned least squares
C_DATA = "#5c167f"   # L_data -- supervised

C_BARE = C_LS
C_MG = C_PLS
MUTED = "#6b6b6b"
TEXT = "#1a1a1a"

# Losses as written in the paper. B = I here (Euclidean residual norm), so no weighting subscript.
LAB_BARE = r"$L_{\mathrm{LS}}(\mathbf{u})=\frac{1}{2}\|A\mathbf{u}-\mathbf{f}\|^2$"
LAB_MG = r"$L_{\mathrm{PLS}}(\mathbf{u})=\frac{1}{2}\|P[A\mathbf{u}-\mathbf{f}]\|^2$"
LEG_BARE = r"$L_{\mathrm{LS}}$"
LEG_MG = r"$L_{\mathrm{PLS}}$"


def log_bin_edges(n, n_target):
    """Integer, strictly increasing, log-spaced bin edges over ``1..n`` (never empty)."""
    e = np.unique(np.round(np.geomspace(1, n + 1, n_target + 1)).astype(np.int64))
    return e[e <= n + 1]


def aggregate(amp, x_edges, y_edges):
    """RMS-aggregate ``amp`` (modes x iterations) onto the given bin edges.

    Aggregating rather than subsampling matters: at 5.5 in the panel is ~600 px wide and the data
    is 4000 columns, so picking every 7th column would drop the fine striping instead of averaging
    it in.
    """
    a2 = amp ** 2
    sx, cx = np.add.reduceat(a2, x_edges[:-1] - 1, axis=1), np.diff(x_edges)
    a2 = sx / cx[None, :]
    sy, cy = np.add.reduceat(a2, y_edges[:-1] - 1, axis=0), np.diff(y_edges)
    return np.sqrt(sy / cy[:, None])


def logbin_curve(steps, err, nbins):
    """Geometric-mean the curve within log-spaced iteration bins.

    Smoothing must be uniform in the *log* view, which is what is plotted: with every step
    recorded, iterations 1-10 hold 10 samples and 1000-4000 hold 3000, so a fixed-width moving
    average would flatten the early descent while barely touching the tail.
    """
    edges = np.geomspace(steps[0], steps[-1], nbins + 1)
    idx = np.clip(np.digitize(steps, edges) - 1, 0, nbins - 1)
    xs, ys = [], []
    for k in range(nbins):
        m = idx == k
        if m.any():
            xs.append(np.exp(np.log(steps[m]).mean()))
            ys.append(10.0 ** np.log10(np.maximum(err[m], 1e-300)).mean())
    return np.array(xs), np.array(ys)


def matrix_panel(ax, d, nx, ny, vmin):
    lam, spec, spec0 = d["lam"], d["spec"], d["spec0"]
    order = np.argsort(lam)
    amp = spec[:, order].astype(np.float64).T / np.linalg.norm(spec0)   # modes x iterations
    n_mode, n_it = amp.shape
    xe, ye = log_bin_edges(n_it, nx), log_bin_edges(n_mode, ny)
    Z = np.log10(np.maximum(aggregate(amp, xe, ye), 1e-12))
    steps = d["steps"].astype(float)
    xcoord = np.interp(xe, np.arange(1, n_it + 1), steps)   # bin edges -> true iteration numbers
    im = ax.pcolormesh(xcoord, ye.astype(float), Z, cmap="magma", vmin=vmin, vmax=0,
                       rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(1, n_mode)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bare", required=True)
    ap.add_argument("--mg", required=True)
    ap.add_argument("--out", default="output/graphical_abstract/graphical_abstract.pdf")
    ap.add_argument("--width", type=float, default=5.5, help="inches; the ICLR text width")
    ap.add_argument("--height", type=float, default=2.15)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--vmin", type=float, default=-5.0)
    ap.add_argument("--smooth", type=int, default=220,
                    help="log-spaced bins used to smooth the convergence curves (fewer = smoother)")
    ap.add_argument("--nx", type=int, default=700, help="aggregated columns per matrix panel")
    ap.add_argument("--ny", type=int, default=600, help="aggregated rows per matrix panel")
    args = ap.parse_args()

    db, dm = np.load(args.bare), np.load(args.mg)

    FS, FS_TICK = 9.0, 7.5
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    })

    fig = plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
    # Column 3 is an empty spacer: the colorbar's label sits on its right and the convergence
    # panel's y-label on its left, so without a gap between them the two overlap.
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1.0, 1.0, 0.05, 0.20, 1.05],
                  wspace=0.26, left=0.072, right=0.985, bottom=0.215, top=0.855)
    ax0, ax1, cax, ax2 = (fig.add_subplot(gs[0, i]) for i in (0, 1, 2, 4))

    im = matrix_panel(ax0, db, args.nx, args.ny, args.vmin)
    matrix_panel(ax1, dm, args.nx, args.ny, args.vmin)
    for ax, title in ((ax0, LAB_BARE), (ax1, LAB_MG)):
        ax.set_title(title, fontsize=FS, pad=4)
        ax.set_xlabel("iteration", fontsize=FS, labelpad=1.5)
        ax.tick_params(length=2.5, width=0.6, pad=1.5)
        # Auto-ticking gives only two decades on a 1.5 in log axis; force one per decade.
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6))
    ax0.set_ylabel("error per mode", fontsize=FS, labelpad=2)
    ax1.set_yticklabels([])

    cb = fig.colorbar(im, cax=cax, ticks=list(range(int(args.vmin), 1)))
    cb.ax.set_yticklabels([rf"$10^{{{k}}}$" for k in range(int(args.vmin), 1)])
    cb.ax.tick_params(labelsize=FS_TICK, length=2, width=0.6, pad=1.5)
    cb.outline.set_linewidth(0.5)

    # --- convergence panel ---
    # Dual-encode the two curves (colour *and* dash) so they stay distinguishable under colour
    # vision deficiency and in greyscale print.
    for d, colr, lab, ls in ((db, C_BARE, LEG_BARE, (0, (1.1, 1.7))),
                             (dm, C_MG, LEG_MG, "solid")):
        x, y = logbin_curve(d["steps"].astype(float), d["err_adam"].astype(float), args.smooth)
        ax2.plot(x, y, color=colr, lw=2.2, ls=ls, label=lab, zorder=4,
                 solid_capstyle="round", dash_capstyle="round")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("iteration", fontsize=FS, labelpad=1.5)
    ax2.set_ylabel("relative error", fontsize=FS, labelpad=2)
    ax2.tick_params(length=2.5, width=0.6, pad=1.5)
    ax2.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6))
    ax2.grid(True, which="major", color=MUTED, alpha=0.18, lw=0.5)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.legend(loc="lower left", frameon=False, fontsize=FS, handlelength=1.4,
               borderaxespad=0.2, labelspacing=0.25)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    # PDF alongside: text and curves stay vector, the two heatmaps are already rasterized. This is
    # the one that goes in the paper -- *.png is git-ignored repo-wide, so a PNG under
    # the paper repo's figures/ could not be committed and the build would break on a fresh clone.
    pdf = os.path.splitext(args.out)[0] + ".pdf"
    fig.savefig(pdf, dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {pdf}")
    px = (int(args.width * args.dpi), int(args.height * args.dpi))
    print(f"wrote {args.out}  ({px[0]}x{px[1]} px, {args.width}x{args.height} in at {args.dpi} dpi)")


if __name__ == "__main__":
    main()
