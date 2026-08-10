"""Graphical abstract: Adam on the plain vs. preconditioned least-squares loss.

PLACEHOLDER. The numbers below are hand-made and only serve to fix the layout of the
figure in the paper. Replace ``PLAIN`` / ``PRECOND`` with measured relative L^2 errors
once the real sweep has been run; nothing else in this file needs to change.

The intended experiment: minimise 0.5||Au - f||^2 with Adam, where A is the P1/FD
stiffness matrix of the Dirichlet Poisson problem on (0,1)^2 discretised on an N x N
regular grid, and compare against 0.5||P[Au - f]||^2 with P ~ A^{-1} (one multigrid
V-cycle). Same optimiser, same step budget, same initialisation for both curves.

Usage:
    python experiments/poisson/graphical_abstract/plot_graphical_abstract.py \
        --out paper/figures/graphical_abstract.pdf
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless: render to file, no display needed
import matplotlib.pyplot as plt

# --- placeholder data ----------------------------------------------------------------
# grid points per dimension on the unit square; mesh size h = 1/N
N = [8, 16, 32, 64, 128]

# relative L^2 error after a fixed Adam budget, plain loss 0.5||Au - f||^2.
# cond(nabla^2 L) ~ h^-4, so the error blows up to the trivial predictor u = 0 (100%).
PLAIN = [0.011, 0.079, 0.31, 0.72, 0.98]
PLAIN_LO = [0.008, 0.062, 0.26, 0.66, 0.95]
PLAIN_HI = [0.015, 0.098, 0.37, 0.79, 1.00]

# same budget, preconditioned loss 0.5||P[Au - f]||^2 with P ~ A^{-1}: mesh-independent.
PRECOND = [0.0021, 0.0023, 0.0026, 0.0028, 0.0032]
PRECOND_LO = [0.0017, 0.0019, 0.0022, 0.0023, 0.0026]
PRECOND_HI = [0.0026, 0.0028, 0.0031, 0.0034, 0.0039]

C_PLAIN = "#e07a1f"  # orange
C_PRECOND = "#1b6ec2"  # blue

TEXT = "#1a1a1a"
MUTED = "#6b6b6b"


def make_figure(out_path):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "legend.fontsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.6,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )

    # 2.6in == the wrapfigure column width (0.47\textwidth of 5.5in), so
    # \includegraphics[width=\linewidth] does not rescale the figure and the font
    # sizes above are the ones that reach the page.
    fig, ax = plt.subplots(figsize=(2.6, 1.75))

    # reference: relative error of the trivial predictor u = 0
    ax.axhline(1.0, color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.text(8.4, 1.12, "$u\\equiv 0$", color=MUTED, fontsize=6, va="bottom", ha="left")

    ax.fill_between(N, PLAIN_LO, PLAIN_HI, color=C_PLAIN, alpha=0.18, lw=0, zorder=2)
    ax.fill_between(
        N, PRECOND_LO, PRECOND_HI, color=C_PRECOND, alpha=0.18, lw=0, zorder=2
    )

    ax.plot(
        N,
        PLAIN,
        color=C_PLAIN,
        lw=1.4,
        marker="o",
        ms=3.4,
        mec="white",
        mew=0.7,
        zorder=4,
        label=r"$\|Au-f\|^2$",
    )
    ax.plot(
        N,
        PRECOND,
        color=C_PRECOND,
        lw=1.4,
        marker="s",
        ms=3.4,
        mec="white",
        mew=0.7,
        zorder=4,
        label=r"$\|P[Au-f]\|^2$",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(N)
    ax.set_xticklabels([str(n) for n in N])
    ax.set_xlim(7.0, 146)
    ax.set_ylim(1e-3, 2.2)
    ax.set_xlabel(r"grid points per dimension $N$")
    ax.set_ylabel(r"relative $L^2$ error")

    ax.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
    ax.set_yticklabels(["0.1%", "1%", "10%", "100%"])
    ax.grid(True, which="major", axis="both", color=MUTED, alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # lower-right wedge: above the flat preconditioned curve, right of the steep one
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.20),
        frameon=False,
        handlelength=1.3,
        handletextpad=0.5,
        labelspacing=0.35,
        borderaxespad=0.0,
    )

    # no bbox_inches="tight": keep the saved width exactly at figsize so the PDF drops
    # into the wrapfigure column at scale 1.
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(os.path.splitext(out_path)[0] + ".png", dpi=300)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="paper/figures/graphical_abstract.pdf")
    args = p.parse_args()
    make_figure(args.out)
