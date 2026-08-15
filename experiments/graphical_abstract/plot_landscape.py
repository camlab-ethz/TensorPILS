"""Graphical abstract, part 2: figures from ``run_landscape.py``'s ``.npz``.

Three independent figures, each written as both ``.png`` (quick viewing) and ``.pdf`` (the paper;
note ``*.png`` is git-ignored repo-wide, so only the PDF can be committed under ``paper/figures/``):

  landscape_contours   loss contours in the plane of the extreme eigenvectors of A, with the true
                       Adam trajectory projected in, and an inset showing the contours at their
                       true 1:1 aspect.
  landscape_error      relative error vs iteration for Adam and gradient descent.
  landscape_spectrum   modal amplitude of the error over (iteration, eigenvalue) -- which
                       frequencies the optimizer removes and which it cannot touch.

Usage:
    python experiments/graphical_abstract/plot_landscape.py \
        --npz output/graphical_abstract/landscape.npz --out_dir paper/figures
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt

C_ADAM = "#1b6ec2"     # blue  -- the optimizer under test
C_GD = "#e07a1f"       # orange
MUTED = "#6b6b6b"
TEXT = "#1a1a1a"


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })


def tag(d):
    """Short label for the loss the run minimised, for figure titles."""
    pc = str(d["precond"]) if "precond" in d.files else "none"
    return r"$\frac{1}{2}\|P(Au-b)\|^2$, $P$ = MG V-cycle" if pc == "multigrid" \
        else r"$\frac{1}{2}\|Au-b\|^2$"


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p + ".png", dpi=200)
    fig.savefig(p + ".pdf")
    plt.close(fig)
    print(f"  {p}.png\n  {p}.pdf")


# --------------------------------------------------------------------------- contours
def fig_contours(d, out_dir, render_aspect=None, which="gd"):
    """Contours of L restricted to span(v_max, v_min), plus the projected Adam path.

    On this plane ``L - L* = 1/2 (lam_max^2 a^2 + lam_min^2 b^2)``, so the level sets are ellipses
    of semi-axis ratio ``lam_max/lam_min = kappa(A)``. Note this is the *square root* of the
    Hessian's condition number: the eigenvalues enter the quadratic form, so they enter the axis
    lengths under a root. The drawing therefore understates the pathology by a square root.
    """
    lam_max, lam_min = float(d["lam_max"]), float(d["lam_min"])
    kappa = lam_max / lam_min
    # Default to GD: it follows the classical mode-wise rates, so it hugs the valley floor and the
    # trajectory shares the contours' anisotropy. Adam crosses the soft direction in ~10 steps and
    # then wanders isotropically, which no choice of window can show against 804:1 contours.
    a, b = (d["alpha_gd"], d["beta_gd"]) if which == "gd" else (d["alpha"], d["beta"])
    col = C_GD if which == "gd" else C_ADAM

    # Window: fit the trajectory. The trajectory's own aspect and the contours' 804:1 are
    # different geometries and cannot both be shown undistorted; the inset carries the truth.
    Ra = max(np.abs(a).max(), abs(float(d["alpha0"])), 1e-30) * 1.25
    Rb = max(np.abs(b).max(), abs(float(d["beta0"])), 1e-30) * 1.25
    if render_aspect is not None:
        # Force a target drawn ratio while still containing the trajectory. Which axis binds
        # depends on the optimizer: GD hugs the valley floor (tiny alpha, large beta) so beta
        # binds; Adam wanders isotropically so alpha binds. Taking the max of both candidates
        # for Ra satisfies whichever it is.
        Ra = max(Ra, (render_aspect / kappa) * Rb)
        Rb = (kappa / render_aspect) * Ra

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    A, B = np.meshgrid(np.linspace(-Ra, Ra, 400), np.linspace(-Rb, Rb, 400))
    Z = 0.5 * (lam_max**2 * A**2 + lam_min**2 * B**2)
    levels = np.geomspace(max(Z[Z > 0].min(), Z.max() * 1e-6), Z.max(), 12)
    ax.contour(A, B, Z, levels=levels, colors=MUTED, linewidths=0.5, alpha=0.75)

    ax.plot(a, b, color=col, lw=1.0, alpha=0.95, zorder=4)
    ax.scatter(a, b, s=5, color=col, zorder=5, linewidths=0)
    ax.scatter([d["alpha0"]], [d["beta0"]], s=26, facecolor="white",
               edgecolor=col, linewidths=1.1, zorder=6)
    ax.scatter([0], [0], marker="*", s=70, color="#c0392b", zorder=7)

    # No gradient arrow: at u0 = 0 the error lies in a single eigendirection, so -grad L is exactly
    # parallel to the direction to the minimum. The "gradient points the wrong way" picture needs a
    # mixed-mode error and would be misleading here.
    a0, b0 = float(d["alpha0"]), float(d["beta0"])
    ax.annotate(r"start ($u_0=0$)", xy=(a0, b0), xytext=(8, -2), textcoords="offset points",
                fontsize=7, color=col, va="center")
    ax.annotate("minimum", xy=(0, 0), xytext=(8, 0), textcoords="offset points",
                fontsize=7, color="#c0392b", va="center")
    ax.annotate(f"$10^{{{int(np.log10(d['steps'][-1]))}}}$ steps", xy=(a[-1], b[-1]),
                xytext=(8, 4), textcoords="offset points", fontsize=7, color=col, va="center")

    rendered = kappa * Ra / Rb
    ax.set_xlim(-Ra, Ra); ax.set_ylim(-Rb, Rb)
    ax.set_xlabel(r"error along $v_{\max}$  (stiff)")
    ax.set_ylabel(r"error along $v_{\min}$  (soft)")
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Inset: the same contours with equal data ranges on both axes, i.e. true aspect.
    ins = ax.inset_axes([0.70, 0.05, 0.27, 0.27])
    A2, B2 = np.meshgrid(np.linspace(-Rb, Rb, 240), np.linspace(-Rb, Rb, 240))
    Z2 = 0.5 * (lam_max**2 * A2**2 + lam_min**2 * B2**2)
    ins.contour(A2, B2, Z2, levels=np.geomspace(Z2.max() * 1e-4, Z2.max(), 7),
                colors=MUTED, linewidths=0.4)
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_title(f"true aspect\n{kappa:.0f}:1", fontsize=6, color=MUTED, pad=2)

    ax.set_title(rf"$\kappa(A)={kappa:.0f}$, $\kappa(\nabla^2L)={kappa**2:.1e}$"
                 f"   (drawn {rendered:.0f}:1)", fontsize=8, pad=6)
    fig.tight_layout(pad=0.3)
    save(fig, out_dir, "landscape_contours")


# --------------------------------------------------------------------------- error
def fig_error(d, out_dir):
    s = d["steps"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.text(s[0], 1.06, r"$u\equiv 0$", color=MUTED, fontsize=6.5, va="bottom")
    ax.plot(s, d["err_gd"], color=C_GD, lw=1.6, label="gradient descent")
    ax.plot(s, d["err_adam"], color=C_ADAM, lw=1.8, label=f"Adam (lr {float(d['best_lr']):.1e})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("iteration"); ax.set_ylabel(r"relative $L^2$ error")
    ax.grid(True, which="major", color=MUTED, alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower left", frameon=False, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    save(fig, out_dir, "landscape_error")


# --------------------------------------------------------------------------- spectrum
def fig_spectrum(d, out_dir, smooth=25, which="adam"):
    """Where the remaining error lives, as a function of eigenvalue and iteration.

    Amplitudes are normalised by the *total* initial error norm rather than per mode: with
    ``u0 = 0`` the initial error is smooth, so many high modes start at ~0 and a per-mode ratio
    would be meaningless.
    """
    lam, spec0 = d["lam"], d["spec0"]
    spec = d["spec"] if which == "adam" else d["spec_gd"]
    s = d["steps"]
    nrm0 = np.linalg.norm(spec0)

    # No binning: plot every mode, ordered by eigenvalue. Binning by equal log-lambda width
    # leaves empty rows (the low-lambda modes are sparse); binning by rank truncates the
    # lambda axis and hides the very modes the figure is about.
    order = np.argsort(lam)
    amp = spec[:, order].astype(np.float64) / nrm0
    if smooth > 1:
        # Moving RMS along the mode axis only: denoises the speckle without binning, so the
        # lambda axis stays exact. Normalise by the number of samples actually inside the window
        # -- a plain mode="same" convolution zero-pads, which damps the lowest and highest ~12
        # modes by up to sqrt(0.52) and would understate exactly the band the figure is about.
        k = np.ones(smooth)
        cnt = np.convolve(np.ones(amp.shape[1]), k, mode="same")
        amp = np.sqrt(np.apply_along_axis(
            lambda r: np.convolve(r, k, mode="same"), 1, amp ** 2) / cnt)
    Z = np.log10(np.maximum(amp, 1e-12)).T

    # Explicit geometric cell edges. shading="nearest" centres a cell on each sample and extends it
    # half a (log) spacing either way, which pushes the first cell below iteration 1 -- the axis
    # then shows iterations < 10^0, which do not exist. Edges also make the unequal cell widths
    # tile exactly instead of overlapping.
    def _edges(v):
        e = np.empty(len(v) + 1)
        e[1:-1] = np.sqrt(v[:-1] * v[1:])
        e[0], e[-1] = v[0], v[-1] ** 2 / e[-2]
        return e

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    pcm = ax.pcolormesh(_edges(s.astype(float)), _edges(lam[order]), Z, cmap="magma",
                        vmin=-5, vmax=0, rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(s[0], s[-1])
    ax.set_xlabel("iteration"); ax.set_ylabel(r"eigenvalue $\lambda$ of $A$")
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}$ modal error amplitude", fontsize=8)
    cb.outline.set_linewidth(0.4)
    ax.set_title(f"{'Adam' if which == 'adam' else 'gradient descent'} \u2014 {tag(d)}",
                 fontsize=8, pad=6)
    fig.tight_layout(pad=0.3)
    save(fig, out_dir, "landscape_spectrum" + ("" if which == "adam" else "_gd"))


BANDS = [(0.0, 0.1, r"$\lambda<0.1$"), (0.1, 1.0, r"$0.1\leq\lambda<1$"),
         (1.0, np.inf, r"$\lambda\geq 1$")]


def fig_bands(d, out_dir):
    """Error contributed by each eigenvalue band, per iteration, relative to the initial error.

    Absolute rather than fractional: once preconditioning drives the total to ~1e-4 the fractions
    bounce chaotically because they are fractions of a vanishing quantity. On a log axis this
    shows the decay *and* the composition, and the two runs stay comparable.

    Note the heatmap shows per-*mode* amplitude, which is diluted by how many modes share a band
    (3511 above lambda=1 vs 26 below 0.1), so a broad dim band can carry far more error than a
    narrow bright one. This figure is the un-diluted version and is the one to quote.
    """
    lam, s = d["lam"], d["steps"]
    nrm0 = np.linalg.norm(d["spec0"].astype(np.float64))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, key, name in ((axes[0], "spec", "Adam"), (axes[1], "spec_gd", "gradient descent")):
        c = d[key].astype(np.float64)
        ax.plot(s, np.sqrt((c ** 2).sum(axis=1)) / nrm0, color=TEXT, lw=1.8,
                label="total", zorder=5)
        for (lo, hi, lab), sty, colr in zip(BANDS, ["-", "--", ":"],
                                            ["#1b6ec2", "#e07a1f", "#2e7d32"]):
            m = (lam >= lo) & (lam < hi)
            ax.plot(s, np.sqrt((c[:, m] ** 2).sum(axis=1)) / nrm0, lw=1.3, ls=sty, color=colr,
                    label=f"{lab}  ({int(m.sum())})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(1e-7, 20)
        ax.set_xlabel("iteration")
        ax.set_title(name, fontsize=8, pad=5)
        ax.grid(True, which="major", color=MUTED, alpha=0.18, lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(r"$\|e_k\|$ in band $/\ \|e_0\|$")
    axes[1].legend(loc="lower left", frameon=False, handlelength=1.8, fontsize=6.5, ncol=2)
    fig.suptitle(tag(d), fontsize=8, y=0.99)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
    save(fig, out_dir, "landscape_bands")


def fig_matrix(d, out_dir, which="adam", dpi=200, vmin=-5.0, logx=False, logy=False, smooth=1):
    """Full-resolution matrix view: one pixel per (iteration, mode), linear axes.

    Rows are modes ordered by eigenvalue (index 1 = smallest lambda at the bottom), columns are
    iterations. No log axes, no smoothing, no binning -- this is the raw decomposition, meant for
    inspection at ~1:1 pixel scale rather than for the paper. PNG only: a 4k x 4k rasterised
    image inside a PDF is needlessly heavy.
    """
    lam, spec0 = d["lam"], d["spec0"]
    spec = d["spec"] if which == "adam" else d["spec_gd"]
    s = d["steps"]
    order = np.argsort(lam)
    amp = spec[:, order].astype(np.float64) / np.linalg.norm(spec0)
    if smooth > 1:                      # same mode-axis moving RMS as the summary spectrum figure,
        k = np.ones(smooth)             # count-normalised so the end modes are not damped
        cnt = np.convolve(np.ones(amp.shape[1]), k, mode="same")
        amp = np.sqrt(np.apply_along_axis(
            lambda r: np.convolve(r, k, mode="same"), 1, amp ** 2) / cnt)
    Z = np.log10(np.maximum(amp, 1e-12)).T

    n_mode, n_it = Z.shape
    # The canvas is ~20 inches across, so the shared 8pt style would be illegible; scale type to it.
    fs = 26
    fig, ax = plt.subplots(figsize=(n_it / dpi + 3.0, n_mode / dpi + 2.0), dpi=dpi)
    if logx or logy:
        # Log axes cannot use imshow (it assumes linear extent), so mesh with explicit geometric
        # cell edges. Rank is a near-linear coordinate in lambda here (lambda ~ rank^0.82, Weyl),
        # so a log rank axis is effectively the log-lambda axis of the summary spectrum figure --
        # and it expands the ~0.7% of modes with lambda < 0.1 that carry the story.
        def _edges(v):
            e = np.empty(len(v) + 1)
            e[1:-1] = np.sqrt(v[:-1] * v[1:])
            e[0], e[-1] = v[0], v[-1] ** 2 / e[-2]
            return e
        xs = _edges(d["steps"].astype(float))
        ys = _edges(np.arange(1, n_mode + 1, dtype=float))
        im = ax.pcolormesh(xs, ys, Z, cmap="magma", vmin=vmin, vmax=0, rasterized=True)
        if logx:
            # Limits must be the real iteration numbers, not the record count: with --stride the
            # two differ by the stride factor and the view would clip to the first steps/stride.
            ax.set_xscale("log"); ax.set_xlim(float(s[0]), float(s[-1]))
        if logy:
            ax.set_yscale("log"); ax.set_ylim(1, n_mode)
    else:
        im = ax.imshow(Z, cmap="magma", vmin=vmin, vmax=0, origin="lower", aspect="auto",
                       interpolation="nearest", extent=(float(s[0]), float(s[-1]), 0.5, n_mode + 0.5))
    ax.set_xlabel("iteration", fontsize=fs)
    ax.set_ylabel(r"mode index, ordered by eigenvalue $\lambda$ (1 = smallest)", fontsize=fs)
    ax.set_title(f"{'Adam' if which == 'adam' else 'gradient descent'} \u2014 {tag(d)}",
                 fontsize=fs * 1.15, pad=14)
    ax.tick_params(labelsize=fs * 0.8, length=8, width=1.2)
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.035)
    cb.set_label(r"$\log_{10}$ modal error amplitude $/\ \|e_0\|$", fontsize=fs)
    cb.ax.tick_params(labelsize=fs * 0.8)
    fig.tight_layout(pad=0.6)
    os.makedirs(out_dir, exist_ok=True)
    q = os.path.join(out_dir, "matrix_" + which + ("_log" + ("x" if logx else "") + ("y" if logy else "") if (logx or logy) else ""))
    fig.savefig(q + ".png", dpi=dpi)
    plt.close(fig)
    print(f"  {q}.png   ({n_it} iterations x {n_mode} modes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="output/graphical_abstract/landscape.npz")
    ap.add_argument("--out_dir", default=None,
                    help="default: alongside the npz")
    ap.add_argument("--which", choices=["gd", "adam"], default="gd",
                    help="trajectory drawn in the contour panel (default gd)")
    ap.add_argument("--matrix", action="store_true",
                    help="render only the full-resolution matrix view (needs --record every)")
    ap.add_argument("--matrix_smooth", type=int, default=1,
                    help="mode-axis moving-RMS window for the matrix view (1 = raw; 25 matches "
                         "the summary spectrum figure)")
    ap.add_argument("--logx", action="store_true", help="log iteration axis (matrix view)")
    ap.add_argument("--logy", action="store_true", help="log mode-index axis (matrix view)")
    ap.add_argument("--vmin", type=float, default=-5.0,
                    help="colour floor for the matrix view (default -5; the preconditioned runs "
                         "reach ~1e-7, so -7 shows their tail)")
    ap.add_argument("--render_aspect", type=float, default=12.0,
                    help="force the drawn contour aspect ratio (e.g. 12); default fits the "
                         "trajectory and reports whatever ratio results")
    args = ap.parse_args()

    d = np.load(args.npz)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.npz))
    style()
    if args.matrix:
        arms = ["adam"] if ("skip_gd" in d.files and bool(d["skip_gd"])) else ["adam", "gd"]
        for w in arms:
            fig_matrix(d, out_dir, w, vmin=args.vmin, logx=args.logx, logy=args.logy,
                       smooth=args.matrix_smooth)
        return
    fig_contours(d, out_dir, args.render_aspect, args.which)
    fig_error(d, out_dir)
    fig_spectrum(d, out_dir, which="adam")
    fig_spectrum(d, out_dir, which="gd")
    fig_bands(d, out_dir)


if __name__ == "__main__":
    main()
