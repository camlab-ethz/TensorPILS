"""Paper-ready versions of the blend-sweep overlay and collapse figures.

The exploratory renders live in ``plot_sweep.py``; this is the deliverable, authored to the size
and type conventions established by ``experiments/graphical_abstract/plot_abstract.py``:

* **Smoothed curves only.** The raw per-epoch traces used to sit underneath at low alpha; with
  seven runs they overlapped into indistinguishable noise and carried no per-run information, so
  they are gone. Smoothing is a centered moving average -- no phase lag, which matters when the
  figure's point is *where* a curve converges.
* **Authored at the final size.** The paper includes these at ``width=0.49\\linewidth`` side by
  side, and the ICLR text width is 5.5 in, so the canvas is exactly 0.49 x 5.5 = 2.695 in and
  ``\\includegraphics`` does not rescale. The point sizes below are the ones that reach the page.
* **Type near body size.** Body text is 10 pt; axis labels are 9 pt and ticks 7.5 pt, matching the
  graphical abstract so the two figures look like they belong to the same paper.
* **No titles, no formula box.** The caption carries both.
* **Serif, Times-first**, to match the ICLR body font.

Two departures from ``plot_sweep.py`` worth knowing:

* **Colour encodes ``t`` sequentially** (magma, truncated) rather than by the categorical cycle.
  ``t`` is an ordered parameter, so a ramp reads as an ordering; ten arbitrary hues do not.
* **The collapse panel reports TEST error** at the best-validation checkpoint (``test_rl2``, stored
  per run). The overlay cannot: per-epoch histories exist for validation only.
* **The overlay legend drops the kappa values.** They do not fit at 2.7 in, and the right-hand
  panel *is* the kappa axis -- so nothing is lost, it moves.

The multigrid run has no finite kappa(H), so on the collapse panel it becomes a horizontal
reference line rather than a point: the error the practical preconditioner reaches, against which
the blend curve can be read.

Usage:
    python experiments/poisson/sweep_blend/plot_paper.py
"""

import argparse
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt

# The subset shown in the paper. Dropping t = 0.50, 0.90, 0.99 removes three curves that lie on
# top of t = 0.75 and t = 1.00 and crowd the collapse panel's left end without adding information.
WANT_T = [0.0, 0.01, 0.05, 0.10, 0.25, 0.75, 1.0]

FS_LAB, FS_TICK, FS_LEG, FS_ANN = 9.0, 7.5, 6.5, 6.5
MUTED, TEXT = "#6b6b6b", "#1a1a1a"
W_IN = 0.49 * 5.5                # 2.695 in -- exactly what \includegraphics receives
H_IN = 2.25


def is_multigrid(run):
    return (run.get("precond_kind") or run.get("stats", {}).get("precond_kind")) == "multigrid"


def load(results_dir):
    """-> (blend runs sorted by t and filtered to WANT_T, the multigrid run or None)."""
    blends, mg = [], None
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as fh:
            run = json.load(fh)
        if is_multigrid(run):
            mg = run
            continue
        t = run.get("precond_strength")
        if t is None or not any(abs(t - w) < 1e-9 for w in WANT_T):
            continue
        blends.append(run)
    blends.sort(key=lambda r: r["precond_strength"])
    return blends, mg


def t_label(t):
    """'t=0', 't=0.01', 't=1' -- trailing zeros stripped so the legend stays narrow."""
    s = f"{t:.2f}".rstrip("0").rstrip(".")
    return f"$t={s}$"


def smooth(y, window=15):
    """Centered moving average, edge-padded. No phase lag, unlike an EMA -- which matters when
    the figure's point is *where* a curve converges."""
    y = np.asarray(y, dtype=float)
    w = min(int(window), y.size)
    if w % 2 == 0:
        w -= 1
    if w <= 1:
        return y
    return np.convolve(np.pad(y, w // 2, mode="edge"), np.ones(w) / w, mode="valid")


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": MUTED, "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    })


def finish(ax):
    ax.grid(True, which="major", color=MUTED, alpha=0.22, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def save(fig, out_dir, stem, dpi):
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, stem + ".pdf")
    png = os.path.join(out_dir, stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    print(f"  {pdf}\n  {png}")


# Magma, as in the graphical abstract -- but truncated. The abstract uses the full range on a
# pcolormesh, where the pale top sits on the image; for lines on white paper magma's high end
# (#FCFDBF) is invisible and its low end (#000004) collides with the dashed black GMG curve.
# [0.15, 0.85] keeps the family look and stays legible at both ends.
CMAP_LO, CMAP_HI = 0.15, 0.85


def colours(n):
    cmap = plt.get_cmap("magma")
    return [cmap(CMAP_LO + (CMAP_HI - CMAP_LO) * i / max(n - 1, 1)) for i in range(n)]


def plot_overlay(blends, mg, out_dir, dpi, window):
    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    cols = colours(len(blends))
    for c, r in zip(cols, blends):
        y = r["stats"]["val_rel_l2_errors"]
        if not y:
            continue
        x = np.arange(1, len(y) + 1)
        ax.plot(x, smooth(y, window), lw=1.4, color=c, label=t_label(r["precond_strength"]))
    if mg:
        y = mg["stats"]["val_rel_l2_errors"]
        x = np.arange(1, len(y) + 1)
        ax.plot(x, smooth(y, window), lw=1.6, color="black", ls="--", label="GMG")

    ax.set_yscale("log")
    ax.set_xlabel("epoch", fontsize=FS_LAB, labelpad=1.5)
    ax.set_ylabel(r"validation relative $L^2$", fontsize=FS_LAB, labelpad=2)
    # White backing, borderless: the converged bundle passes close to this corner, and a solid
    # ground under the key keeps it readable without adding a visible box.
    leg = ax.legend(loc="lower left", fontsize=FS_LEG, ncol=2,
                    handlelength=1.3, columnspacing=0.9, labelspacing=0.22, handletextpad=0.45,
                    borderaxespad=0.15, borderpad=0.3,
                    frameon=True, framealpha=0.88, edgecolor="none", facecolor="white")
    leg.get_frame().set_linewidth(0.0)
    finish(ax)
    fig.tight_layout(pad=0.3)
    save(fig, out_dir, "sweep_overlay", dpi)


def plot_collapse(blends, mg, out_dir, dpi):
    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    k = np.array([r["stats"]["precond_cond_h"] for r in blends])
    # test_rl2 is the TEST error of the best-validation checkpoint: selection on validation,
    # reporting on test. Already stored per run, so no recomputation. The overlay panel cannot
    # do the same -- per-epoch histories exist for validation only.
    e = np.array([r["test_rl2"] for r in blends])
    ts = [r["precond_strength"] for r in blends]
    cols = colours(len(blends))

    ax.plot(k, e, "-", lw=1.0, color=MUTED, zorder=1)
    ax.scatter(k, e, s=16, c=cols, zorder=3, edgecolors="white", linewidths=0.4)

    if mg:
        ymg = mg["test_rl2"]
        ax.axhline(ymg, ls="--", lw=1.0, color="black", alpha=0.85, zorder=2)
        # Right end of the line: the left end is where the flat blend points sit.
        ax.annotate("GMG", xy=(0.985, ymg), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points",
                    fontsize=FS_ANN, ha="right", va="bottom")

    ax.set_xscale("log"); ax.set_yscale("log")
    # Headroom so the t=0 point (top right) is not clipped and its label stays inside.
    ax.set_ylim(e.min() * 0.55, e.max() * 2.2)
    ax.set_xlim(k.min() * 0.35, k.max() * 4.0)
    # Label only the two endpoints. Labelling all seven put text on top of the GMG line and on
    # top of neighbouring labels at this size; colour already encodes t, and the overlay panel
    # beside it carries the legend that decodes the ramp. (Selective direct labels, not one
    # number per point.)
    i_lo, i_hi = int(np.argmin(k)), int(np.argmax(k))
    ax.annotate(t_label(ts[i_lo]), (k[i_lo], e[i_lo]), fontsize=FS_ANN,
                textcoords="offset points", xytext=(0, -10), ha="center", va="top")
    ax.annotate(t_label(ts[i_hi]), (k[i_hi], e[i_hi]), fontsize=FS_ANN,
                textcoords="offset points", xytext=(-5, -1), ha="right", va="center")
    ax.set_xlabel(r"conditioning $\kappa(H_t)$", fontsize=FS_LAB, labelpad=1.5)
    ax.set_ylabel(r"test relative $L^2$", fontsize=FS_LAB, labelpad=2)
    finish(ax)
    fig.tight_layout(pad=0.3)
    save(fig, out_dir, "sweep_collapse", dpi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="output/poisson/sweep_blend/pls/results")
    ap.add_argument("--out_dir", default="paper/figures")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--smooth_window", type=int, default=15)
    args = ap.parse_args()

    blends, mg = load(args.results_dir)
    if not blends:
        raise SystemExit(f"no blend runs matching {WANT_T} in {args.results_dir}")
    print(f"{len(blends)} blend runs (t = {[r['precond_strength'] for r in blends]})"
          f"{' + multigrid' if mg else ' (NO multigrid run found)'}")
    style()
    plot_overlay(blends, mg, args.out_dir, args.dpi, args.smooth_window)
    plot_collapse(blends, mg, args.out_dir, args.dpi)
    print(f"canvas {W_IN:.3f} x {H_IN} in  (= 0.49 x 5.5 in text width, no rescaling)")


if __name__ == "__main__":
    main()
