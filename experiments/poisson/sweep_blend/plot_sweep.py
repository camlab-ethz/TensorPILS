"""Aggregate a preconditioner-strength sweep into two figures.

Reads the per-run ``results/*.json`` written by ``Trainer._save_results_json`` and produces:

  1. overlay  — validation relative-L2 vs epoch, one curve per strength (the
                "physics-informed -> supervised transition" figure);
  2. collapse — final relative-L2 vs conditioning: kappa(H)=kappa(PA)^2 for the squared PLS
                loss, kappa(PA) for the preconditioned Deep Ritz surrogate (whose descent
                direction is P*r). The loss form is auto-detected from each run's loss_type.

Both figures also carry a multigrid (GMG) run when present -- the preconditioner that works
in practice. It has no single closed-form conditioning, so it is drawn as a reference curve on
the overlay (dashed black) and omitted from the collapse axis.

Usage (after pulling the sweep output back from the cluster):
    python experiments/poisson/sweep_blend/plot_sweep.py \
        --results_dir output/poisson/sweep_blend/pls/results   # or .../deepritz/results
    # optional: --smooth (centered moving average) / --no-annotate

Flags:
    --smooth / --no-smooth      smooth the stochastic curves with a centered moving average
                                (raw curve kept faint underneath). Default off.
    --smooth_window N           moving-average window in epochs (default 15).
    --annotate / --no-annotate  stamp the loss/gradient formulas on each figure. Default on.
"""

import argparse
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt


def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as fh:
            runs.append(json.load(fh))
    # sort by preconditioner strength when available (NaN -> last)
    runs.sort(key=lambda r: (r.get("precond_strength") if r.get("precond_strength") == r.get("precond_strength") else -1.0))
    return runs


def is_multigrid(run):
    return (run.get("precond_kind") or run.get("stats", {}).get("precond_kind")) == "multigrid"


def conditioning(run):
    """(value, latex symbol) for the collapse axis / legend, per the loss form.

    Deep Ritz preconditions the descent direction ``P*r``, so its natural conditioning is
    ``kappa(PA)``; the squared PLS loss squares it into the Hessian, ``kappa(H)=kappa(PA)^2``.
    """
    if run.get("loss_type") == "deepritz":
        return run["stats"].get("precond_cond_pa", float("nan")), r"\kappa(PA)"
    return run["stats"].get("precond_cond_h", float("nan")), r"\kappa(H)"


def loss_label(runs):
    return "Deep Ritz" if runs and runs[0].get("loss_type") == "deepritz" else "PLS"


def run_label(run):
    """Legend label: the multigrid reference is named explicitly; blend runs carry kappa."""
    if is_multigrid(run):
        return "multigrid (GMG)"
    kval, ksym = conditioning(run)
    return f"t={run['precond_strength']:.2f} (${ksym}$={kval:.1e})"


def smooth_curve(y, window):
    """Centered simple moving average (numpy only), edge-padded to preserve length.

    Chosen over EMA (TensorBoard-style) because a centered mean introduces no phase lag or
    warm-up bias -- important when the figure's whole point is *where* a curve converges.
    """
    y = np.asarray(y, dtype=float)
    if window <= 1 or y.size < 2:
        return y
    w = min(int(window), y.size)
    if w % 2 == 0:                       # force odd so the window is symmetric about each point
        w -= 1
    if w <= 1:
        return y
    pad = w // 2
    ypad = np.pad(y, pad, mode="edge")   # 'nearest' edge handling
    return np.convolve(ypad, np.ones(w) / w, mode="valid")


def annotation_text(runs):
    r"""The loss/(surrogate-)gradient identity stamped on the figures, per loss form."""
    if loss_label(runs) == "Deep Ritz":
        return "\n".join([
            r"$L(c)=\frac{1}{2}\,c^{\top}Ac-b^{\top}c$",
            r"$\tilde{\nabla}L(c)=P_t\,[\,Ac-b\,]$",
            r"$P_t=(1-t)\,I+t\,A^{-1}$",
            r"($\tilde{\nabla}$: surrogate descent, not $\nabla L$)",
        ])
    return "\n".join([
        r"$L(c)=\frac{1}{2}\,\|P_t(Ac-b)\|^2$",
        r"$\nabla L(c)=A\,P_t^{2}\,(Ac-b)$",
        r"$P_t=(1-t)\,I+t\,A^{-1}$",
    ])


def stamp_annotation(ax, runs, loc=("left", "bottom")):
    ha, va = loc
    x = 0.03 if ha == "left" else 0.97
    y = 0.03 if va == "bottom" else 0.97
    ax.text(x, y, annotation_text(runs), transform=ax.transAxes, ha=ha, va=va,
            fontsize=9, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.9))


def save_figure(fig, out_path, dpi=150):
    """Write both a raster and a vector copy of ``out_path``, returning the PDF path.

    The ``.png`` is for quick viewing and the working notes in ``notes/paper_story/``; the
    ``.pdf`` is what the paper includes. Two reasons the vector copy matters: it stays sharp
    when a reviewer zooms, and ``*.png`` is git-ignored repo-wide (``.gitignore``), so a raster
    figure dropped under ``paper/figures/`` would never be committed and the paper would fail
    to build on a fresh clone or in Overleaf.
    """
    fig.savefig(out_path, dpi=dpi)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path)
    return pdf_path


def plot_overlay(runs, out_path, smooth=False, window=15, annotate=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in runs:
        rl2 = r["stats"]["val_rel_l2_errors"]
        if not rl2:
            continue
        x = range(1, len(rl2) + 1)
        mg = is_multigrid(r)
        style = dict(color="black", ls="--") if mg else {}
        if smooth:
            ys = smooth_curve(rl2, window)
            base, = ax.plot(x, rl2, lw=0.8, alpha=0.2, **style)
            ax.plot(x, ys, lw=2.4 if mg else 1.8, label=run_label(r),
                    color=base.get_color(), ls=style.get("ls", "-"))
        else:
            ax.plot(x, rl2, lw=2.4 if mg else 1.6, label=run_label(r), **style)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation relative $L^2$")
    ax.set_yscale("log")
    ax.set_title(f"{loss_label(runs)}: convergence vs preconditioner strength $t$")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    if annotate:
        stamp_annotation(ax, runs, loc=("left", "bottom"))
    fig.tight_layout()
    print(f"overlay  -> {out_path}\n         -> {save_figure(fig, out_path)}")


def plot_collapse(runs, out_path, annotate=True):
    sym = conditioning(runs[0])[1]         # uniform per results dir (one loss form)
    kappa, final_rl2, strengths = [], [], []
    for r in runs:
        rl2 = r["stats"]["val_rel_l2_errors"]
        kval, sym = conditioning(r)
        if not rl2 or kval != kval:         # skip empty / NaN-conditioning runs (e.g. multigrid)
            continue
        kappa.append(kval)
        final_rl2.append(min(rl2))          # best achieved
        strengths.append(r["precond_strength"])

    if not kappa:                           # only multigrid / NaN-conditioning runs present so far
        print(f"collapse -> skipped (no blend runs with finite conditioning in {out_path})")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(kappa, final_rl2, "o-", lw=1.6)
    for k, e, t in zip(kappa, final_rl2, strengths):
        ax.annotate(f"t={t:.2f}", (k, e), fontsize=8,
                    textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel(rf"conditioning ${sym}$")
    ax.set_ylabel("best validation relative $L^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{loss_label(runs)} collapse: final error vs conditioning")
    ax.grid(True, which="both", alpha=0.3)
    if annotate:
        stamp_annotation(ax, runs, loc=("right", "top"))
    fig.tight_layout()
    print(f"collapse -> {out_path}\n         -> {save_figure(fig, out_path)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="output/poisson/sweep_blend/pls/results")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write the figures (default: alongside results_dir).")
    ap.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=False,
                    help="Smooth curves with a centered moving average (default off).")
    ap.add_argument("--smooth_window", type=int, default=15,
                    help="Moving-average window in epochs (default 15).")
    ap.add_argument("--annotate", action=argparse.BooleanOptionalAction, default=True,
                    help="Stamp the loss/gradient formulas on the figures (default on).")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No results found in {args.results_dir}")
    print(f"Loaded {len(runs)} runs.")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results_dir))
    os.makedirs(out_dir, exist_ok=True)
    plot_overlay(runs, os.path.join(out_dir, "sweep_overlay.png"),
                 smooth=args.smooth, window=args.smooth_window, annotate=args.annotate)
    plot_collapse(runs, os.path.join(out_dir, "sweep_collapse.png"), annotate=args.annotate)


if __name__ == "__main__":
    main()
