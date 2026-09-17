#!/usr/bin/env python
r"""Poisson on the unstructured disc against the same losses on the structured grid.

The structured GAOT arms are not re-run: ``experiments/poisson/gaot_arch`` already produced
``data``, ``galerkin`` and ``pls`` at the same source distribution, budget and seeds, so the only
new thing here is the geometry and the comparison is a read.

Read the table **down a loss, not across one**. The two rows of a loss are on different domains
with different labels (the disc has no closed form), so a like-for-like absolute comparison is
not available -- what transfers, or fails to, is the *ranking* of the losses within each column.

Two questions, two views:

* **Does the loss ranking transfer?** The table groups by loss, so ``pls`` vs ``galerkin``
  within GAOT is read off adjacent rows. That comparison is the claim being tested. The
  architecture-to-architecture gap is readable too and is close to a fair one: at the CLI
  defaults GAOT has 3,396,033 trainable parameters against the FNO's 3,008,417, a 13 %
  difference — not matched on purpose, but near enough that a large gap could not be explained
  by capacity alone.
* **Did it converge, or just stop?** The left panel is validation relative L2 against epoch. On
  the FNO the bare-residual arm is still descending at 500 epochs, so "did not converge" and
  "converged worse" are different claims, and only the curve separates them.

    python experiments/poisson/unstructured/compare.py
    python experiments/poisson/unstructured/compare.py --lr_sweep     # stage-1 table only
"""
import argparse
import glob
import json
import os

import numpy as np

# (loss label, results glob, style) per architecture. The globs are deliberately loose on the
# preconditioner tag so a change of V-cycle settings does not silently drop a row.
ARCHES = {
    "square": ("output/poisson/gaot_arch/seed*/results/gaot_{stem}.json", dict(ls="--", lw=1.4)),
    "disc": ("output/poisson/unstructured/seed*/results/gaot_{stem}_circle-n*.json",
             dict(ls="-", lw=1.8)),
}
LOSSES = [
    ("$L_\\mathrm{data}$ (supervised)", "data_K4_*", "0.35"),
    ("$L_\\mathrm{PLS}$ (preconditioned)", "pls_*_K4_*", "tab:blue"),
    ("$L_\\mathrm{LS}$ (bare residual)", "galerkin_K4_*", "tab:orange"),
]
LR_GLOB = "output/poisson/unstructured/lr/lr*/results/gaot_{stem}_circle-n*.json"


def load(pattern):
    recs = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            r = json.load(fh)
        r["_path"] = f
        recs.append(r)
    return recs


def _best_val(rec):
    """Best validation relative L2 over training (the selection metric), as a fraction."""
    series = rec["stats"].get("val_rel_l2_errors") or []
    return min(series) if series else float("nan")


def lr_table():
    """Stage 1: best validation relative L2 per (loss, lr). Pick the minimum of each block."""
    print(f"{'arm':30s} {'lr':>8}  {'best val rel-L2':>16}  {'epoch':>6}")
    print("-" * 66)
    any_found = False
    for label, stem, _ in LOSSES:
        rows = []
        for rec in load(LR_GLOB.format(stem=stem)):
            # .../lr/lr3e-4/results/....json  ->  3e-4
            lr = os.path.basename(os.path.dirname(os.path.dirname(rec["_path"])))[2:]
            rows.append((lr, _best_val(rec), rec["stats"]["best_epoch"]))
        if not rows:
            print(f"{label:30s} {'--':>8}  {'(no results yet)':>16}")
            continue
        any_found = True
        best = min(rows, key=lambda r: r[1])
        for lr, v, ep in sorted(rows, key=lambda r: float(r[0])):
            mark = "  <-- pick" if (lr, v, ep) == best else ""
            print(f"{label:30s} {lr:>8}  {v * 100:15.3f}%  {ep:6d}{mark}")
        print()
    if not any_found:
        print("\nNothing to read yet -- run `clsubmit experiments/poisson/unstructured/sweep.txt`.")


def final_table():
    print(f"{'loss':32s} {'mesh':>7} {'seeds':>6}  {'test rel-L2':>12}   {'range':>15}  "
          f"{'best epoch':>10}")
    print("-" * 92)
    curves = []
    for label, stem, color in LOSSES:
        for arch, (pattern, style) in ARCHES.items():
            recs = load(pattern.format(stem=stem))
            if not recs:
                print(f"{label:32s} {arch:>7} {'--':>6}  {'(no results yet)':>12}")
                continue
            e = np.array([r["test_rl2"] for r in recs])
            be = [r["stats"]["best_epoch"] for r in recs]
            print(f"{label:32s} {arch:>7} {len(e):6d}  {e.mean() * 100:11.2f}%   "
                  f"{e.min() * 100:6.2f}-{e.max() * 100:6.2f}%  {min(be):4d}-{max(be):4d}")
            curves.append((f"{arch}, {label}", recs, dict(color=color, **style)))
        print()
    return curves


def plot(curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, recs, style in curves:
        # Seeds share an epoch axis; plot the median and shade the spread.
        series = [r["stats"]["val_rel_l2_errors"] for r in recs]
        n = min(len(s) for s in series)
        arr = np.array([s[:n] for s in series]) * 100
        ep = np.arange(1, n + 1)
        ax_curve.plot(ep, np.median(arr, axis=0), label=name, **style)
        if arr.shape[0] > 1:
            ax_curve.fill_between(ep, arr.min(axis=0), arr.max(axis=0),
                                  color=style.get("color"), alpha=0.15, lw=0)
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("epoch")
    ax_curve.set_ylabel("validation relative $L^2$ [%]")
    ax_curve.set_title("Convergence (median over seeds, min-max shaded)")
    ax_curve.legend(fontsize=7)
    ax_curve.grid(alpha=0.3)

    names = [c[0] for c in curves]
    means = [np.mean([r["test_rl2"] for r in c[1]]) * 100 for c in curves]
    errs = [np.std([r["test_rl2"] for r in c[1]]) * 100 for c in curves]
    # The FNO rows are drawn hollow so the eye groups by loss (colour) first, not by model.
    ax_bar.barh(range(len(names)), means, xerr=errs,
                color=["none" if n.startswith("square") else c[2].get("color", "0.5")
                       for n, c in zip(names, curves)],
                edgecolor=[c[2].get("color", "0.5") for c in curves], lw=1.4)
    ax_bar.set_yticks(range(len(names)))
    ax_bar.set_yticklabels(names, fontsize=7)
    ax_bar.set_xscale("log")
    ax_bar.set_xlabel("test relative $L^2$ [%]")
    ax_bar.set_title("Final test error (mean $\\pm$ sd over seeds)")
    ax_bar.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/poisson/unstructured/disc_vs_square.pdf")
    ap.add_argument("--lr_sweep", action="store_true", help="stage-1 learning-rate table only")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    if args.lr_sweep:
        lr_table()
        return

    curves = final_table()
    if curves and not args.no_plot:
        plot(curves, args.out)
    elif not curves:
        print("Nothing to read yet -- run `clsubmit experiments/poisson/unstructured/sweep.txt`.")


if __name__ == "__main__":
    main()
