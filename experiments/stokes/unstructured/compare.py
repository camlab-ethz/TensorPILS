#!/usr/bin/env python
r"""Stokes past an obstacle against the same losses on the structured grid.

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

    python experiments/stokes/unstructured/compare.py
    python experiments/stokes/unstructured/compare.py --lr_sweep     # stage-1 table only
"""
import argparse
import glob
import json
import os

import numpy as np

# (loss label, results glob, style) per architecture. The globs are deliberately loose on the
# preconditioner tag so a change of V-cycle settings does not silently drop a row.
# Per loss: (label, structured glob, unstructured glob, colour). Explicit rather than one
# wildcard, because the structured runs sit at two different depths -- final/arm/cell/seed for
# the two table arms, diagnostics/arm/cell for the bare-residual control, which was never a
# candidate -- and a pattern loose enough for both would also sweep up the lr-sweep cells and
# silently inflate the seed counts.
_SQ = "output/stokes/stokes_paper/stokes_benchmark"
_UN = "output/stokes/unstructured"
LOSSES = [
    ("$L_\\mathrm{data}$ (supervised)",
     f"{_SQ}/final/data/lr3e-4/seed*/results/fno_stokes_data_mu1_gr65_K10_*.json",
     f"{_UN}/seed*/results/gaot_stokes_data_mu1_obstacle-n*_K10_*.json", "0.35"),
    ("$L_\\mathrm{PLS}$ (preconditioned)",
     f"{_SQ}/final/pls/lr1e-3_om256/seed*/results/fno_stokes_pls_*w256_mu1_gr65_K10_*.json",
     f"{_UN}/seed*/results/gaot_stokes_pls_amg-*w16_mu1_obstacle-n*_K10_*.json", "tab:blue"),
    ("$L_\\mathrm{LS}$ (bare residual)",
     f"{_SQ}/diagnostics/galerkin/*/results/fno_stokes_galerkin_mu1_gr65_K10_*.json",
     f"{_UN}/seed*/results/gaot_stokes_galerkin_mu1_obstacle-n*_K10_*.json", "tab:orange"),
    # The physics-informed baseline that CAN follow off the grid: its residual is autodiff
    # through a coordinate trunk, not a finite-difference stencil. PINO has no unstructured
    # row at all, which is the point -- but an absent competitor proves nothing, so this one
    # is what makes the unstructured column a comparison.
    ("PI-DeepONet (strong form, autodiff)",
     f"{_SQ}/final/pideeponet/*/seed*/results/deeponet_stokes_pi-*_mu1_gr65_K10_*.json",
     f"{_UN}/seed*/results/deeponet_stokes_pi-*_mu1_obstacle-n*_K10_*.json", "tab:purple"),
]
MESHES = [("square", 1, dict(ls="--", lw=1.4)), ("obstacle", 2, dict(ls="-", lw=1.8))]
LR_GLOB = "output/stokes/unstructured/lr/*/results/gaot_stokes_*.json"


def load(pattern):
    """``glob`` with brace expansion, so one pattern can span final/ and diagnostics/."""
    import itertools, re
    m = re.search(r"\{([^}]*)\}", pattern)
    pats = ([pattern.replace(m.group(0), alt) for alt in m.group(1).split(",")]
            if m else [pattern])
    recs = []
    for f in sorted(itertools.chain.from_iterable(glob.glob(pt) for pt in pats)):
        with open(f) as fh:
            r = json.load(fh)
        r["_path"] = f
        recs.append(r)
    return recs


def _best_val(rec):
    """Best validation error over training (the selection metric): the mean of the two fields."""
    series = rec["stats"].get("val_rel_l2_errors") or []
    return min(series) if series else float("nan")


def _test_errors(rec):
    """``(velocity, pressure)`` test relative FE-L2, as fractions."""
    return rec["test_rel_l2_u"], rec["test_rel_l2_p"]


def lr_table():
    """Stage 1: best validation error per swept cell. Pick the minimum of each block."""
    print(f"{'arm':34s} {'cell':>14}  {'best val (u+p)/2':>17}  {'epoch':>6}")
    print("-" * 78)
    found = False
    for entry in LOSSES:
        label = entry[0]
        rows = []
        for rec in load(LR_GLOB):
            cell = os.path.basename(os.path.dirname(os.path.dirname(rec["_path"])))
            arm = cell.split("_")[0]
            if arm not in label.lower() and not (arm == "pls" and "PLS" in label) \
                    and not (arm == "data" and "data" in label):
                continue
            rows.append((cell, _best_val(rec), rec["stats"]["best_epoch"]))
        if not rows:
            print(f"{label:34s} {'--':>14}  {'(no results yet)':>17}")
            continue
        found = True
        best = min(rows, key=lambda r: r[1])
        for cell, v, ep in sorted(rows):
            mark = "  <-- pick" if (cell, v, ep) == best else ""
            print(f"{label:34s} {cell:>14}  {v * 100:16.3f}%  {ep:6d}{mark}")
        print()
    if not found:
        print("\nNothing to read yet -- run "
              "`clsubmit experiments/stokes/unstructured/lr_sweep.txt`.")


def final_table():
    print(f"{'loss':32s} {'mesh':>9} {'seeds':>5}   {'velocity':>17}   {'pressure':>17}   "
          f"{'best epoch':>10}")
    print("-" * 104)
    curves = []
    for entry in LOSSES:
        label, color = entry[0], entry[3]
        for mesh, slot, style in MESHES:
            recs = load(entry[slot])
            if not recs:
                print(f"{label:32s} {mesh:>9} {'--':>5}   {'(no results yet)':>17}")
                continue
            eu = np.array([_test_errors(r)[0] for r in recs])
            ep = np.array([_test_errors(r)[1] for r in recs])
            be = [r["stats"]["best_epoch"] for r in recs]
            print(f"{label:32s} {mesh:>9} {len(eu):5d}   {eu.mean()*100:7.2f} +- "
                  f"{eu.std()*100:5.2f} %   {ep.mean()*100:7.2f} +- {ep.std()*100:5.2f} %   "
                  f"{min(be):4d}-{max(be):4d}")
            curves.append((f"{mesh}, {label}", recs, dict(color=color, **style)))
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
    means = [np.mean([_test_errors(r)[0] for r in c[1]]) * 100 for c in curves]
    errs = [np.std([_test_errors(r)[0] for r in c[1]]) * 100 for c in curves]
    # The FNO rows are drawn hollow so the eye groups by loss (colour) first, not by model.
    ax_bar.barh(range(len(names)), means, xerr=errs,
                color=["none" if n.startswith("square") else c[2].get("color", "0.5")
                       for n, c in zip(names, curves)],
                edgecolor=[c[2].get("color", "0.5") for c in curves], lw=1.4)
    ax_bar.set_yticks(range(len(names)))
    ax_bar.set_yticklabels(names, fontsize=7)
    ax_bar.set_xscale("log")
    ax_bar.set_xlabel("test velocity relative FE-$L^2$ [%]")
    ax_bar.set_title("Final test error (mean $\\pm$ sd over seeds)")
    ax_bar.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/stokes/unstructured/obstacle_vs_square.pdf")
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
        print("Nothing to read yet -- run `clsubmit experiments/stokes/unstructured/sweep.txt`.")


if __name__ == "__main__":
    main()
