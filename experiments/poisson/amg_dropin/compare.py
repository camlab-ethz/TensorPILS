#!/usr/bin/env python
r"""Read the algebraic-V-cycle arm against the geometric one and the two anchors.

The reference arms are not re-run here: ``experiments/baselines/poisson`` already produced
``data`` (supervised), ``pls`` + GMG and ``galerkin`` (bare residual) at the same grid, data,
budget and seeds, so the only new arm is ``pls`` + AMG.

Prints the table and writes a two-panel figure: validation relative L2 against epoch (the curve
matters --- the bare-residual arm is still descending at 500 epochs, so "did not converge" and
"converged worse" are different claims and only the curve separates them), and the final test
error per arm.

    python experiments/poisson/amg_dropin/compare.py
"""
import argparse
import glob
import json
import os

import numpy as np

ARMS = [
    # label                       results glob                                          style
    ("$L_\\mathrm{data}$ (supervised)", "output/baselines/poisson/seed*/results/fno_data_K4_*.json",
     dict(color="0.35", ls="--")),
    ("$L_\\mathrm{PLS}$, geometric MG", "output/baselines/poisson/seed*/results/fno_pls_mg-*_K4_*.json",
     dict(color="tab:blue", ls="-")),
    ("$L_\\mathrm{PLS}$, algebraic MG", "output/poisson/amg_dropin/seed*/results/fno_pls_amg-*_K4_*.json",
     dict(color="tab:red", ls="-")),
    ("$L_\\mathrm{LS}$ (bare residual)", "output/baselines/poisson/seed*/results/fno_galerkin_K4_*.json",
     dict(color="tab:orange", ls=":")),
]


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/poisson/amg_dropin/amg_dropin.pdf")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    rows = []
    print(f"{'arm':34s} {'seeds':>5}  {'test rel-L2':>12}   {'range':>15}  {'best epoch':>10}")
    print("-" * 88)
    for label, pattern, style in ARMS:
        recs = load(pattern)
        if not recs:
            print(f"{label:34s} {'--':>5}  {'(no results yet)':>12}")
            continue
        e = np.array([r["test_rl2"] for r in recs])
        be = [r["stats"]["best_epoch"] for r in recs]
        print(f"{label:34s} {len(e):5d}  {e.mean()*100:11.2f}%   "
              f"{e.min()*100:6.2f}-{e.max()*100:6.2f}%  {min(be):4d}-{max(be):4d}")
        rows.append((label, recs, style, e))

    amg = next((r for r in rows if "algebraic" in r[0]), None)
    gmg = next((r for r in rows if "geometric" in r[0]), None)
    if amg and gmg:
        ratio = amg[3].mean() / gmg[3].mean()
        # "Within the seed spread" is the honest bar here: the GMG arm's own seed-to-seed range
        # is the resolution of this comparison, so a difference smaller than it is not a result.
        spread = (gmg[3].max() - gmg[3].min()) / gmg[3].mean()
        print(f"\nalgebraic / geometric = {ratio:.2f}x   "
              f"(geometric arm's own seed spread: {spread*100:.0f}% of its mean)")
        print("VERDICT: " + ("indistinguishable within the seed spread"
                             if abs(ratio - 1) <= spread else
                             f"a real {abs(ratio-1)*100:.0f}% {'gap' if ratio > 1 else 'gain'} "
                             f"-- check the curve and the learning rate before concluding"))

    if args.no_plot or not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.5, 3.6), width_ratios=[2, 1])
    for label, recs, style, e in rows:
        curves = np.array([r["stats"]["val_rel_l2_errors"] for r in recs])
        m = curves.mean(0)
        ax.plot(np.arange(1, len(m) + 1), m * 100, label=label, lw=1.6, **style)
        ax.fill_between(np.arange(1, len(m) + 1), curves.min(0) * 100, curves.max(0) * 100,
                        alpha=0.18, color=style["color"], lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation relative $L^2$  [%]")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3, which="both", lw=0.4)

    labels = [r[0] for r in rows]
    means = [r[3].mean() * 100 for r in rows]
    errs = [[m - r[3].min() * 100 for m, r in zip(means, rows)],
            [r[3].max() * 100 - m for m, r in zip(means, rows)]]
    bx.barh(range(len(rows)), means, xerr=errs, color=[r[2]["color"] for r in rows],
            alpha=0.85, height=0.6)
    bx.set_yticks(range(len(rows)))
    bx.set_yticklabels([l.replace("$L_\\mathrm{", "").replace("}$", "") for l in labels],
                       fontsize=7)
    bx.set_xscale("log")
    bx.set_xlabel("test relative $L^2$  [%]")
    bx.grid(alpha=0.3, axis="x", which="both", lw=0.4)
    bx.invert_yaxis()

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nfigure -> {args.out}")


if __name__ == "__main__":
    main()
