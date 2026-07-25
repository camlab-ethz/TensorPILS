"""Plot the measured Stokes conditioning against the mesh, with fitted exponents.

Reads ``cond_gr*.json`` from :mod:`measure_conditioning` and draws every quantity on one
log-log axis against ``1/h``, annotating each with the least-squares exponent ``s`` in
``kappa ~ h^-s``. The point of the figure is that three lines are parallel to a reference slope:
``kappa(K)`` and ``kappa(P)`` to ``h^-2``, ``kappa(K^2)`` to ``h^-4``, while the eigenvalue ratio
``max|lam(PK)|/min|lam(PK)|`` is flat — the O(1) assumption the whole preconditioning argument
rests on.

Usage:
    python experiments/stokes/conditioning/plot_conditioning.py \
        --results_dir output/stokes/conditioning
"""

import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = [
    ("cond_K2",    r"$\kappa(\mathcal{K}^2)$ — $G$ of $L_{\rm LS}$",        "#b91c1c", "^", "-"),
    ("cond_naive", r"$\kappa((\mathcal{PK})^\top\mathcal{PK})$ — $G$ of $L_{\rm naive}$",
     "#b45309", "v", "-"),
    ("cond_KPK",   r"$\kappa(\mathcal{KPK})$ — $G$ of $L_{\rm PLS}$",       "#0e7490", "s", "-"),
    ("cond_K",     r"$\kappa(\mathcal{K})$",                                "#6d28d9", "o", "--"),
    ("cond_P",     r"$\kappa(\mathcal{P})$",                                "#0f7b4f", "D", "--"),
    ("ratio_PK",   r"$\max|\lambda(\mathcal{PK})|/\min|\lambda(\mathcal{PK})|$",
     "#111111", "*", ":"),
]


def slope(xs, ys):
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(xs)
    mx, my = sum(lx) / n, sum(ly) / n
    den = sum((a - mx) ** 2 for a in lx)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results_dir", default="output/stokes/conditioning")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "cond_gr*.json"))):
        with open(path) as fh:
            rows.append(json.load(fh))
    if not rows:
        raise SystemExit(f"no cond_gr*.json in {args.results_dir}")
    rows.sort(key=lambda r: r["grid"])
    inv_h = [1.0 / r["h"] for r in rows]
    print("grids:", [r["grid"] for r in rows], " 1/h:", [round(x, 2) for x in inv_h])

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for key, label, color, marker, ls in SERIES:
        ys = [r.get(key) for r in rows]
        if any(y is None for y in ys):
            continue
        s = slope(inv_h, ys) if len(rows) > 1 else float("nan")
        ax.plot(inv_h, ys, ls, color=color, marker=marker, ms=6.5, lw=1.7,
                label=f"{label}   $s={s:.2f}$")
        print(f"{key:12s} s={s:+.3f}   " + "  ".join(f"{y:.3e}" for y in ys))

    # reference slopes, anchored at the left edge so they are visually comparable
    x0, x1 = inv_h[0], inv_h[-1]
    for p, style in ((2, (0, (6, 4))), (4, (0, (2, 3)))):
        y0 = rows[0]["cond_K"] if p == 2 else rows[0]["cond_K2"]
        ax.plot([x0, x1], [y0, y0 * (x1 / x0) ** p], color="#999999", lw=1.0, ls=style)
        ax.annotate(rf"$h^{{-{p}}}$", xy=(x1, y0 * (x1 / x0) ** p), fontsize=9,
                    color="#777777", xytext=(4, -2), textcoords="offset points")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$1/h$   (pressure-grid element size $h=1/(n_p-1)$)")
    ax.set_ylabel("condition number")
    ax.set_title("Stokes: measured conditioning of the Gauss--Newton matrices (float64)")
    ax.set_xticks(inv_h)
    ax.set_xticklabels([f"{x:g}\n${r['grid']}^2$" for x, r in zip(inv_h, rows)])
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(0.01, 0.62))
    fig.tight_layout()
    out = os.path.join(args.results_dir, "conditioning.png")
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")

    if "exact_ratio_PK" in rows[0]:
        print("\nexact-block validation (theory: 3 eigenvalues, ratio 2.618034):")
        for r in rows:
            print(f"  {r['grid']:4d}^2  ratio={r['exact_ratio_PK']:.6f}  "
                  f"distinct={[round(v, 5) for v in r['exact_distinct']]}")


if __name__ == "__main__":
    main()
