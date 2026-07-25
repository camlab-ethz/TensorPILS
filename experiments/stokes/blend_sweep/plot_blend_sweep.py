"""Aggregate the Stokes blend sweep into the overlay + collapse pair.

Mirrors ``experiments/poisson/sweep_blend/plot_sweep.py``, with one difference that matters: for
Poisson the conditioning of each blend has a closed form (the preconditioner is built from an
eigendecomposition), whereas here ``P_t`` contains a multigrid V-cycle and the conditioning has to
be **measured**. So this script joins the training runs against
``measure_blend_cond.py``'s output by ``t``:

  1. ``blend_overlay.png``  — validation relative-L2 vs epoch, one curve per ``t``, annotated with
     the measured ``kappa(K P_t K)``; the ``t=0`` curve is the bare least-squares loss and ``t=1``
     the block-preconditioned one, so the family spans exactly the two rows of the head-to-head;
  2. ``blend_collapse.png`` — final error against that conditioning, which is the claim itself:
     error should be monotone in ``kappa`` and nothing else about ``t`` should matter.

Usage:
    python experiments/stokes/blend_sweep/measure_blend_cond.py --grid 65     # x-axis first
    python experiments/stokes/blend_sweep/plot_blend_sweep.py \
        --results_dir output/stokes/blend_sweep/results
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as fh:
            r = json.load(fh)
        if r.get("pde") == "stokes" and r.get("loss_type") == "pls":
            t = r.get("precond_strength")
            runs.append((1.0 if t is None else float(t), r))
    runs.sort(key=lambda p: p[0])
    return runs


def load_cond(cond_path):
    if not cond_path or not os.path.exists(cond_path):
        return {}
    with open(cond_path) as fh:
        rec = json.load(fh)
    return {round(e["t"], 6): e for e in rec["entries"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results_dir", default="output/stokes/blend_sweep/results")
    ap.add_argument("--cond_json", default=None,
                    help="Default: <parent of results_dir>/cond_blend_gr<grid>.json")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no Stokes pls result JSONs in {args.results_dir}")
    out_dir = args.out_dir or os.path.dirname(args.results_dir.rstrip("/")) or "."
    grid = runs[0][1]["grid"][0]
    cond = load_cond(args.cond_json or os.path.join(out_dir, f"cond_blend_gr{grid}.json"))
    if not cond:
        print("! no conditioning file found — the collapse plot needs measure_blend_cond.py first")

    colours = cm.viridis([i / max(len(runs) - 1, 1) for i in range(len(runs))])

    # ---------------- overlay ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for (t, r), c in zip(runs, colours):
        k = cond.get(round(t, 6), {}).get("cond_KPK")
        lbl = f"t={t:g}" + (rf"  ($\kappa$={k:.1e})" if k else "")
        for ax, key in ((axes[0], "val_rel_l2_u"), (axes[1], "val_rel_l2_p")):
            y = [100.0 * v for v in r["stats"][key]]
            ax.plot(range(1, len(y) + 1), y, lw=1.4, color=c, label=lbl)
    for ax, name in ((axes[0], "velocity"), (axes[1], "pressure")):
        ax.axhline(100.0, color="#888888", ls=":", lw=1.1)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"validation relative FE-$L^2$, {name} [%]")
        ax.set_title(f"Stokes blend sweep, {name}  (velocity grid ${grid}^2$)")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7.2, ncol=2, loc="lower left")
    fig.suptitle(r"$P_t=(1-t)\,\alpha I + t\,P_{\rm block}$:  $t=0$ is $\frac{1}{2}\|r\|^2$, "
                 r"$t=1$ is $\frac{1}{2} r^\top\! P r$", fontsize=10.5, y=1.0)
    fig.tight_layout()
    p = os.path.join(out_dir, "blend_overlay.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"figure -> {p}")

    # ---------------- collapse ----------------
    pts = [(cond[round(t, 6)]["cond_KPK"], r, t) for t, r in runs if round(t, 6) in cond]
    if len(pts) < 2:
        print("! not enough measured conditioning values for the collapse plot")
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for key, name, marker in (("test_rel_l2_u", "velocity", "o"),
                              ("test_rel_l2_p", "pressure", "s")):
        ax.plot([k for k, _, _ in pts], [100.0 * r[key] for _, r, _ in pts],
                marker + "-", lw=1.6, ms=7, label=name)
    for k, r, t in pts:
        ax.annotate(f"{t:g}", xy=(k, 100.0 * r["test_rel_l2_u"]), fontsize=7.5,
                    xytext=(0, 7), textcoords="offset points", ha="center", color="#444444")
    ax.axhline(100.0, color="#888888", ls=":", lw=1.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"measured $\kappa(\mathcal{K P}_t\mathcal{K})$ — the Gauss--Newton conditioning")
    ax.set_ylabel("test relative FE-$L^2$ [%]")
    ax.set_title(f"Stokes: error collapses onto loss conditioning (${grid}^2$, labels are $t$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, "blend_collapse.png")
    fig.savefig(p, dpi=150)
    print(f"figure -> {p}")

    print(f"\n{'t':>7} {'kappa(KP_tK)':>13} {'vel %':>8} {'pre %':>8}")
    for k, r, t in pts:
        print(f"{t:>7g} {k:13.4e} {100*r['test_rel_l2_u']:8.2f} {100*r['test_rel_l2_p']:8.2f}")


if __name__ == "__main__":
    main()
