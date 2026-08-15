"""Error-vs-iteration panel comparing the bare and preconditioned least-squares losses.

Companion to the two matrix views in ``plot_landscape.py --matrix``: those show *where in the
spectrum* the error sits, this shows *how much of it is left*. Intended as the third panel of a
three-panel figure, so the type is sized to sit beside the large matrix renders rather than to be
read at journal scale.

Both runs must use the same optimizer, learning rate and initialisation -- the point is that the
only difference is the loss. Pass the two ``.npz`` files written by ``run_landscape.py``.

Usage:
    python experiments/graphical_abstract/plot_compare.py \
        --bare output/graphical_abstract/lr_sweep/bare/lr1e-3_1e5steps/run.npz \
        --mg   output/graphical_abstract/lr_sweep/mg/lr1e-3_1e5steps/run.npz \
        --out_dir output/graphical_abstract/lr_sweep
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt

C_BARE = "#e07a1f"
C_MG = "#1b6ec2"
MUTED = "#6b6b6b"
TEXT = "#1a1a1a"


def first_below(steps, err, tol):
    """First recorded iteration at or below ``tol`` (None if never reached)."""
    hit = np.flatnonzero(np.asarray(err) <= tol)
    return int(steps[hit[0]]) if hit.size else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bare", required=True, help="npz of the 1/2||Au-b||^2 run")
    ap.add_argument("--mg", required=True, help="npz of the 1/2||P(Au-b)||^2 run")
    ap.add_argument("--out_dir", default="output/graphical_abstract")
    ap.add_argument("--name", default="compare_error")
    ap.add_argument("--width", type=float, default=12.0)
    ap.add_argument("--height", type=float, default=11.0)
    ap.add_argument("--fs", type=float, default=26.0, help="base font size")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="accuracy whose iteration cost is annotated (default 1e-3)")
    args = ap.parse_args()

    db, dm = np.load(args.bare), np.load(args.mg)
    lr_b, lr_m = float(db["best_lr"]), float(dm["best_lr"])
    if abs(lr_b - lr_m) / max(lr_b, 1e-30) > 1e-9:
        print(f"WARNING: learning rates differ ({lr_b:.2e} vs {lr_m:.2e}); the two curves are then "
              "not a like-for-like comparison of the losses alone")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": MUTED, "axes.linewidth": 1.0, "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fs = args.fs
    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi)

    ax.axhline(1.0, color=MUTED, lw=1.2, ls=(0, (5, 4)), zorder=1)
    ax.text(float(db["steps"][0]), 1.15, r"$u\equiv 0$", color=MUTED, fontsize=fs * 0.7,
            va="bottom")

    for d, colr, lab in ((db, C_BARE, r"$\frac{1}{2}\|Au-b\|^2$"),
                         (dm, C_MG, r"$\frac{1}{2}\|P(Au-b)\|^2$")):
        k = first_below(d["steps"], d["err_adam"], args.tol)
        tail = f"  ({k} it. to {args.tol:g})" if k else f"  (never reaches {args.tol:g})"
        ax.plot(d["steps"], d["err_adam"], color=colr, lw=2.6, label=lab + tail, zorder=4)
        if k:                                     # mark where the target accuracy is first met
            ax.plot([k], [args.tol], marker="o", ms=fs * 0.45, mfc="white", mec=colr,
                    mew=2.2, zorder=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("iteration", fontsize=fs)
    ax.set_ylabel(r"relative $L^2$ error", fontsize=fs)
    ax.tick_params(labelsize=fs * 0.8, length=8, width=1.2)
    ax.grid(True, which="major", color=MUTED, alpha=0.2, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=fs * 0.85, handlelength=1.8)
    lm = db["lr_min"] if "lr_min" in db.files else np.nan
    sched = "" if not np.isfinite(lm) else rf" $\rightarrow$ {float(lm):.0e} (cosine)"
    ax.set_title(rf"Adam, lr $=${lr_b:.0e}{sched}", fontsize=fs * 1.15, pad=14)

    fig.tight_layout(pad=0.6)
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, args.name)
    fig.savefig(p + ".png", dpi=args.dpi)
    fig.savefig(p + ".pdf")
    plt.close(fig)
    print(f"  {p}.png\n  {p}.pdf")

    for d, nm in ((db, "bare   "), (dm, "precond")):
        e, s = d["err_adam"], d["steps"]
        print(f"  {nm}: min {e.min():.3e}  final {e[-1]:.3e}  "
              f"it. to 1e-2 {first_below(s, e, 1e-2)}  to 1e-3 {first_below(s, e, 1e-3)}")


if __name__ == "__main__":
    main()
