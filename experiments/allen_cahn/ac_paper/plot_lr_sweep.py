"""Learning-rate selection curves for the Allen-Cahn benchmark.

One panel per arm, one curve per learning rate: **validation space-time relative L2** against
epoch. Prints the selected rate per arm in a form that pastes straight into ``sweep.sbatch``'s
``ARMS_LR``.

Two things differ from the Poisson twin
(``experiments/poisson/poisson_paper/poisson_benchmark/plot_lr_sweep.py``):

* **The metric.** Allen-Cahn is an autoregressive rollout, so the per-epoch series is
  ``stats.val_st_rel_l2`` -- the space-time aggregate over the rollout window -- not
  ``val_rel_l2_errors``, which does not exist for this PDE.
* **Divergence is reported, not silently plotted.** The bare least-squares arm is known to fail
  outright here (the preliminary ``realistic_ac`` study measured 0.9997, i.e. total collapse) and
  the failure can present as NaN or as a flat line at ~1.0 depending on the seed. Either would
  otherwise be picked up as an ordinary "best" value and quietly win a comparison against other
  equally-broken rates, so both are flagged AND excluded from selection. An arm where every rate
  collapses is reported as such rather than being assigned an arbitrary winner.

Usage:
    python experiments/allen_cahn/ac_paper/plot_lr_sweep.py
"""

import argparse
import glob
import json
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt

# Prefix -> arm. Order fixes the panels and the ARMS_LR order, and must match the sbatch's
# ARMS_TAG. Note the ls/pls test: BOTH start with "fno_ac_galerkin_ls_", and only the
# preconditioned one carries "precmg" -- so pls must be tested before ls, or by exclusion.
ARMS = [
    ("data",       "data-driven",     lambda n: n.startswith("fno_ac_data_")),
    ("ls",         "least squares",   lambda n: n.startswith("fno_ac_galerkin_ls_") and "precmg" not in n),
    ("pls",        "LS + multigrid",  lambda n: n.startswith("fno_ac_galerkin_ls_") and "precmg" in n),
    ("pino",       "PINO",            lambda n: n.startswith("fno_ac_pino_")),
    ("pideeponet", "PI-DeepONet",     lambda n: n.startswith("deeponet_ac_pi_")),
]
MUTED, TEXT = "#6b6b6b", "#1a1a1a"

# A run at or above this is not "worse", it is broken: the rollout has collapsed to the trivial
# state. realistic_ac's bare-LS arm sat at 0.9997.
COLLAPSE = 0.99


def load(root):
    """{arm: {lr: val_st_rel_l2 history}} from ``<root>/lr<LR>/results/*.json``."""
    out = {k: {} for k, _, _ in ARMS}
    for path in sorted(glob.glob(os.path.join(root, "lr*", "results", "*.json"))):
        m = re.search(r"lr([0-9eE.+-]+)[/\\]results", path)
        if not m:
            continue
        lr = float(m.group(1))
        name = os.path.basename(path)
        with open(path) as fh:
            hist = json.load(fh).get("stats", {}).get("val_st_rel_l2") or []
        if not hist:
            continue
        for key, _, match in ARMS:
            if match(name):
                out[key][lr] = np.asarray(hist, dtype=float)
                break
    return out


def diagnose(h):
    """-> (best, note, ok). ``ok`` is False when the rate must not be selected.

    A run that dips low and *then* diverges to NaN is not a usable rate: the minimum over its
    finite values is a real number, so without this it would be selected on the strength of a
    transient before the blow-up. Divergence is the documented Allen-Cahn failure mode, so this
    is the case to get right, not an edge case.
    """
    finite = h[np.isfinite(h)]
    if finite.size == 0:
        return float("nan"), "all NaN", False
    best = float(finite.min())
    notes, ok = [], True
    if finite.size < h.size:
        notes.append(f"NaN from ep {int(np.argmax(~np.isfinite(h))) + 1}")
        ok = False                      # diverged, whatever it reached first
    if best >= COLLAPSE:
        notes.append("collapsed")
        ok = False
    return best, ", ".join(notes), ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/allen_cahn/ac_paper/lr_sweep")
    ap.add_argument("--out_dir", default=None, help="default: alongside --root")
    args = ap.parse_args()
    out_dir = args.out_dir or args.root

    data = load(args.root)
    n_found = sum(len(v) for v in data.values())
    if not n_found:
        raise SystemExit(f"no runs with a validation history under {args.root}")
    print(f"loaded {n_found} runs")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.edgecolor": MUTED, "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    picks, flags = {}, []
    for ax, (key, title, _) in zip(axes.ravel(), ARMS):
        runs = data[key]
        if not runs:
            ax.set_title(f"{title}\n(no runs)", fontsize=10); ax.set_axis_off(); continue
        cmap = plt.get_cmap("viridis")
        lrs = sorted(runs)
        scored = {}
        for i, lr in enumerate(lrs):
            h = runs[lr]
            best, note, ok = diagnose(h)
            if ok:
                scored[lr] = best
            if note:
                flags.append(f"  {title:<16s} lr={lr:.0e}  {note}")
            ax.plot(np.arange(1, len(h) + 1), h, lw=1.5,
                    color=cmap(i / max(len(lrs) - 1, 1)),
                    label=f"{lr:.0e}  ({best:.3f})" if np.isfinite(best) else f"{lr:.0e}  (NaN)")
        # ``scored`` already holds only the rates diagnose() judged usable. Choosing the "best"
        # among rates that all collapse would report an arbitrary winner for an arm that simply
        # does not train -- which is exactly the bare-LS case, and the finding, not a rate to tune.
        usable = scored
        if usable:
            best_lr = min(usable, key=usable.get)
            picks[key] = (best_lr, usable[best_lr])
        ax.set_yscale("log"); ax.set_xlabel("epoch")
        ax.set_title(f"{title}   best lr {best_lr:.0e}" if usable else f"{title}   (no rate trains)",
                     fontsize=10)
        ax.grid(True, which="major", color=MUTED, alpha=0.2, lw=0.5); ax.set_axisbelow(True)
        ax.legend(fontsize=7, frameon=False, title="lr (best val)", title_fontsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0, 0].set_ylabel(r"validation space-time relative $L^2$")
    axes[1, 0].set_ylabel(r"validation space-time relative $L^2$")
    axes[-1, -1].set_axis_off()                       # 5 arms in a 6-panel grid
    fig.tight_layout(pad=0.6)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "ac_lr_sweep")
    fig.savefig(p + ".png", dpi=200); fig.savefig(p + ".pdf")
    plt.close(fig)
    print(f"  {p}.png\n  {p}.pdf")

    if flags:
        print("\ndiverged or collapsed runs (plotted, but excluded from selection):")
        print("\n".join(flags))

    print("\nselected learning rates (lowest best validation space-time relative L2):")
    for key, title, _ in ARMS:
        if key not in picks:
            n = len(data[key])
            print(f"  {title:<22s} NO RATE TRAINS -- all {n} diverged or collapsed (>= {COLLAPSE})")
            continue
        lr, err = picks[key]
        # Compare against the rates actually swept for THIS arm, not a hardcoded pair: the grid
        # may have been extended for one arm and not another.
        swept = sorted(data[key])
        edge = "  <-- EDGE OF GRID, widen the sweep" if lr in (swept[0], swept[-1]) else ""
        print(f"  {title:<22s} lr = {lr:.0e}   val st-rel L2 = {err:.4f}{edge}")

    print("\npaste into sweep.sbatch ARMS_LR (" + " ".join(k for k, _, _ in ARMS) + "):")
    print("ARMS_LR=(" + " ".join(f"{picks[k][0]:.0e}" if k in picks else "1e-3"
                                 for k, _, _ in ARMS) + ")")


if __name__ == "__main__":
    main()
