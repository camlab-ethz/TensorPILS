"""Learning-rate selection curves for the Poisson paper table.

One panel per arm, one curve per learning rate: **validation** relative L2 against epoch. The
validation set is the right one to select on -- the training set would conflate fitting with
generalisation, and the test set must stay untouched until the final table.

Prints the selected lr per arm (lowest best-validation relative L2) in a form that can be pasted
straight into ``sweep.sbatch``'s ARMS_LR.

**Epoch 0.** The trainer validates only *after* each epoch, so a run's first logged point already
sits 32 optimizer steps in -- at a different place for each lr, which makes the curves in a panel
look like they start from unrelated errors. ``init_probe.sh`` recovers the true initialization
error (a 1-epoch run at lr = 1e-12, i.e. a no-op) and this script prepends it as epoch 0, giving
every curve in a panel one shared, measured origin. If the probe has not been run the curves simply
start at epoch 1 as before.

Learning-rate selection deliberately ignores that epoch-0 point: it is the *training* history that
is being scored, and including a common constant could otherwise let a diverging run "win" on its
initialization.

Usage:
    python experiments/poisson/poisson_paper/plot_lr_sweep.py \
        --root output/poisson/poisson_paper/lr_sweep
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

# Map a run-file prefix onto the arm it belongs to. Order fixes the panel and ARMS_LR order.
ARMS = [
    ("data",       "data-driven",            lambda n: n.startswith("fno_data_")),
    ("ls",         "least squares",          lambda n: n.startswith("fno_galerkin_")),
    ("pls",        "LS + multigrid",         lambda n: n.startswith("fno_pls_")),
    ("deepritz",   "Deep Ritz",              lambda n: n.startswith("fno_deepritz_bc-hard_")),
    ("pdeepritz",  "Deep Ritz + precond",    lambda n: "deepritz_precond" in n),
    ("pino",       "PINO",                   lambda n: n.startswith("fno_pino")),
    ("pideeponet", "PI-DeepONet",            lambda n: n.startswith("deeponet_pi-")),
]
MUTED, TEXT = "#6b6b6b", "#1a1a1a"


def load_init(root):
    """{arm_key: initialization val_rel_l2} from ``init_probe.sh``'s 1-epoch no-op runs.

    Returns an empty dict if the probe directory is absent -- the probe is optional.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "results", "*.json"))):
        name = os.path.basename(path)
        with open(path) as fh:
            hist = json.load(fh).get("stats", {}).get("val_rel_l2_errors") or []
        if not hist:
            continue
        for key, _, match in ARMS:
            if match(name):
                out[key] = float(hist[0])
                break
    return out


def load(root):
    """{arm_key: {lr: val_rel_l2_history}} from output/.../lr<LR>/results/*.json."""
    out = {k: {} for k, _, _ in ARMS}
    for path in sorted(glob.glob(os.path.join(root, "lr*", "results", "*.json"))):
        m = re.search(r"lr([0-9eE.+-]+)[/\\]results", path)
        if not m:
            continue
        lr = float(m.group(1))
        name = os.path.basename(path)
        with open(path) as fh:
            run = json.load(fh)
        hist = run.get("stats", {}).get("val_rel_l2_errors") or []
        if not hist:
            continue
        for key, _, match in ARMS:
            if match(name):
                out[key][lr] = np.asarray(hist, dtype=float)
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/poisson/poisson_paper/lr_sweep")
    ap.add_argument("--out_dir", default=None, help="default: alongside --root")
    ap.add_argument("--init_probe", default="output/poisson/poisson_paper/init_probe",
                    help="1-epoch no-op runs supplying the shared epoch-0 point; optional")
    args = ap.parse_args()
    out_dir = args.out_dir or args.root

    data = load(args.root)
    n_found = sum(len(v) for v in data.values())
    if not n_found:
        raise SystemExit(f"no runs with a validation history under {args.root}")
    print(f"loaded {n_found} runs")
    inits = load_init(args.init_probe)
    if inits:
        print(f"epoch-0 initialization errors from {args.init_probe}:")
        for key, title, _ in ARMS:
            if key in inits:
                print(f"  {title:<22s} {inits[key]:.4f}")
        absent = [k for k, _, _ in ARMS if k not in inits and data[k]]
        if absent:
            print(f"  (no probe for: {', '.join(absent)} -- those curves start at epoch 1)")
    else:
        print(f"no init probe under {args.init_probe}; curves start at epoch 1 "
              f"(run init_probe.sh for a shared origin)")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.edgecolor": MUTED, "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    picks = {}
    for ax, (key, title, _) in zip(axes.ravel(), ARMS):
        runs = data[key]
        if not runs:
            ax.set_title(f"{title}\n(no runs)", fontsize=10); ax.set_axis_off(); continue
        cmap = plt.get_cmap("viridis")
        lrs = sorted(runs)
        e0 = inits.get(key)
        for i, lr in enumerate(lrs):
            h = runs[lr]
            if e0 is None:
                x, y = np.arange(1, len(h) + 1), h
            else:
                x, y = np.arange(0, len(h) + 1), np.concatenate([[e0], h])
            ax.plot(x, y, lw=1.5, color=cmap(i / max(len(lrs) - 1, 1)),
                    label=f"{lr:.0e}  ({h.min():.3f})")
        if e0 is not None:
            # One marker at the shared origin: it is a measured point, not an extrapolation.
            ax.plot([0], [e0], marker="o", ms=4, color=TEXT, zorder=5, clip_on=False)
        # Selection scores the training history only -- never the shared epoch-0 constant.
        best = min(lrs, key=lambda l: runs[l].min())
        picks[key] = (best, runs[best].min())
        ax.set_yscale("log"); ax.set_xlabel("epoch")
        ax.set_title(f"{title}   best lr {best:.0e}", fontsize=10)
        ax.grid(True, which="major", color=MUTED, alpha=0.2, lw=0.5); ax.set_axisbelow(True)
        ax.legend(fontsize=7, frameon=False, title="lr (best val)", title_fontsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0, 0].set_ylabel(r"validation relative $L^2$")
    axes[1, 0].set_ylabel(r"validation relative $L^2$")
    axes[-1, -1].set_axis_off()                       # 7 arms in an 8-panel grid
    fig.tight_layout(pad=0.6)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "lr_sweep")
    fig.savefig(p + ".png", dpi=200); fig.savefig(p + ".pdf")
    plt.close(fig)
    print(f"  {p}.png\n  {p}.pdf")

    print("\nselected learning rates (lowest best validation relative L2):")
    for key, title, _ in ARMS:
        if key in picks:
            lr, err = picks[key]
            edge = "  <-- EDGE OF GRID, widen the sweep" if lr in (1e-4, 3e-2) else ""
            print(f"  {title:<22s} lr = {lr:.0e}   val rel L2 = {err:.4f}{edge}")
    print("\npaste into sweep.sbatch ARMS_LR (data ls pls deepritz pdeepritz pino pideeponet):")
    print("ARMS_LR=(" + " ".join(f"{picks[k][0]:.0e}" if k in picks else "1e-3"
                                 for k, _, _ in ARMS) + ")")


if __name__ == "__main__":
    main()
