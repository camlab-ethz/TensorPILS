"""Paper-ready Allen-Cahn figures and table rows (the deliverable; ``plot_final.py`` is the
exploratory view).

Two panels, included side by side at ``width=0.49\\linewidth`` like Figure 3, so each canvas is
authored at exactly 0.49 x 5.5 = 2.695 in and ``\\includegraphics`` does not rescale:

``ac_val_curves.{pdf,png}``   (a) validation space-time relative L2 vs epoch, mean over seeds with a
                              min-max band. Read from ``<final>/seed*/results/*.json``.
``ac_long_rollout.{pdf,png}`` (b) test relative L2 per time step of a free rollout to T = 1, the mean
                              over test samples of the per-sample error, best seed per arm; the
                              failed PI-DeepONet arm is left out. Read from ``test_eval.json``
                              written on Euler by ``evaluate_test.py``.
``ac_legend.{pdf,png}``       one-row legend shared by both panels, placed below them at full width.

With ``test_eval.json`` present it also prints the Allen-Cahn rows of the paper's summary table:
the test space-time relative L2 as the mean over test samples of the per-sample error (steps 0-10),
then mean [min, max] over seeds.

Conventions shared with ``experiments/poisson/sweep_blend/plot_paper.py`` so the figures look like
one paper: Times, 9 pt labels / 7.5 pt ticks, recessive grid, no titles (the caption carries them),
centered moving-average smoothing (no phase lag). Colours: the paper-wide magma-path triple for our
losses (L_LS orange, L_PLS magenta, L_data purple) and two cool hues for the baselines, validated
together with the dataviz checks (all-pairs CVD dE >= 11.8, normal-vision dE >= 20.6). Each arm
also has its own dash pattern, so identity never rests on hue alone.

Usage:
    python experiments/allen_cahn/ac_paper/plot_paper.py --figures_dir paper/figures
"""

import argparse
import json
import os
import shutil

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import NullFormatter

from plot_final import load as load_final

# key, legend label, colour, dash pattern. Order fixes the legend and the table.
ARMS = [
    ("pls",        r"$L_{\mathrm{PLS}}$",  "#bf3a77", "-"),
    ("ls",         r"$L_{\mathrm{LS}}$",   "#e07a1f", (0, (1.2, 1.2))),
    ("data",       r"$L_{\mathrm{data}}$", "#5c167f", (0, (4, 2))),
    ("pino",       "PINO",                 "#09672e", (0, (5, 1.5, 1, 1.5))),
    ("pideeponet", "PI-DeepONet",          "#708df7", (0, (2.5, 1.2))),
]
TABLE_NAMES = {"pls": r"$L_\textup{PLS}$", "data": r"$L_\textup{data}$",
               "pino": "PINO", "pideeponet": "PI-DeepONet", "ls": r"$L_\textup{LS}$"}

FS_LAB, FS_TICK, FS_LEG = 9.0, 7.5, 6.5
MUTED, TEXT = "#6b6b6b", "#1a1a1a"
W_IN = 0.49 * 5.5                # 2.695 in -- exactly what \includegraphics receives
H_IN = 2.25


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
    ax.yaxis.set_minor_formatter(NullFormatter())    # no "9.95x10^-1" minor labels on a narrow log range
    ax.grid(True, which="major", color=MUTED, alpha=0.22, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def smooth(y, window=15):
    """Centered moving average, edge-padded; NaN-tolerant (a NaN epoch does not blank a window)."""
    y = np.asarray(y, dtype=float)
    w = min(int(window), y.size)
    if w % 2 == 0:
        w -= 1
    if w <= 1:
        return y
    pad = np.pad(y, w // 2, mode="edge")
    ok = np.isfinite(pad)
    num = np.convolve(np.where(ok, pad, 0.0), np.ones(w), mode="valid")
    cnt = np.convolve(ok.astype(float), np.ones(w), mode="valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, num / cnt, np.nan)


def save(fig, out_dir, stem, figures_dir, dpi=300):
    os.makedirs(out_dir, exist_ok=True)
    pdf, png = os.path.join(out_dir, stem + ".pdf"), os.path.join(out_dir, stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    print(f"  {pdf}\n  {png}")
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        shutil.copy(pdf, os.path.join(figures_dir, stem + ".pdf"))
        print(f"  -> {os.path.join(figures_dir, stem + '.pdf')}")


def legend(ax, **kw):
    ax.legend(frameon=False, fontsize=FS_LEG, handlelength=2.4, labelspacing=0.25,
              borderaxespad=0.3, **kw)


# --------------------------------------------------------------------------- panel (a)
def panel_val_curves(final_root, out_dir, figures_dir, window):
    data = load_final(final_root)
    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    for key, label, colour, dash in ARMS:
        runs = data.get(key) or {}
        if not runs:
            print(f"  (no final runs for {key})")
            continue
        n = min(len(r["hist"]) for r in runs.values())
        arr = np.vstack([r["hist"][:n] for _, r in sorted(runs.items())])
        ep = np.arange(1, n + 1)
        with np.errstate(all="ignore"):
            mean, lo, hi = np.nanmean(arr, 0), np.nanmin(arr, 0), np.nanmax(arr, 0)
        if len(runs) > 1:
            ax.fill_between(ep, smooth(lo, window), smooth(hi, window), color=colour, alpha=0.11, lw=0)
        ax.plot(ep, smooth(mean, window), color=colour, lw=1.2, ls=dash)
    ax.set_yscale("log")
    ax.set_xlim(0, 500)
    ax.set_xlabel("epoch", fontsize=FS_LAB, labelpad=1.5)
    ax.set_ylabel(r"val. space-time rel. $L^2$", fontsize=FS_LAB, labelpad=2)
    finish(ax)
    fig.tight_layout(pad=0.3)
    save(fig, out_dir, "ac_val_curves", figures_dir)


# --------------------------------------------------------------------------- panel (b)
# PI-DeepONet is left out: it fails (error ~0.7-1 throughout) and would stretch the axis over two
# decades, flattening the differences between the arms that train. The caption says so.
LONG_EXCLUDE = ("pideeponet",)
NICE_TICKS = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


def panel_long_rollout(ev, out_dir, figures_dir):
    dt, horizon = ev["config"]["dt"], ev["config"]["rollout_steps"]
    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    ymin, ymax = np.inf, 0.0
    for key, label, colour, dash in ARMS:
        if key in LONG_EXCLUDE:
            continue
        runs = [r for r in ev["runs"] if r["arm"] == key and r["best_seed_for_arm"]]
        if not runs:
            print(f"  (no long rollout for {key})")
            continue
        r = runs[0]
        y = np.asarray(r["step_rel_mean"], dtype=float)
        t = dt * np.arange(y.size)
        bad = np.asarray(r["step_rel_n_nonfinite"])
        if bad.any():
            print(f"  WARNING {key} seed {r['seed']}: non-finite samples from step "
                  f"{int(np.argmax(bad > 0))} (max {bad.max()} of {r['n_samples']}); mean over finite ones")
        ax.plot(t[1:], y[1:], color=colour, lw=1.2, ls=dash)             # step 0 is the exact IC
        ymin, ymax = min(ymin, np.nanmin(y[1:])), max(ymax, np.nanmax(y[1:]))
    ax.set_yscale("log")
    lo, hi = ymin / 1.25, ymax * 1.25
    ax.set_ylim(lo, hi)
    # Less than a decade of range: label round values, not just the one power of ten in view.
    ax.set_yticks([v for v in NICE_TICKS if lo <= v <= hi])
    ax.set_yticklabels([f"{v:g}" for v in NICE_TICKS if lo <= v <= hi])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    th = dt * horizon
    ax.axvline(th, color=TEXT, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.text(th + 0.012, hi / 1.08, "training horizon", fontsize=FS_LEG, color=TEXT, va="top", ha="left")
    ax.set_xlim(0, dt * (len(t) - 1))
    ax.set_xlabel(r"time $t$", fontsize=FS_LAB, labelpad=1.5)
    ax.set_ylabel(r"test rel. $L^2$", fontsize=FS_LAB, labelpad=2)
    finish(ax)
    fig.tight_layout(pad=0.3)
    save(fig, out_dir, "ac_long_rollout", figures_dir)


# --------------------------------------------------------------------------- shared legend
def legend_strip(out_dir, figures_dir):
    """One legend below both panels, authored at the full text width (5.5 in) in a single row."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=1.2, ls=d) for _, _, c, d in ARMS]
    fig = plt.figure(figsize=(5.5, 0.22))
    fig.legend(handles, [lab for _, lab, _, _ in ARMS], loc="center", ncol=len(ARMS), frameon=False,
               fontsize=FS_LAB - 1, handlelength=2.6, columnspacing=1.6, borderaxespad=0)
    save(fig, out_dir, "ac_legend", figures_dir)


# --------------------------------------------------------------------------- table rows
def table_rows(ev):
    """AC rows of the summary table: per-sample test space-time rel. L2 (steps 0-10), averaged over
    the test set, then mean [min, max] over seeds. Seeds with non-finite samples are flagged."""
    print("\nsummary-table rows (test, mean over samples of per-sample space-time rel. L2, steps 0-10):")
    for key, _, _, _ in ARMS:
        vals, notes = [], []
        for r in sorted((r for r in ev["runs"] if r["arm"] == key), key=lambda r: r["seed"]):
            x = np.asarray(r["st_rel_0_10"], dtype=float)
            nf = int((~np.isfinite(x)).sum())
            vals.append(float(x[np.isfinite(x)].mean()) if nf < x.size else np.nan)
            if nf:
                notes.append(f"seed {r['seed']}: {nf}/{x.size} test samples non-finite, excluded")
        if not vals:
            continue
        v = np.asarray(vals)
        f = v[np.isfinite(v)]
        cell = (f"${f.mean():.3f}$ $[{f.min():.3f}, {f.max():.3f}]$" if f.size > 1
                else f"${f.mean():.3f}$ (1 seed)" if f.size else "---")
        print(f"      & {TABLE_NAMES[key]:<22s} & ... & {cell} & --- \\\\"
              f"   % seeds {[r['seed'] for r in ev['runs'] if r['arm'] == key]}")
        for n in notes:
            print(f"        % {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", default="output/allen_cahn/ac_paper/final")
    ap.add_argument("--eval", default=None, help="default: <final>/test_eval.json")
    ap.add_argument("--out_dir", default="output/allen_cahn/ac_paper/paper")
    ap.add_argument("--figures_dir", default=None,
                    help="also copy the PDFs here, e.g. paper/figures")
    ap.add_argument("--window", type=int, default=15, help="smoothing window (epochs)")
    args = ap.parse_args()
    style()

    print("panel (a): validation curves")
    panel_val_curves(args.final, args.out_dir, args.figures_dir, args.window)
    print("shared legend")
    legend_strip(args.out_dir, args.figures_dir)

    ev_path = args.eval or os.path.join(args.final, "test_eval.json")
    if not os.path.exists(ev_path):
        print(f"\npanel (b) and table rows skipped: {ev_path} not found (run evaluate_test.py on Euler)")
        return
    with open(ev_path) as fh:
        ev = json.load(fh)
    print("\npanel (b): long rollout")
    panel_long_rollout(ev, args.out_dir, args.figures_dir)
    table_rows(ev)


if __name__ == "__main__":
    main()
