"""Test error against dataset size: preconditioned least squares vs supervised training.

Reads the per-run ``results/*.json`` written by ``PoissonTrainer._save_results_json`` and draws
one curve per arm against ``n_train`` on log-log axes, plus a horizontal reference line for the
PLS streaming run (the infinite-data limit, where no sample is ever seen twice).

Rewritten rather than adapted from ``experiments/poisson/data_scaling/plot_scaling.py``: that
version plots a single curve and has no notion of arms, because the original sweep ran PLS alone.

**Why there is no streaming line for the supervised arm.** PLS's infinite-data limit is reachable
-- fresh source terms cost nothing, since the loss needs no labels. Infinite *labelled* data would
require a PDE solve per sample; it is free here only because this dataset is manufactured. The
asymmetry is the figure's point, not a gap in it.

Seeds are aggregated to a mean with a min-max band when more than one is present, so the script
does not need changing if seeds are added later.

Usage:
    python experiments/poisson/poisson_paper/infinite_data/plot_scaling.py
"""

import argparse
import collections
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt

# Prefix -> arm. Same matching convention as poisson_benchmark's plotting scripts.
# Colours are fixed across the paper by the graphical abstract: orange is L_LS, blue is L_PLS.
# Colours come from the paper-wide triple on a magma sub-path: L_LS orange #e07a1f, L_PLS magenta
# #bf3a77, L_data deep purple #5c167f. Pairwise OKLab dE 20.5 / 41.0 / 22.8, all well clear of the
# 8 floor. Chosen so the blend-sweep ramp between L_LS and L_data can pass through L_PLS without
# crossing the achromatic axis; see experiments/poisson/sweep_blend/plot_paper.py.
# The two arms coincide to within 1-4% at every size, so a solid line simply hides the other and
# markers only re-hide it at the sample points. One solid, one dashed: the dashes let the curve
# underneath show through, which reads as "these lie on top of each other" rather than "one is
# missing". The infinite-data reference is dotted so it stays distinct from L_data's dashes.
ARMS = [
    # Lightened from the canonical #bf3a77. In this figure L_PLS and L_data are the two curves
    # that lie on top of each other, and they are the closest pair in the paper triple (dE 22.8).
    # #d24d87 opens that to 27.5 while staying only dE 5.5 from the canonical magenta, so it still
    # reads as the same colour next to the other figures. Local deviation, deliberate.
    ("pls",  r"$L_{\mathrm{PLS}}$",  "#d24d87", "-",        lambda n: n.startswith("fno_pls_")),
    ("data", r"$L_{\mathrm{data}}$", "#5c167f", (0, (4, 2)), lambda n: n.startswith("fno_data_")),
]
STREAM_LS = (0, (1, 1.6))     # dotted; must not be confusable with L_data's dashes
PLAIN = {"pls": "L_PLS", "data": "L_data"}       # for the terminal table; math is for the figure
MUTED, TEXT = "#6b6b6b", "#1a1a1a"

# Authored at the size the paper actually gives it: a wrapfigure of 0.40\textwidth, and the ICLR
# text width is 5.5 in, so \includegraphics[width=\linewidth] does not rescale and the point sizes
# below are the ones that reach the page. Body text is 10 pt; 9 pt labels and 7.5 pt ticks match
# the graphical abstract and the blend-sweep figures, so all of them look like one paper.
W_IN, H_IN = 0.40 * 5.5, 1.95
FS_LAB, FS_TICK, FS_LEG = 9.0, 7.5, 7.0

# Largest dataset size shown. The curve is still bending at the top end, which points at the
# fixed 32,000-step budget being the binding constraint there rather than the data -- so the
# last point would invite a convergence reading the experiment cannot support. Deliberate cut,
# not a data problem: the run exists and is on disk.
MAX_N = 2048


def load(results_dir):
    """-> ({arm: {n_train: [values]}}, {arm: [streaming values]}) keyed on the samples-<n> tag."""
    finite = {k: defaultdict(list) for k, _, _, _, _ in ARMS}
    stream = {k: [] for k, _, _, _, _ in ARMS}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path)
        with open(path) as fh:
            run = json.load(fh)
        arm = next((k for k, _, _, _, m in ARMS if m(name)), None)
        if arm is None:
            continue
        y = run.get("test_rl2")
        if y is None:
            continue
        m = re.search(r"samples-([0-9]+|inf)-", name)
        if not m:
            continue
        if m.group(1) == "inf" or run.get("stream"):
            stream[arm].append(float(y))
        else:
            n = int(m.group(1))
            if n <= MAX_N:
                finite[arm][n].append(float(y))
    return finite, stream


def _decade_ladder(lo, hi):
    """1-2-5 ticks covering [lo, hi] — the readable ladder on a log axis of ~1.5 decades."""
    out = []
    d = int(np.floor(np.log10(lo)))
    while 10.0 ** d <= hi * 10:
        for m in (1, 2, 5):
            v = m * 10.0 ** d
            if lo <= v <= hi:
                out.append(v)
        d += 1
    return out


def _fmt(v):
    """0.02 not 2e-2, and no trailing zeros."""
    return f"{v:.10f}".rstrip("0").rstrip(".")


def band(per_n):
    """{n: [vals]} -> (ns, mean, lo, hi), sorted by n."""
    ns = np.array(sorted(per_n))
    v = [np.asarray(per_n[n], dtype=float) for n in ns]
    return ns, np.array([x.mean() for x in v]), \
           np.array([x.min() for x in v]), np.array([x.max() for x in v])


def _report_label_modes(results_root):
    """Print which label mode(s) the loaded runs used, and shout if they are mixed.

    ``dataset_solution`` distinguishes analytic labels from FEM ones. Runs of the two kinds share
    a filename, so without this a stale file sits silently beside a fresh one. ``None`` means the
    run predates the field -- i.e. analytic, since that was the only behaviour then.
    """
    modes = collections.Counter()
    for path in glob.glob(os.path.join(results_root, "**", "*.json"), recursive=True):
        with open(path) as fh:
            try:
                modes[json.load(fh).get("dataset_solution") or "analytic (pre-flag)"] += 1
            except json.JSONDecodeError:
                continue
    if not modes:
        return
    label = ", ".join(f"{k}: {v}" for k, v in sorted(modes.items()))
    print(f"label mode -> {label}")
    if len(modes) > 1:
        print("  WARNING: MIXED label modes in one directory -- these numbers are not comparable")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/poisson/poisson_paper/infinite_data")
    ap.add_argument("--out_dir", default=None, help="default: alongside --root")
    args = ap.parse_args()
    out_dir = args.out_dir or args.root

    finite, stream = load(os.path.join(args.root, "results"))
    n_found = sum(len(v) for v in finite.values()) + sum(len(v) for v in stream.values())
    if not n_found:
        raise SystemExit(f"no runs found under {args.root}/results")
    print(f"loaded {n_found} runs")
    _report_label_modes(os.path.join(args.root, "results"))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.edgecolor": MUTED, "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    })
    fig, ax = plt.subplots(figsize=(W_IN, H_IN))

    for key, label, colour, ls, _ in ARMS:
        if not finite[key]:
            continue
        ns, mean, lo, hi = band(finite[key])
        if np.any(hi > lo):
            ax.fill_between(ns, lo, hi, color=colour, alpha=0.15, lw=0)
        ax.plot(ns, mean, ls=ls, lw=1.7, color=colour, label=label)

    # Streaming runs: no finite abscissa, so a horizontal reference line across the axis.
    for key, label, colour, _, _ in ARMS:
        if not stream[key]:
            continue
        y = float(np.mean(stream[key]))
        # "infinite data" alone: the colour already ties it to L_PLS, and the legend is narrow.
        ax.axhline(y, ls=STREAM_LS, lw=1.7, color=colour, alpha=0.9, label="infinite data")

    ax.set_xscale("log", base=2); ax.set_yscale("log")
    # Plain integers, not 2^k: the sizes are the quantity of interest, not the exponent.
    all_ns = sorted({n for k in finite for n in finite[k]})
    ax.set_xticks(all_ns)
    ax.set_xticklabels([str(n) for n in all_ns])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # A 1-2-5 ladder with plain decimals, instead of matplotlib's "6 x 10^-1" offset labels.
    lo, hi = ax.get_ylim()
    ticks = [t for t in _decade_ladder(lo, hi)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([_fmt(t) for t in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("training set size", fontsize=FS_LAB, labelpad=1.5)
    ax.set_ylabel(r"test relative $L^2$", fontsize=FS_LAB, labelpad=2)
    ax.grid(True, which="major", color=MUTED, alpha=0.22, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=FS_LEG, handlelength=1.9, labelspacing=0.25,
              handletextpad=0.5, borderaxespad=0.2, loc="upper right")
    fig.tight_layout(pad=0.3)

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "infinite_data")
    fig.savefig(p + ".png", dpi=200); fig.savefig(p + ".pdf")
    plt.close(fig)
    print(f"  {p}.png\n  {p}.pdf")

    # ---------------------------------------------------------------- summary
    all_ns = sorted({n for k in finite for n in finite[k]})
    print(f"\n{'n_train':>9}" + "".join(f"{PLAIN[k]:>18}" for k, _, _, _, _ in ARMS) + f"{'ratio':>9}")
    for n in all_ns:
        cells, vals = "", {}
        for key, _, _, _, _ in ARMS:
            v = finite[key].get(n)
            vals[key] = float(np.mean(v)) if v else None
            cells += f"{vals[key]:18.5f}" if v else f"{'-':>18}"
        r = (f"{vals['data']/vals['pls']:9.2f}"
             if vals.get('data') and vals.get('pls') else f"{'-':>9}")
        print(f"{n:9d}{cells}{r}")
    for key, _, _, _, _ in ARMS:
        if stream[key]:
            print(f"\n{PLAIN[key]} infinite data: {np.mean(stream[key]):.5f}")

    # How much labelled data does supervised training need to match PLS at each size?
    if finite["pls"] and finite["data"]:
        dn, dv = band(finite["data"])[0], band(finite["data"])[1]
        for key, target in (("pls_max", max(finite["pls"])),):
            pv = float(np.mean(finite["pls"][target]))
            reached = dn[dv <= pv]
            msg = (f"n_train >= {int(reached[0])}" if len(reached)
                   else f"NOT reached within n_train <= {int(dn[-1])}")
            print(f"\nsupervised matches PLS@n={target} ({pv:.5f}) at {msg}")


if __name__ == "__main__":
    main()
