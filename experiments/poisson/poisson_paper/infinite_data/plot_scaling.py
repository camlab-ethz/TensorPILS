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
ARMS = [
    ("pls",  "PLS (multigrid)", "#0072B2", "o", lambda n: n.startswith("fno_pls_")),
    ("data", "data-driven",     "#D55E00", "s", lambda n: n.startswith("fno_data_")),
]
MUTED, TEXT = "#6b6b6b", "#1a1a1a"


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
            finite[arm][int(m.group(1))].append(float(y))
    return finite, stream


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
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.edgecolor": MUTED, "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    for key, label, colour, marker, _ in ARMS:
        if not finite[key]:
            continue
        ns, mean, lo, hi = band(finite[key])
        if np.any(hi > lo):
            ax.fill_between(ns, lo, hi, color=colour, alpha=0.15, lw=0)
        ax.plot(ns, mean, marker=marker, ms=5, lw=1.8, color=colour, label=label)

    # Streaming runs: no finite abscissa, so a horizontal reference line across the axis.
    for key, label, colour, _, _ in ARMS:
        if not stream[key]:
            continue
        y = float(np.mean(stream[key]))
        ax.axhline(y, ls="--", lw=1.5, color=colour, alpha=0.85)
        ax.annotate(f"{label}, infinite stream: {y:.3%}",
                    xy=(0.99, y), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8, color=colour)

    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel(r"training set size $n_{\mathrm{train}}$")
    ax.set_ylabel(r"test relative $L^2$")
    ax.grid(True, which="major", color=MUTED, alpha=0.2, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(pad=0.6)

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "infinite_data")
    fig.savefig(p + ".png", dpi=200); fig.savefig(p + ".pdf")
    plt.close(fig)
    print(f"  {p}.png\n  {p}.pdf")

    # ---------------------------------------------------------------- summary
    all_ns = sorted({n for k in finite for n in finite[k]})
    print(f"\n{'n_train':>9}" + "".join(f"{lab:>18}" for _, lab, _, _, _ in ARMS) + f"{'ratio':>9}")
    for n in all_ns:
        cells, vals = "", {}
        for key, _, _, _, _ in ARMS:
            v = finite[key].get(n)
            vals[key] = float(np.mean(v)) if v else None
            cells += f"{vals[key]:18.5f}" if v else f"{'-':>18}"
        r = (f"{vals['data']/vals['pls']:9.2f}"
             if vals.get('data') and vals.get('pls') else f"{'-':>9}")
        print(f"{n:9d}{cells}{r}")
    for key, label, _, _, _ in ARMS:
        if stream[key]:
            print(f"\n{label} streaming: {np.mean(stream[key]):.5f}")

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
