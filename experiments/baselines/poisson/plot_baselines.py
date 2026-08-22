"""Aggregate the Poisson baseline sweep into the paper's head-to-head table.

Reads the per-run ``results/*.json`` written by ``PoissonTrainer._save_results_json`` (the
baseline trainers inherit it) and produces

  1. ``baselines.png`` — validation relative-L2 vs epoch, one line per (architecture, loss).
     The curves matter as much as the final numbers: an arm that is still descending at the
     budget is "slower", not "worse", and only the curve distinguishes those;
  2. ``baselines_bar.png`` — final test relative-L2 per arm, grouped by architecture, with the
     "predict zero" level (100%) marked;
  3. a markdown table on stdout and a LaTeX table in ``table.tex``.

With ``--lr_table`` it instead reports best validation error per (arm, learning rate), which is
how the winners for ``sweep.sbatch``'s ``ARMS_LR`` array are read off.

Usage:
    python experiments/baselines/poisson/plot_baselines.py \
        --results_dir output/baselines/poisson          # scans seed*/results recursively
    python experiments/baselines/poisson/plot_baselines.py \
        --results_dir output/baselines/poisson/lr --lr_table
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One colour/marker per arm. Baselines are warm, ours are cool, supervised is black -- so the
# figure reads "black is the target, cool matches it, warm does not" without the legend.
STYLE = {
    ("fno", "data"):            dict(color="#111111", ls="-",  label=r"FNO  $L_{\rm data}$ (supervised)"),
    ("fno", "pls"):             dict(color="#0e7490", ls="-",  label=r"FNO  $L_{\rm PLS}$ (ours)"),
    ("fno", "galerkin"):        dict(color="#b91c1c", ls="-",  label=r"FNO  $L_{\rm LS}$ (bare)"),
    ("fno", "pino"):            dict(color="#ea580c", ls="--", label=r"FNO  PINO (FD strong form)"),
    ("deeponet", "data"):       dict(color="#111111", ls=":",  label=r"DeepONet  $L_{\rm data}$"),
    ("deeponet", "pls"):        dict(color="#0e7490", ls=":",  label=r"DeepONet  $L_{\rm PLS}$ (ours)"),
    ("deeponet", "pideeponet"): dict(color="#a21caf", ls="--", label=r"DeepONet  PI-DeepONet (autodiff)"),
}
ORDER = [("fno", "data"), ("fno", "pls"), ("fno", "pino"), ("fno", "galerkin"),
         ("deeponet", "data"), ("deeponet", "pls"), ("deeponet", "pideeponet")]


def _arch(prefix):
    return "deeponet" if prefix.startswith("deeponet") else "fno"


def _loss(rec):
    """The loss actually trained with. ``loss_type`` is authoritative; the baseline trainers
    overwrite it after construction precisely so the JSON records what ran."""
    lt = rec.get("loss_type", "")
    return "pino" if lt.startswith("pino") else lt


def load_runs(results_dir):
    """All runs under ``results_dir``, scanning ``results/`` subdirectories recursively so a
    seed-per-directory or lr-per-directory layout both work."""
    runs = []
    patterns = [os.path.join(results_dir, "*.json"),
                os.path.join(results_dir, "**", "results", "*.json"),
                os.path.join(results_dir, "results", "*.json")]
    seen = set()
    for pat in patterns:
        for path in sorted(glob.glob(pat, recursive=True)):
            if path in seen:
                continue
            seen.add(path)
            # The lr sweep and the ingredient ablation live under the same parent as the
            # seed dirs but are NOT table rows -- folding them in silently inflates the seed
            # count and averages a 100-epoch tuning run together with a 500-epoch table run.
            if f"{os.sep}lr{os.sep}" in path or f"{os.sep}ablation{os.sep}" in path:
                continue
            with open(path) as fh:
                rec = json.load(fh)
            if "test_rl2" not in rec:        # rollout (AC) records live in the sibling script
                continue
            rec["_path"] = path
            rec["_arch"] = _arch(rec.get("prefix", ""))
            rec["_loss"] = _loss(rec)
            m = re.search(r"seed(\d+)", path)
            rec["_seed"] = int(m.group(1)) if m else None
            m = re.search(r"lr([0-9.e+-]+)", path)
            rec["_lr"] = m.group(1) if m else None
            runs.append(rec)
    return runs


def _fmt(mean, spread, n):
    if n <= 1:
        return f"{mean * 100:.2f}"
    return f"{mean * 100:.2f} ± {spread * 100:.2f}"


def lr_table(runs):
    """Best validation error per (arm, lr) — how sweep.sbatch's ARMS_LR is chosen."""
    by = defaultdict(dict)
    for r in runs:
        by[(r["_arch"], r["_loss"])][r["_lr"]] = r["stats"]["best_val_error"]
    lrs = sorted({r["_lr"] for r in runs if r["_lr"]},
                 key=lambda s: float(s))
    print(f"\n{'arm':<28}" + "".join(f"{l:>12}" for l in lrs) + "   best")
    print("-" * (28 + 12 * len(lrs) + 8))
    for key in ORDER:
        if key not in by:
            continue
        row = by[key]
        cells = "".join(f"{row[l]:>12.2e}" if l in row else f"{'-':>12}" for l in lrs)
        best = min(row, key=row.get) if row else "-"
        print(f"{key[0] + '/' + key[1]:<28}{cells}   {best}")
    print("\nTranscribe the 'best' column into ARMS_LR in sweep.sbatch.\n")


def summarize(runs):
    """(arch, loss) -> (mean test rel-L2, max-min spread, n_seeds, records)."""
    by = defaultdict(list)
    for r in runs:
        by[(r["_arch"], r["_loss"])].append(r)
    out = {}
    for key, recs in by.items():
        vals = [r["test_rl2"] for r in recs]
        mean = sum(vals) / len(vals)
        out[key] = (mean, (max(vals) - min(vals)) / 2.0, len(vals), recs)
    return out


def plot_curves(summary, path):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for key in ORDER:
        if key not in summary:
            continue
        style = dict(STYLE[key])
        label = style.pop("label")
        # Plot the median-performing seed so the curve corresponds to a real run.
        recs = sorted(summary[key][3], key=lambda r: r["test_rl2"])
        rec = recs[len(recs) // 2]
        y = [v * 100 for v in rec["stats"]["val_rel_l2_errors"]]
        ax.plot(range(len(y)), y, label=label, linewidth=1.8, **style)
    ax.axhline(100.0, color="#888888", ls="-.", lw=1, zorder=0)
    ax.text(0.995, 101, "predict zero", ha="right", va="bottom",
            fontsize=8, color="#888888", transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"validation relative $L^2$ (%)")
    ax.set_title("Poisson: published physics-informed baselines vs preconditioned FEM training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def plot_bar(summary, path):
    keys = [k for k in ORDER if k in summary]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    xs = range(len(keys))
    vals = [summary[k][0] * 100 for k in keys]
    errs = [summary[k][1] * 100 for k in keys]
    colors = [STYLE[k]["color"] for k in keys]
    hatch = ["" if k[0] == "fno" else "//" for k in keys]
    bars = ax.bar(xs, vals, yerr=errs, color=colors, capsize=3)
    for b, h in zip(bars, hatch):
        b.set_hatch(h)
    ax.axhline(100.0, color="#888888", ls="-.", lw=1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{k[0]}\n{k[1]}" for k in keys], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel(r"test relative $L^2$ (%)")
    ax.set_title("Final test error (hatched = DeepONet)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def write_tables(summary, out_dir):
    print(f"\n{'architecture':<14}{'loss':<14}{'labels':<9}{'derivative':<14}"
          f"{'test rel-L2 (%)':>18}{'seeds':>7}")
    print("-" * 76)
    meta = {"data": ("yes", "--"), "galerkin": ("no", "FEM"), "pls": ("no", "FEM"),
            "pino": ("no", "FD"), "pideeponet": ("no", "autodiff")}
    rows = []
    for key in ORDER:
        if key not in summary:
            continue
        mean, spread, n, _ = summary[key]
        labels, deriv = meta.get(key[1], ("?", "?"))
        print(f"{key[0]:<14}{key[1]:<14}{labels:<9}{deriv:<14}"
              f"{_fmt(mean, spread, n):>18}{n:>7}")
        rows.append((key, labels, deriv, _fmt(mean, spread, n)))

    tex = [r"\begin{tabular}{llccr}", r"\toprule",
           r"architecture & loss & labels & derivative & rel.\ $L^2$ (\%) \\", r"\midrule"]
    for (arch, loss), labels, deriv, cell in rows:
        name = {"data": r"$L_{\textup{data}}$", "galerkin": r"$L_{\textup{LS}}$",
                "pls": r"$L_{\textup{PLS}}$", "pino": r"PINO",
                "pideeponet": r"PI-DeepONet"}.get(loss, loss)
        tex.append(f"{arch} & {name} & {labels} & {deriv} & {cell} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    path = os.path.join(out_dir, "table.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"\n  -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="output/baselines/poisson")
    ap.add_argument("--out_dir", default=None,
                    help="where figures/table land (default: --results_dir)")
    ap.add_argument("--lr_table", action="store_true",
                    help="report best val error per (arm, lr) instead of the head-to-head table")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no Poisson result JSONs under {args.results_dir}")
    print(f"Loaded {len(runs)} runs from {args.results_dir}")

    if args.lr_table:
        lr_table(runs)
        return

    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    summary = summarize(runs)
    plot_curves(summary, os.path.join(out_dir, "baselines.png"))
    plot_bar(summary, os.path.join(out_dir, "baselines_bar.png"))
    write_tables(summary, out_dir)

    missing = [k for k in ORDER if k not in summary]
    if missing:
        print("\nNot yet run: " + ", ".join(f"{a}/{l}" for a, l in missing))


if __name__ == "__main__":
    main()
