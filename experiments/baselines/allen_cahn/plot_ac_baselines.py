"""Aggregate the Allen-Cahn baseline sweep.

Reads the per-run ``results/*.json`` written by ``RolloutTrainer._save_results_json`` (the
baseline trainers inherit it) and produces

  1. ``ac_baselines.png`` — validation space-time relative FEM-L2 vs epoch, one line per
     (architecture, loss);
  2. ``ac_baselines_steps.png`` — test relative L2 per rollout step, which is where a stepper
     that is merely *stable* separates from one that is *accurate*;
  3. a markdown table on stdout and a LaTeX table in ``table.tex``.

Usage:
    python experiments/baselines/allen_cahn/plot_ac_baselines.py \
        --results_dir output/baselines/allen_cahn
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

STYLE = {
    ("fno", "data"):            dict(color="#111111", ls="-",  label=r"FNO  $L_{\rm data}$ (supervised)"),
    ("fno", "pls"):             dict(color="#0e7490", ls="-",  label=r"FNO  $L_{\rm PLS}$ (ours, precond.)"),
    ("fno", "min_movement"):    dict(color="#15803d", ls="-",  label=r"FNO  $L_{\rm MM}$ (ours, min. movement)"),
    ("fno", "galerkin"):        dict(color="#b91c1c", ls="-",  label=r"FNO  $L_{\rm LS}$ (bare)"),
    ("fno", "pino"):            dict(color="#ea580c", ls="--", label=r"FNO  PINO (FD strong form)"),
    ("deeponet", "data"):       dict(color="#111111", ls=":",  label=r"DeepONet  $L_{\rm data}$"),
    ("deeponet", "pideeponet"): dict(color="#a21caf", ls="--", label=r"DeepONet  PI-DeepONet (autodiff)"),
}
ORDER = [("fno", "data"), ("fno", "min_movement"), ("fno", "pls"), ("fno", "galerkin"),
         ("fno", "pino"), ("deeponet", "data"), ("deeponet", "pideeponet")]


def load_runs(results_dir):
    runs, seen = [], set()
    patterns = [os.path.join(results_dir, "*.json"),
                os.path.join(results_dir, "**", "results", "*.json"),
                os.path.join(results_dir, "results", "*.json")]
    for pat in patterns:
        for path in sorted(glob.glob(pat, recursive=True)):
            if path in seen:
                continue
            seen.add(path)
            with open(path) as fh:
                rec = json.load(fh)
            if "test_st_rel_l2" not in rec:       # static (Poisson) records go to the sibling script
                continue
            prefix = rec.get("prefix", "")
            rec["_path"] = path
            rec["_arch"] = "deeponet" if prefix.startswith("deeponet") else "fno"
            lt = rec.get("loss_type", "")
            # Three arms share loss_type='galerkin' and are told apart only by the tags the
            # trainer puts in the prefix: '_precmg' (preconditioned least squares) and '_mm_'
            # (the minimizing-movement objective). Matching on loss_type alone silently
            # averages them into one row.
            if "_precmg" in prefix:
                rec["_loss"] = "pls"
            elif "_mm_" in prefix:
                rec["_loss"] = "min_movement"
            else:
                rec["_loss"] = lt
            m = re.search(r"seed(\d+)", path)
            rec["_seed"] = int(m.group(1)) if m else None
            runs.append(rec)
    return runs


def summarize(runs):
    by = defaultdict(list)
    for r in runs:
        by[(r["_arch"], r["_loss"])].append(r)
    out = {}
    for key, recs in by.items():
        vals = [r["test_st_rel_l2"] for r in recs]
        mean = sum(vals) / len(vals)
        out[key] = (mean, (max(vals) - min(vals)) / 2.0, len(vals), recs)
    return out


def _median_run(recs):
    return sorted(recs, key=lambda r: r["test_st_rel_l2"])[len(recs) // 2]


def plot_curves(summary, path):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for key in ORDER:
        if key not in summary:
            continue
        style = dict(STYLE[key])
        label = style.pop("label")
        rec = _median_run(summary[key][3])
        y = [v * 100 for v in rec["stats"]["val_st_rel_l2"]]
        ax.plot(range(len(y)), y, label=label, linewidth=1.8, **style)
    ax.axhline(100.0, color="#888888", ls="-.", lw=1, zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"validation space-time relative $L^2$ (%)")
    ax.set_title("Allen–Cahn: physics-informed baselines vs preconditioned FEM training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def plot_steps(summary, path):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for key in ORDER:
        if key not in summary:
            continue
        style = dict(STYLE[key])
        label = style.pop("label")
        rec = _median_run(summary[key][3])
        y = [v * 100 for v in rec.get("test_rel_l2_steps", [])]
        if not y:
            continue
        ax.plot(range(1, len(y) + 1), y, label=label, linewidth=1.8, marker="o",
                markersize=3, **style)
    ax.set_yscale("log")
    ax.set_xlabel("rollout step")
    ax.set_ylabel(r"test relative $L^2$ (%)")
    ax.set_title("Error growth along the rollout")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def write_tables(summary, out_dir):
    meta = {"data": ("yes", "--"), "galerkin": ("no", "FEM"), "pls": ("no", "FEM + P"),
            "min_movement": ("no", "FEM (energy)"),
            "pino": ("no", "FD"), "pideeponet": ("no", "autodiff")}
    print(f"\n{'architecture':<14}{'loss':<14}{'labels':<9}{'derivative':<14}"
          f"{'space-time rel-L2 (%)':>24}{'seeds':>7}")
    print("-" * 82)
    rows = []
    for key in ORDER:
        if key not in summary:
            continue
        mean, spread, n, _ = summary[key]
        labels, deriv = meta.get(key[1], ("?", "?"))
        cell = f"{mean * 100:.2f}" if n <= 1 else f"{mean * 100:.2f} ± {spread * 100:.2f}"
        print(f"{key[0]:<14}{key[1]:<14}{labels:<9}{deriv:<14}{cell:>24}{n:>7}")
        rows.append((key, labels, deriv, cell))

    tex = [r"\begin{tabular}{llccr}", r"\toprule",
           r"architecture & loss & labels & derivative & space-time rel.\ $L^2$ (\%) \\",
           r"\midrule"]
    for (arch, loss), labels, deriv, cell in rows:
        name = {"data": r"$L_{\textup{data}}$", "galerkin": r"$L_{\textup{LS}}$",
                "pls": r"$L_{\textup{PLS}}$", "min_movement": r"$L_{\textup{MM}}$",
                "pino": r"PINO", "pideeponet": r"PI-DeepONet"}.get(loss, loss)
        tex.append(f"{arch} & {name} & {labels} & {deriv} & {cell} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    path = os.path.join(out_dir, "table.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(tex) + "\n")
    print(f"\n  -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="output/baselines/allen_cahn")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no Allen-Cahn result JSONs under {args.results_dir}")
    print(f"Loaded {len(runs)} runs from {args.results_dir}")

    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    summary = summarize(runs)
    plot_curves(summary, os.path.join(out_dir, "ac_baselines.png"))
    plot_steps(summary, os.path.join(out_dir, "ac_baselines_steps.png"))
    write_tables(summary, out_dir)

    missing = [k for k in ORDER if k not in summary]
    if missing:
        print("\nNot yet run: " + ", ".join(f"{a}/{l}" for a, l in missing))


if __name__ == "__main__":
    main()
