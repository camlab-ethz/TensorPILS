"""Aggregate the Stokes h-refinement sweep: test error vs mesh, one line per loss.

Reads the per-run ``results/*.json`` written by ``StokesTrainer._save_results_json`` and produces

  1. ``h_refinement.png`` — velocity and pressure relative FE-L2 vs the velocity grid (log-log),
     one line per ``--loss``, with the "predict zero" level (100%) marked: the bare least-squares
     arm lives near it, which is the whole point of having it in the sweep;
  2. ``h_refinement_curves.png`` — the underlying validation curves (combined rel-L2 vs epoch),
     grouped by grid, so a non-monotone final number can be read as "never converged" rather
     than "converged to something worse";
  3. a LaTeX table on stdout (and in ``table.tex``) for the paper.

Usage:
    python experiments/stokes/h_refinement/plot_h_refinement.py \
        --results_dir output/stokes/h_refinement/results
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One colour/marker per loss, shared by both figures.
STYLE = {
    "data":     dict(color="#111111", marker="o", label=r"$L_{\rm data}$ (supervised)"),
    "pls":      dict(color="#0e7490", marker="s", label=r"$L_{\rm PLS}$  $\frac{1}{2} r^\top\!Pr$"),
    "galerkin": dict(color="#b91c1c", marker="^", label=r"$L_{\rm LS}$  $\frac{1}{2}\|r\|^2$"),
}
ORDER = ["data", "pls", "galerkin"]


def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as fh:
            r = json.load(fh)
        if r.get("pde") == "stokes":
            runs.append(r)
    return runs


def by_loss(runs):
    """``{loss_type: [(grid, run), ...]}`` sorted by grid."""
    out = {}
    for r in runs:
        out.setdefault(r["loss_type"], []).append((r["grid"][0], r))
    for v in out.values():
        v.sort(key=lambda t: t[0])
    return out


def plot_refinement(groups, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in ((axes[0], "test_rel_l2_u", "velocity"),
                           (axes[1], "test_rel_l2_p", "pressure")):
        for loss in ORDER:
            if loss not in groups:
                continue
            grids = [g for g, _ in groups[loss]]
            errs = [100.0 * r[key] for _, r in groups[loss]]
            ax.plot(grids, errs, lw=1.8, ms=6, **STYLE[loss])
        ax.axhline(100.0, color="#888888", ls=":", lw=1.2)
        ax.text(0.02, 0.965, "predicting zero", transform=ax.transAxes,
                fontsize=8, color="#888888", va="top")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("velocity grid $N$  (pressure $(N{+}1)/2$)")
        ax.set_ylabel(f"test relative FE-$L^2$, {title} [%]")
        ax.set_title(f"Stokes {title} error under refinement")
        ax.grid(alpha=0.3, which="both")
        if grids:
            ax.set_xticks(grids)
            ax.set_xticklabels([str(g) for g in grids])
    axes[0].legend(fontsize=8.5, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"figure -> {save_path}")


def plot_curves(groups, save_path):
    grids = sorted({g for v in groups.values() for g, _ in v})
    fig, axes = plt.subplots(1, len(grids), figsize=(4.2 * len(grids), 3.8), squeeze=False)
    for ax, grid in zip(axes[0], grids):
        for loss in ORDER:
            for g, r in groups.get(loss, []):
                if g != grid:
                    continue
                y = [100.0 * v for v in r["stats"]["val_rel_l2_errors"]]
                ax.plot(range(1, len(y) + 1), y, lw=1.3, color=STYLE[loss]["color"],
                        label=STYLE[loss]["label"])
        ax.axhline(100.0, color="#888888", ls=":", lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"val rel-$L^2$, mean of $u,p$ [%]")
        ax.set_title(f"velocity grid ${grid}^2$")
        ax.grid(alpha=0.3, which="both")
    axes[0][0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"figure -> {save_path}")


def latex_table(groups):
    grids = sorted({g for v in groups.values() for g, _ in v})
    head = " & ".join(f"$u$ & $p$" for _ in grids)
    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Stokes under mesh refinement: test relative FE-$L^2$ error of velocity and "
        r"pressure. Identical FNO, optimiser, schedule, seed, sample count ($K=4$, 512 training "
        r"samples, 300 epochs) at every resolution --- only the grid and the loss change. "
        r"$100\%$ is what predicting $0$ scores.}",
        r"\label{tab:stokes_refinement}",
        r"\begin{tabular}{l" + "cc" * len(grids) + "}", r"\toprule",
        "Loss & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{${g}^2$}}" for g in grids) + r" \\",
        r"\cmidrule(lr){2-" + str(1 + 2 * len(grids)) + "}",
        " & " + head + r" \\", r"\midrule",
    ]
    names = {"data": r"$L_{\textup{data}}$ (supervised)",
             "galerkin": r"$L_{\textup{LS}}$ (bare)",
             "pls": r"$L_{\textup{PLS}}$ (preconditioned)"}
    for loss in ORDER:
        if loss not in groups:
            continue
        d = {g: r for g, r in groups[loss]}
        cells = []
        for g in grids:
            r = d.get(g)
            cells += ["---", "---"] if r is None else [
                f"${100*r['test_rel_l2_u']:.2f}$", f"${100*r['test_rel_l2_p']:.2f}$"]
        lines.append(f"{names.get(loss, loss)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\smallskip", r"\footnotesize All entries in \%.", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results_dir", default="output/stokes/h_refinement/results")
    ap.add_argument("--out_dir", default=None, help="Default: alongside --results_dir.")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no Stokes result JSONs in {args.results_dir}")
    groups = by_loss(runs)
    out_dir = args.out_dir or os.path.dirname(args.results_dir.rstrip("/")) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"{len(runs)} runs: " + ", ".join(
        f"{loss}@{[g for g, _ in v]}" for loss, v in sorted(groups.items())))
    plot_refinement(groups, os.path.join(out_dir, "h_refinement.png"))
    plot_curves(groups, os.path.join(out_dir, "h_refinement_curves.png"))
    tex = latex_table(groups)
    print("\n" + tex + "\n")
    with open(os.path.join(out_dir, "table.tex"), "w") as fh:
        fh.write(tex + "\n")
    print(f"table -> {os.path.join(out_dir, 'table.tex')}")


if __name__ == "__main__":
    main()
