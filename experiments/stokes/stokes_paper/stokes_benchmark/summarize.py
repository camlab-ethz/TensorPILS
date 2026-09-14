"""Aggregate the Stokes Table-1 benchmark and print the four table rows.

Layout (see README.md):
    <root>/lr_sweep/<arm>/<cell>/results/*.json          stage 1, seed 42
    <root>/final/<arm>/<cell>/seed<S>/results/*.json      stage 2, three seeds
with <cell> = lr<lr>[_dw<w>][_lam<lam>] carrying every swept hyperparameter (the run prefix
encodes neither lr nor seed).

Selection: per arm, the cell with the lowest best VALIDATION error, where the validation error is
the mean of the velocity and pressure relative FE-L2 (``stats.val_rel_l2_errors``, the trainer's
own model-selection metric). Test errors are printed for information only.

Modes:
    python summarize.py                 full report (stage 1 ranking per arm, stage 2 per seed,
                                        LaTeX rows for tab:summary, ARMS lines for sweep.sbatch)
    python summarize.py --select        one machine-readable line per arm: "arm lr w lam"
                                        (x where unused) -- consumed by run_local.sh
"""

import argparse
import glob
import json
import os
import re
import statistics

ARMS = ["pls", "data", "pino", "pideeponet"]            # Table 1 row order
LABEL = {"pls": ("$L_\\textup{PLS}$", "yes", "FEM"), "data": ("$L_\\textup{data}$", "no", "---"),
         "pino": ("PINO", "yes", "FD"), "pideeponet": ("PI-DeepONet", "yes", "AD")}
CELL_RE = re.compile(r"^lr(?P<lr>[0-9.e+-]+)(?:_(?:dw|om)(?P<w>[0-9.e+-]+))?(?:_lam(?P<lam>[0-9.e+-]+))?$")


def parse_cell(name):
    m = CELL_RE.match(name)
    if not m:
        raise ValueError(f"cannot parse cell directory name {name!r}")
    return {k: (v if v is not None else "x") for k, v in m.groupdict().items()}


def load(path):
    d = json.load(open(path))
    st = d["stats"]
    v = st["val_rel_l2_errors"]
    i = min(range(len(v)), key=v.__getitem__)
    times = st.get("epoch_times") or []
    return dict(best_val=v[i], best_epoch=i, n_epochs=len(v),
                val_u=st["val_rel_l2_u"][i], val_p=st["val_rel_l2_p"][i],
                test_u=float(d["test_rel_l2_u"]), test_p=float(d["test_rel_l2_p"]),
                s_per_epoch=(statistics.median(times) if times else None))


def stage1(root, arm):
    rows = []
    for path in glob.glob(os.path.join(root, "lr_sweep", arm, "*", "results", "*.json")):
        cell = os.path.basename(os.path.dirname(os.path.dirname(path)))
        rows.append((cell, parse_cell(cell), load(path)))
    rows.sort(key=lambda r: r[2]["best_val"])
    return rows


def edges(rows, sel):
    """Hyperparameters whose selected value is the smallest or largest tried for this arm."""
    out = []
    for key in ("lr", "w", "lam"):
        vals = sorted({float(h[key]) for _, h, _ in rows if h[key] != "x"})
        if len(vals) > 1 and float(sel[key]) in (vals[0], vals[-1]):
            out.append(f"{key}={sel[key]} is the {'smallest' if float(sel[key]) == vals[0] else 'largest'} tried")
    return out


def stage2(root, arm, cell):
    runs = []
    for path in sorted(glob.glob(os.path.join(root, "final", arm, cell, "seed*", "results", "*.json"))):
        runs.append((os.path.basename(os.path.dirname(os.path.dirname(path))), load(path)))
    return runs


def mean_half(xs):
    return sum(xs) / len(xs), 0.5 * (max(xs) - min(xs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/stokes/stokes_paper/stokes_benchmark")
    ap.add_argument("--select", action="store_true", help="print 'arm lr w lam' per arm and exit")
    args = ap.parse_args()

    selected = {}
    for arm in ARMS:
        rows = stage1(args.root, arm)
        if rows:
            selected[arm] = rows[0]
        if args.select:
            if rows:
                h = rows[0][1]
                print(f"{arm} {h['lr']} {h['w']} {h['lam']}")
            continue
        print(f"\n=== {arm}: stage 1 (seed 42, 500 epochs), ranked by best validation mean(u, p) ===")
        if not rows:
            print("  (no runs)"); continue
        print(f"  {'cell':<26} {'best-val':>8} {'val u':>7} {'val p':>7} {'@ep':>4} {'test u':>7} {'test p':>7} {'s/ep':>6}")
        for cell, _, r in rows:
            t = f"{r['s_per_epoch']:6.2f}" if r["s_per_epoch"] else "     -"
            print(f"  {cell:<26} {100*r['best_val']:7.2f}% {100*r['val_u']:6.2f}% {100*r['val_p']:6.2f}% "
                  f"{r['best_epoch']:>4} {100*r['test_u']:6.2f}% {100*r['test_p']:6.2f}% {t}")
        cell, h, r = rows[0]
        print(f"  selected: {cell}   (lr={h['lr']}, {'omega' if arm == 'pls' else 'w'}={h['w']}, lam={h['lam']})")
        for e in edges(rows, h):
            print(f"  EDGE: {e} -- extend the grid in that direction before trusting this selection")
        if r["best_epoch"] >= r["n_epochs"] - 5:
            print("  note: best epoch within 5 of the budget -- still descending, budget-limited")
    if args.select:
        return

    latex, arms_lines = [], {"LR": [], "W": [], "LAM": []}
    print("\n=== stage 2 (three seeds at the selected cell; test error at the best-validation checkpoint) ===")
    for arm in ARMS:
        if arm not in selected:
            print(f"  {arm}: no stage-1 selection"); continue
        cell, h, _ = selected[arm]
        arms_lines["LR"].append(h["lr"]); arms_lines["W"].append(h["w"]); arms_lines["LAM"].append(h["lam"])
        runs = stage2(args.root, arm, cell)
        others = [os.path.basename(p) for p in glob.glob(os.path.join(args.root, "final", arm, "*")) if os.path.basename(p) != cell]
        print(f"\n  {arm} @ {cell}" + (f"   (other final/ cells present, not selected: {', '.join(others)})" if others else ""))
        if not runs:
            print("    (no stage-2 runs yet)"); continue
        for seed, r in runs:
            t = f"  {r['s_per_epoch']:.2f} s/epoch" if r["s_per_epoch"] else ""
            print(f"    {seed}: test u {100*r['test_u']:6.2f}%  p {100*r['test_p']:6.2f}%   best-val {100*r['best_val']:6.2f}% @ {r['best_epoch']}{t}")
        mu, hu = mean_half([r["test_u"] for _, r in runs]); mp, hp = mean_half([r["test_p"] for _, r in runs])
        times = [r["s_per_epoch"] for _, r in runs if r["s_per_epoch"]]
        tcell = f"{statistics.median(times):.1f}" if times else "---"
        print(f"    -> u {mu:.3f} +/- {hu:.3f}   p {mp:.3f} +/- {hp:.3f}   ({len(runs)} seeds)"
              + ("   WARNING: fewer than 3 seeds" if len(runs) < 3 else ""))
        name, pi, der = LABEL[arm]
        latex.append(f"      & {name} & {pi} & {der} & ${mu:.3f} \\pm {hu:.3f}$ / ${mp:.3f} \\pm {hp:.3f}$ & {tcell} \\\\")
    if latex:
        print("\nLaTeX rows for tab:summary (Stokes block; cells are velocity / pressure, time is median s/epoch):")
        print("\n".join(latex))
    if arms_lines["LR"]:
        print("\nsweep.sbatch (order pls data pino pideeponet):")
        for k, v in arms_lines.items():
            print(f"ARMS_{k}=( {' '.join(v)} )")


if __name__ == "__main__":
    main()
