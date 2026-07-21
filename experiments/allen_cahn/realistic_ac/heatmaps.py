"""realistic_ac rollout heatmaps (single stage, runs on Euler where the checkpoints live).

For a few samples, roll every trained checkpoint forward and render 2D snapshots of the predicted
field next to the **convex-concave (Eyre) reference** as ground truth. One figure per sample: rows
are rollout timesteps, columns are ``[CC | data-driven | min-movement | bare-LS]``, so each model
is directly comparable to the ground truth (and to the others) at each time. The colour scale is
shared per row (symmetric about 0, ``RdBu_r``); each model panel is annotated with its relative L2
vs CC. Diverged panels are drawn grey and flagged.

This needs the actual fields (hence the checkpoints), so it runs on the cluster and emits PNGs
directly. It reuses the IC / CC-reference / rollout code from ``compute_rollout.py`` (imported by
path, so run it plainly).

Usage (on Euler):
    python experiments/allen_cahn/realistic_ac/heatmaps.py --steps 100
    python experiments/allen_cahn/realistic_ac/heatmaps.py --split train --samples 0 3 \
        --runs data ls --snap_steps 0 10 25 50 100
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorpils.meshing import structured_quad_mesh, node_to_grid
from tensorpils.physics import ACProblem
from tensorpils.models import FNOModel

# Import the sibling module by path (not as a package) so this script is immune to where the
# experiment folder lives -- run it plainly with `python .../realistic_ac/heatmaps.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_rollout import (                                          # noqa: E402
    DEFAULT_CFG, loss_key, parse_samples, split_initial_conditions, rollout,
)
from plot_rollout import LOSS_NAME, ORDER                              # noqa: E402


def default_snaps(steps, horizon):
    """3 snapshots inside the training horizon and 3 beyond it (scaled to ``steps``)."""
    h = min(horizon, steps)
    within = [int(round(x)) for x in np.linspace(0, h, 3)]
    beyond = [int(round(x)) for x in np.linspace(h, steps, 4)[1:]] if steps > h else []
    return sorted(set(within + beyond))


def run_matches(rec, tokens):
    """A --runs token matches by loss key ('data'/'mm'/'ls') or as a prefix substring."""
    if not tokens:
        return True
    key, prefix = loss_key(rec), rec["prefix"]
    return any(t == key or t in prefix for t in tokens)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="output/allen_cahn/realistic_ac/results")
    ap.add_argument("--checkpoints_dir", default="output/allen_cahn/realistic_ac/checkpoints")
    ap.add_argument("--out_dir", default="output/allen_cahn/realistic_ac/heatmaps")
    ap.add_argument("--steps", type=int, default=100, help="rollout horizon")
    ap.add_argument("--snap_steps", type=int, nargs="+", default=None,
                    help="timesteps to show as rows (default: 3 in [0,horizon] and 3 beyond)")
    ap.add_argument("--samples", type=int, nargs="+", default=[0, 1, 2],
                    help="sample indices within the split, one figure each")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--runs", type=str, nargs="+", default=None,
                    help="subset of columns: loss keys (data/mm/ls) or prefix substrings (default: all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid_resolution", type=int, default=256)
    ap.add_argument("--ic_r", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.results_dir, "*.json")))]
    if not recs:
        raise SystemExit(f"No results JSONs in {args.results_dir}")

    runs = []
    for r in recs:
        ck = os.path.join(args.checkpoints_dir, r["prefix"] + "_best.pth")
        if os.path.exists(ck) and run_matches(r, args.runs):
            runs.append((r, ck))
    if not runs:
        raise SystemExit("No matching checkpoints (check --runs / --checkpoints_dir)")
    order = {k: i for i, k in enumerate(ORDER)}
    runs.sort(key=lambda rc: order.get(loss_key(rc[0]), 99))

    r0 = runs[0][0]
    a, dt, K, n_train = r0["a"], r0["dt"], r0["K"], r0["n_train"]
    eps = float(r0["eps"])
    _, n_val, n_test = parse_samples(r0["prefix"])
    total = n_train + n_val + n_test
    nx = ny = args.grid_resolution
    horizon = r0.get("rollout_steps", 10)
    snaps = sorted(set(s for s in (args.snap_steps or default_snaps(args.steps, horizon)) if s <= args.steps))

    mesh = structured_quad_mesh(nx, ny)
    prob = ACProblem(mesh).to(device)
    n_cap = max(args.samples) + 1
    ics = split_initial_conditions(args.split, total, n_train, n_val, n_test, K, args.seed, mesh,
                                   args.ic_r, n_cap).to(device)                   # [n_cap, N]
    cc = prob.fem_reference(ics, a=a, eps=eps, dt=dt, n_steps=args.steps,
                            integrator="convex_concave")                          # [n_cap, steps+1, N]
    cc_grid0 = node_to_grid(cc[:, 0], nx, ny)

    preds_by_run = []
    for rec, ck_path in runs:
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model = FNOModel(**(ck.get("model_config") or DEFAULT_CFG)).to(device).float()
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.eval()
        preds = rollout(model, cc_grid0, prob, nx, ny, args.steps, device)        # [n_cap, steps, H, W]
        preds_by_run.append((rec, preds.double().cpu().numpy()))

    def field_at(step, sample, pr):
        return node_to_grid(cc[sample:sample + 1, step], nx, ny)[0].cpu().numpy() if step == 0 \
            else pr[sample, step - 1]

    for s in args.samples:
        if s >= n_cap:
            continue
        ncol = 1 + len(runs)
        fig, ax = plt.subplots(len(snaps), ncol, figsize=(2.1 * ncol, 2.3 * len(snaps)), squeeze=False)
        for row, step in enumerate(snaps):
            cc_f = node_to_grid(cc[s:s + 1, step], nx, ny)[0].cpu().numpy()
            model_fs = [field_at(step, s, pr) for _, pr in preds_by_run]
            allf = np.stack([cc_f] + model_fs)
            vlim = float(np.nanmax(np.abs(allf))) or 1.0                          # symmetric, white=0
            for col in range(ncol):
                a_ = ax[row, col]
                f = cc_f if col == 0 else model_fs[col - 1]
                a_.set_facecolor("0.85")                                          # shows through NaNs
                a_.imshow(np.ma.masked_invalid(f), cmap="RdBu_r", origin="lower", vmin=-vlim, vmax=vlim)
                a_.set_xticks([]); a_.set_yticks([])
                if col == 0:
                    a_.set_ylabel(f"step {step}", fontsize=10)
                elif not np.all(np.isfinite(f)):
                    a_.text(0.5, 0.5, "diverged", color="0.25", ha="center", va="center",
                            transform=a_.transAxes, fontsize=10, fontweight="bold")
                else:
                    rel = np.sqrt(((f - cc_f) ** 2).sum() / max((cc_f ** 2).sum(), 1e-30)) * 100
                    a_.text(0.97, 0.04, f"{rel:.1f}%", color="0.15", ha="right", va="bottom",
                            transform=a_.transAxes, fontsize=8,
                            bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.6))
                if row == 0:
                    a_.set_title("CC reference" if col == 0 else LOSS_NAME.get(loss_key(preds_by_run[col - 1][0])),
                                 fontsize=9)
        fig.suptitle(f"realistic_ac fields  (eps={eps:g}, {args.split} sample #{s}, {args.steps} steps)",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        p = os.path.join(args.out_dir, f"heatmap_{args.split}_sample{s}.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f"  {p}")


if __name__ == "__main__":
    main()
