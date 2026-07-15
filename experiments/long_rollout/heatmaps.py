"""Long-rollout heatmaps (single stage, runs on Euler where the checkpoints live).

For a few test samples, roll every trained checkpoint forward and render 2D snapshots of the
predicted field next to the **convex-concave (Eyre) reference** as ground truth. One figure per
(eps, sample): rows are rollout timesteps, columns are ``[CC | run1 | run2 | ...]``, so each model
is directly comparable to the ground truth (and to the others) at each time. The colour scale is
shared per row (symmetric about 0, ``RdBu_r``) and each model panel is annotated with its relative
L2 vs CC at that step. Diverged panels are drawn grey and flagged.

Unlike ``compute_long_rollout.py`` (scalar metrics -> tiny JSON, plotted anywhere) this needs the
actual fields, hence the checkpoints -- so it runs on the cluster and emits PNGs directly. Download
the finished figures (or a few checkpoints) when assembling paper figures.

It reuses the IC regeneration, CC reference and rollout from ``compute_long_rollout.py`` verbatim.

Usage (on Euler):
    python experiments/long_rollout/heatmaps.py --steps 20            # all eps, samples 0 1 2, all runs
    python experiments/long_rollout/heatmaps.py --eps 16 --samples 0 3 \
        --runs mm:pushforward ls:full_bptt data:pushforward --snap_steps 0 5 10 15 20
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorpils.meshing import structured_quad_mesh, node_to_grid
from tensorpils.physics import ACProblem
from tensorpils.models import FNOModel

from experiments.long_rollout.compute_long_rollout import (
    DEFAULT_CFG, loss_key, parse_samples, test_initial_conditions, rollout,
)
from experiments.long_rollout.plot_long_rollout import LOSS_NAME, MODE_NAME, ORDER


def default_snaps(steps):
    """3 snapshots inside the 10-step training horizon and 3 beyond it (scaled to ``steps``)."""
    h = min(10, steps)
    within = [int(round(x)) for x in np.linspace(0, h, 3)]
    beyond = [int(round(x)) for x in np.linspace(h, steps, 4)[1:]] if steps > h else []
    return sorted(set(within + beyond))


def run_key(rec):
    return f"{loss_key(rec)}:{rec.get('bptt_mode')}"


def run_matches(rec, tokens):
    """A --runs token matches a run by exact 'loss:mode' key, by loss alone, or as a prefix substring."""
    if not tokens:
        return True
    key, loss, prefix = run_key(rec), loss_key(rec), rec["prefix"]
    return any(t == key or t == loss or t in prefix for t in tokens)


def col_label(rec):
    return f"{LOSS_NAME[loss_key(rec)]}\n{MODE_NAME.get(rec.get('bptt_mode'), rec.get('bptt_mode'))}"


def field_at(step, sample, cc, preds_grid, nx, ny):
    """Grid field [H, W] for one sample at rollout ``step``; step 0 is the shared IC."""
    if step == 0:
        return node_to_grid(cc[sample:sample + 1, 0], nx, ny)[0]
    return preds_grid[sample, step - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="output/compare_bptt/results")
    ap.add_argument("--checkpoints_dir", default="output/compare_bptt/checkpoints")
    ap.add_argument("--out_dir", default="output/long_rollout/heatmaps")
    ap.add_argument("--steps", type=int, default=20, help="rollout horizon")
    ap.add_argument("--snap_steps", type=int, nargs="+", default=None,
                    help="timesteps to show as rows (default: 3 in [0,10] and 3 beyond)")
    ap.add_argument("--samples", type=int, nargs="+", default=[0, 1, 2],
                    help="test-sample indices, one figure each")
    ap.add_argument("--eps", type=float, nargs="+", default=None, help="eps to plot (default: all)")
    ap.add_argument("--runs", type=str, nargs="+", default=None,
                    help="subset of columns: 'loss:mode' keys, a loss, or prefix substrings (default: all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid_resolution", type=int, default=64)
    ap.add_argument("--ic_r", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    snaps = sorted(set(s for s in (args.snap_steps or default_snaps(args.steps)) if s <= args.steps))
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.results_dir, "*.json")))]
    if not recs:
        raise SystemExit(f"No results JSONs in {args.results_dir}")

    by_eps = {}
    for r in recs:
        ck = os.path.join(args.checkpoints_dir, r["prefix"] + "_best.pth")
        if os.path.exists(ck) and run_matches(r, args.runs):
            by_eps.setdefault(float(r["eps"]), []).append((r, ck))

    order = {k: i for i, k in enumerate(ORDER)}
    for eps in sorted(by_eps):
        if args.eps is not None and eps not in args.eps:
            continue
        runs = sorted(by_eps[eps], key=lambda rc: order.get((loss_key(rc[0]), rc[0].get("bptt_mode")), 99))
        r0 = runs[0][0]
        a, dt, K, n_train = r0["a"], r0["dt"], r0["K"], r0["n_train"]
        _, n_val, n_test = parse_samples(r0["prefix"])
        total = n_train + n_val + n_test
        nx = ny = args.grid_resolution

        mesh = structured_quad_mesh(nx, ny)
        prob = ACProblem(mesh).to(device)
        n_cap = max(args.samples) + 1
        ics = test_initial_conditions(total, n_train, n_val, n_test, K, args.seed, mesh, args.ic_r,
                                      n_cap).to(device)                              # [n_cap, N]
        cc = prob.fem_reference(ics, a=a, eps=eps, dt=dt, n_steps=args.steps,
                                integrator="convex_concave")                        # [n_cap, steps+1, N]
        cc_grid0 = node_to_grid(cc[:, 0], nx, ny)

        preds_by_run = []
        for rec, ck_path in runs:
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            model = FNOModel(**(ck.get("model_config") or DEFAULT_CFG)).to(device).float()
            model.load_state_dict(ck["model_state_dict"], strict=False)
            model.eval()
            preds = rollout(model, cc_grid0, prob, nx, ny, args.steps, device)       # [n_cap, steps, H, W]
            preds_by_run.append((rec, preds.double().cpu().numpy()))
        cc_np = cc.cpu().numpy()

        for s in args.samples:
            if s >= n_cap:
                continue
            ncol = 1 + len(runs)
            fig, ax = plt.subplots(len(snaps), ncol, figsize=(2.1 * ncol, 2.3 * len(snaps)),
                                   squeeze=False)
            for row, step in enumerate(snaps):
                cc_f = field_at(step, s, cc_np, None, nx, ny) if step == 0 else \
                    node_to_grid(cc[s:s + 1, step], nx, ny)[0].cpu().numpy()
                model_fs = [field_at(step, s, cc_np, pr, nx, ny) for _, pr in preds_by_run]
                allf = np.stack([cc_f] + model_fs)
                vlim = float(np.nanmax(np.abs(allf))) or 1.0                         # symmetric, white=0

                for col in range(ncol):
                    a_ = ax[row, col]
                    f = cc_f if col == 0 else model_fs[col - 1]
                    a_.set_facecolor("0.85")                                         # shows through NaNs
                    a_.imshow(np.ma.masked_invalid(f), cmap="RdBu_r", origin="lower",
                              vmin=-vlim, vmax=vlim)
                    a_.set_xticks([]); a_.set_yticks([])
                    if col == 0:
                        a_.set_ylabel(f"step {step}", fontsize=10)
                    else:
                        if not np.all(np.isfinite(f)):
                            a_.text(0.5, 0.5, "diverged", color="0.25", ha="center", va="center",
                                    transform=a_.transAxes, fontsize=10, fontweight="bold")
                        else:
                            rel = np.sqrt(((f - cc_f) ** 2).sum() / max((cc_f ** 2).sum(), 1e-30)) * 100
                            a_.text(0.97, 0.04, f"{rel:.1f}%", color="0.15", ha="right", va="bottom",
                                    transform=a_.transAxes, fontsize=8,
                                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.6))
                    if row == 0:
                        a_.set_title("CC reference" if col == 0 else col_label(preds_by_run[col - 1][0]),
                                     fontsize=9)
            fig.suptitle(f"Long-rollout fields  (eps={eps:g}, test sample #{s}, {args.steps} steps)",
                         fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            p = os.path.join(args.out_dir, f"heatmap_eps{eps:g}_sample{s}.png")
            fig.savefig(p, dpi=150); plt.close(fig)
            print(f"  {p}")


if __name__ == "__main__":
    main()
