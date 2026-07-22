"""realistic_ac rollout analysis (stage 1: compute, runs on Euler where the checkpoints live).

For every trained checkpoint (one per training loss in the realistic_ac sweep), roll the FNO
forward for ``--steps`` steps (far past the 10-step training horizon) and record, averaged over a
split, both

  * energy(step)  -- Ginzburg-Landau energy of the predicted frame. Allen-Cahn is a gradient flow,
                     so it should be non-increasing; a climbing curve means the rollout blew up.
                     The convex-concave reference energy is stored per split as a baseline.
  * error(step)   -- relative FEM-L2 of the predicted frame vs the convex-concave (Eyre) reference
                     for the SAME initial conditions (ground truth from the learned stepper's view).

Unlike long_rollout this reports **both the test and the train split** (so the generalization gap
is visible) and assumes a single eps / single bptt mode (realistic_ac fixes eps=64, pushforward),
keying runs by loss only. Writes a compact ``metrics.json`` (numbers only) so the heavy checkpoints
never leave Euler; plot from it anywhere with ``plot_rollout.py``.

Usage (on Euler, after the realistic_ac checkpoints exist):
    python experiments/allen_cahn/realistic_ac/compute_rollout.py \
        --results_dir output/allen_cahn/realistic_ac/results \
        --checkpoints_dir output/allen_cahn/realistic_ac/checkpoints \
        --steps 100 --n_samples 128
"""

import argparse
import glob
import json
import os
import re

import torch

from tensorpils.meshing import structured_quad_mesh, node_to_grid, grid_to_node
from tensorpils.physics import ACProblem, apply_zero_boundary
from tensorpils.models import FNOModel
from tensormesh.dataset import WaveMultiFrequency

DEFAULT_CFG = dict(n_modes=(16, 16), hidden_channels=64, in_channels=1, out_channels=1, n_layers=5)


def loss_key(rec):
    """data / mm (minimizing-movement) / ls (bare least-squares) / pls (preconditioned LS)."""
    if rec.get("loss_type") == "data" or (rec.get("lambda_data") and not rec.get("lambda_galerkin")):
        return "data"
    if rec.get("ac_loss_form") == "min_movement":
        return "mm"
    return "pls" if rec.get("ac_precond") else "ls"


def parse_samples(prefix):
    m = re.search(r"samples-(\d+)-(\d+)-(\d+)", prefix)
    return tuple(int(x) for x in m.groups()) if m else (None, None, None)


def split_initial_conditions(split, total, n_train, n_val, n_test, K, seed, mesh, r, n_cap):
    """Regenerate a split's ICs exactly as ACDataset does (deterministic; no reference solve).

    Splits are the nested index ranges create_ac_datasets uses: train = [0, n_train),
    test = [n_train+n_val, n_train+n_val+n_test)."""
    torch.manual_seed(seed)
    l_a = (torch.rand(total, K, K) * 2 - 1)
    u0 = WaveMultiFrequency(a=l_a, r=r).initial_condition(mesh.points).double()   # [total, N]
    if split == "train":
        sel = u0[:n_train]
    elif split == "test":
        sel = u0[n_train + n_val: n_train + n_val + n_test]
    else:
        raise ValueError(f"unknown split {split!r}")
    return sel[:n_cap]


@torch.no_grad()
def rollout(model, ic_grid, prob, nx, ny, steps, device):
    """Batched free-running rollout. ic_grid [B, H, W] -> preds [B, steps, H, W] (boundary-zeroed)."""
    mask = prob.boundary_mask.to(device)

    def proj(g):
        return node_to_grid(apply_zero_boundary(grid_to_node(g, nx, ny), mask), nx, ny)

    mdtype = next(model.parameters()).dtype                       # roll out in the model's dtype (fp32)
    window = [proj(ic_grid.to(device=device, dtype=mdtype))]
    preds = []
    for _ in range(steps):
        nxt = proj(model(torch.stack(window, dim=1)).squeeze(1))
        preds.append(nxt)
        window = window[1:] + [nxt]
    return torch.stack(preds, dim=1)


def _jsonable(vec):
    """Map non-finite entries to None (valid JSON) and return (list, first_bad_index or -1)."""
    out, bad = [], -1
    for i, x in enumerate(vec):
        f = float(x)
        if f != f or f in (float("inf"), float("-inf")):
            out.append(None)
            bad = i if bad < 0 else bad
        else:
            out.append(f)
    return out, bad


def split_metrics(split, runs, cfg, args, device):
    """Compute {ref_energy, runs:[{loss, energy, error, diverged, ...}]} for one split."""
    a, dt, K, eps, n_train, n_val, n_test = cfg
    total = n_train + n_val + n_test
    nx = ny = args.grid_resolution

    mesh = structured_quad_mesh(nx, ny)
    prob = ACProblem(mesh).to(device)
    M = prob.M
    ics = split_initial_conditions(split, total, n_train, n_val, n_test, K, args.seed, mesh,
                                   args.ic_r, args.n_samples).to(device)          # [B, N]
    cc = prob.fem_reference(ics, a=a, eps=eps, dt=dt, n_steps=args.steps,
                            integrator="convex_concave")                          # [B, steps+1, N]
    ref_E = [prob.energy(cc[:, k], a=a, eps=eps, reduce="none").mean().item()
             for k in range(args.steps + 1)]
    cc_grid0 = node_to_grid(cc[:, 0], nx, ny)
    den = [(cc[:, k] * prob._spmm(M, cc[:, k])).sum(-1).clamp_min(1e-30) for k in range(args.steps + 1)]

    print(f"  {split}: {ics.shape[0]} samples, {len(runs)} runs, {args.steps} steps")
    out_runs = []
    for rec, ck_path in runs:
        ck = torch.load(ck_path, map_location=device, weights_only=False)         # our own trusted files
        model = FNOModel(**(ck.get("model_config") or DEFAULT_CFG)).to(device).float()  # trained fp32
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.eval()
        preds = rollout(model, cc_grid0, prob, nx, ny, args.steps, device)        # [B, steps, H, W] fp32
        pn = grid_to_node(preds, nx, ny).double()                                # [B, steps, N] fp64 math

        energy, error = [ref_E[0]], [0.0]                                        # frame 0 = shared IC
        for k in range(args.steps):
            uk = pn[:, k]
            energy.append(prob.energy(uk, a=a, eps=eps, reduce="none").mean().item())
            e = uk - cc[:, k + 1]
            num = (e * prob._spmm(M, e)).sum(-1).clamp_min(0)
            error.append((num / den[k + 1]).sqrt().mean().item())
        E_list, E_bad = _jsonable(energy)
        err_list, err_bad = _jsonable(error)
        diverge = max((b for b in (E_bad, err_bad) if b >= 0), default=-1)
        out_runs.append({
            "loss": loss_key(rec), "mode": rec.get("bptt_mode"), "prefix": rec["prefix"],
            "energy": E_list, "error": err_list,
            "diverged": diverge >= 0, "diverge_step": diverge,
        })
    return {"ref_energy": ref_E, "runs": out_runs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="output/allen_cahn/realistic_ac/results")
    ap.add_argument("--checkpoints_dir", default="output/allen_cahn/realistic_ac/checkpoints")
    ap.add_argument("--out", default="output/allen_cahn/realistic_ac/metrics.json")
    ap.add_argument("--steps", type=int, default=100, help="rollout horizon (>= training's 10)")
    ap.add_argument("--n_samples", type=int, default=128,
                    help="cap on samples per split to average over (the CC reference to --steps is "
                         "the expensive part at 256^2, so this is deliberately modest)")
    ap.add_argument("--splits", nargs="+", default=["test", "train"], choices=["test", "train"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid_resolution", type=int, default=256)
    ap.add_argument("--ic_r", type=float, default=0.5, help="WaveMultiFrequency decay r (ACDataset default)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.results_dir, "*.json")))]
    if not recs:
        raise SystemExit(f"No results JSONs in {args.results_dir}")

    runs = []
    for r in recs:
        ck = os.path.join(args.checkpoints_dir, r["prefix"] + "_best.pth")
        if os.path.exists(ck):
            runs.append((r, ck))
    if not runs:
        raise SystemExit(f"No matching *_best.pth in {args.checkpoints_dir}")

    r0 = runs[0][0]
    eps_set = {float(r["eps"]) for r, _ in runs}
    if len(eps_set) != 1:
        raise SystemExit(f"realistic_ac expects a single eps; found {sorted(eps_set)}")
    eps = eps_set.pop()
    _, n_val, n_test = parse_samples(r0["prefix"])
    cfg = (r0["a"], r0["dt"], r0["K"], eps, r0["n_train"], n_val, n_test)

    device = args.device
    print(f"eps={eps:g}, {len(runs)} runs, splits={args.splits}")
    out = {"steps": args.steps, "n_samples": args.n_samples, "eps": eps,
           "a": r0["a"], "dt": r0["dt"], "K": r0["K"], "grid": args.grid_resolution,
           "train_horizon": r0.get("rollout_steps", 10), "splits": {}}
    for split in args.splits:
        out["splits"][split] = split_metrics(split, runs, cfg, args, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"\nmetrics -> {args.out}  "
          f"({len(out['splits'])} splits x {len(runs)} runs, {args.steps} steps)")


if __name__ == "__main__":
    main()
