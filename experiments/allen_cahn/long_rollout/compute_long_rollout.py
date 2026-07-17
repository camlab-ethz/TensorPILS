"""Long-horizon rollout analysis for the compare_bptt models (stage 1: compute, runs on Euler).

For every trained checkpoint (globbed from a compare_bptt run), roll the FNO forward for
``--steps`` steps (longer than the 10-step training horizon) from the test-set initial conditions,
and record, averaged over the test fraction:

  * energy(step)  -- the Ginzburg-Landau energy of the predicted frame (should decrease; a blowing-
                     up rollout climbs). The convex-concave reference energy is stored as a baseline.
  * error(step)   -- relative FEM-L2 of the predicted frame vs the convex-concave (Eyre) reference
                     trajectory generated for the same initial conditions (the ground truth from
                     the learned stepper's perspective; steps > 10 are beyond the training horizon).

Writes a compact ``metrics.json`` (numbers only -- KB) so the heavy checkpoints never leave Euler;
plot from it anywhere with ``plot_long_rollout.py``. Non-finite values (diverged rollouts) are
stored as ``null`` and flagged per run, so divergence is visible rather than dropped.

Usage (on Euler, after the compare_bptt checkpoints exist):
    python experiments/allen_cahn/long_rollout/compute_long_rollout.py \
        --results_dir output/allen_cahn/compare_bptt/results \
        --checkpoints_dir output/allen_cahn/compare_bptt/checkpoints \
        --steps 20        # horizon (argparse; cheap to change)
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
    if rec.get("loss_type") == "data" or (rec.get("lambda_data") and not rec.get("lambda_galerkin")):
        return "data"
    return "mm" if rec.get("ac_loss_form") == "min_movement" else "ls"


def parse_samples(prefix):
    m = re.search(r"samples-(\d+)-(\d+)-(\d+)", prefix)
    return tuple(int(x) for x in m.groups()) if m else (None, None, None)


def test_initial_conditions(total, n_train, n_val, n_test, K, seed, mesh, r, n_cap):
    """Regenerate the test-split ICs exactly as ACDataset does (deterministic; no reference solve)."""
    torch.manual_seed(seed)
    l_a = (torch.rand(total, K, K) * 2 - 1)
    u0 = WaveMultiFrequency(a=l_a, r=r).initial_condition(mesh.points).double()   # [total, N]
    test = u0[n_train + n_val: n_train + n_val + n_test]
    return test[:n_cap]


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="output/allen_cahn/compare_bptt/results")
    ap.add_argument("--checkpoints_dir", default="output/allen_cahn/compare_bptt/checkpoints")
    ap.add_argument("--out", default="output/allen_cahn/long_rollout/metrics.json")
    ap.add_argument("--steps", type=int, default=20, help="rollout horizon (>= training's 10)")
    ap.add_argument("--n_samples", type=int, default=256, help="cap on test samples to average over")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid_resolution", type=int, default=64)
    ap.add_argument("--ic_r", type=float, default=0.5, help="WaveMultiFrequency decay r (ACDataset default)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.results_dir, "*.json")))]
    if not recs:
        raise SystemExit(f"No results JSONs in {args.results_dir}")

    by_eps = {}
    for r in recs:
        ck = os.path.join(args.checkpoints_dir, r["prefix"] + "_best.pth")
        if os.path.exists(ck):
            by_eps.setdefault(float(r["eps"]), []).append((r, ck))

    device = args.device
    out = {"steps": args.steps, "n_samples": args.n_samples, "by_eps": {}}

    for eps in sorted(by_eps):
        runs = by_eps[eps]
        r0 = runs[0][0]
        a, dt, K = r0["a"], r0["dt"], r0["K"]
        n_train = r0["n_train"]
        _, n_val, n_test = parse_samples(r0["prefix"])
        total = n_train + n_val + n_test
        nx = ny = args.grid_resolution

        mesh = structured_quad_mesh(nx, ny)
        prob = ACProblem(mesh).to(device)
        M = prob.M
        ics = test_initial_conditions(total, n_train, n_val, n_test, K, args.seed, mesh, args.ic_r,
                                      args.n_samples).to(device)                 # [B, N]
        cc = prob.fem_reference(ics, a=a, eps=eps, dt=dt, n_steps=args.steps,
                                integrator="convex_concave")                     # [B, steps+1, N]
        # reference energy per frame (mean over samples)
        ref_E = [prob.energy(cc[:, k], a=a, eps=eps, reduce="none").mean().item()
                 for k in range(args.steps + 1)]
        cc_grid0 = node_to_grid(cc[:, 0], nx, ny)
        # per-sample squared reference norm per frame (for relative L2)
        den = [(cc[:, k] * prob._spmm(M, cc[:, k])).sum(-1).clamp_min(1e-30) for k in range(args.steps + 1)]

        print(f"eps={eps:g}: {len(runs)} runs, {ics.shape[0]} samples, {args.steps} steps")
        eps_out = {"ref_energy": ref_E, "runs": []}
        for rec, ck_path in runs:
            ck = torch.load(ck_path, map_location=device, weights_only=False)   # our own trusted files
            model = FNOModel(**(ck.get("model_config") or DEFAULT_CFG)).to(device).float()  # trained fp32
            model.load_state_dict(ck["model_state_dict"], strict=False)
            model.eval()
            preds = rollout(model, cc_grid0, prob, nx, ny, args.steps, device)    # [B, steps, H, W] fp32
            pn = grid_to_node(preds, nx, ny).double()                            # [B, steps, N] -> fp64 math

            energy = [ref_E[0]]                                                   # frame 0 = shared IC
            error = [0.0]
            for k in range(args.steps):
                uk = pn[:, k]
                energy.append(prob.energy(uk, a=a, eps=eps, reduce="none").mean().item())
                e = uk - cc[:, k + 1]
                num = (e * prob._spmm(M, e)).sum(-1).clamp_min(0)
                error.append((num / den[k + 1]).sqrt().mean().item())
            E_list, E_bad = _jsonable(energy)
            err_list, err_bad = _jsonable(error)
            diverge = max((b for b in (E_bad, err_bad) if b >= 0), default=-1)
            eps_out["runs"].append({
                "loss": loss_key(rec), "mode": rec.get("bptt_mode"),
                "integrator": rec.get("ac_integrator"), "prefix": rec["prefix"],
                "energy": E_list, "error": err_list,
                "diverged": diverge >= 0, "diverge_step": diverge,
            })
        out["by_eps"][f"{eps:g}"] = eps_out

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"\nmetrics -> {args.out}  ({len(out['by_eps'])} eps, "
          f"{sum(len(v['runs']) for v in out['by_eps'].values())} runs)")


if __name__ == "__main__":
    main()
