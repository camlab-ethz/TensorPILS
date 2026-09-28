"""Per-sample test errors and long rollouts for the Allen-Cahn benchmark finals.

For every final run ``<root>/seed<S>/results/<prefix>.json`` with its best-validation checkpoint
``<root>/seed<S>/checkpoints/<prefix>_best.pth`` this

1. rebuilds the exact architecture (FNO, FNO wrapped in the PINO boundary wrapper, or DeepONet)
   and loads the weights with ``strict=True`` -- a mismatched rebuild fails loudly instead of
   silently evaluating an untrained network;
2. regenerates that seed's test initial conditions exactly as ``ACDataset`` does (same seed, same
   draw, same float32 cast, same Newton chunking) and solves the convex-concave reference;
3. rolls the model out freely from the exact initial condition, exactly as ``RolloutTrainer``
   evaluates (boundary projected after every step, prediction fed back);
4. records, per test sample s and time step k, ``||e_s^k||_M^2`` and ``||u_s^k||_M^2``.

From these it stores, per run:

* ``st_rel_0_10`` -- per-sample space-time relative L2 over steps k = 0..R (R = 10), i.e.
  ``sqrt(sum_k ||e^k||^2 / sum_k ||u^k||^2)``. Both sums include k = 0, where the error is zero
  (the model receives the exact initial condition), so it is one consistent quadrature.
  The summary table reports its mean over the test set.
* ``st_rel_1_10`` -- the same over k = 1..R, the variant the trainer uses.
* ``step_rel_mean`` -- mean over test samples of the per-sample relative L2 at each step k,
  over the training window, or to ``--long_steps`` for the long-rollout runs.
* ``pooled_st_rel_1_10`` -- the trainer's own pooled metric, recomputed as a consistency check
  against ``test_st_rel_l2`` in the results JSON (agreement to ~1e-6; the trainer sums in fp32).

Long rollouts (to ``--long_steps``, default 100 = T 1 at dt 0.01) are computed for the **best seed
of each arm**, chosen on *validation* (``best_val_st_rel_l2``) so the test set plays no part in
the choice. Every other run is rolled out over the training window only.

Output: one compact JSON, ``<root>/test_eval.json`` by default; pull it back and plot locally with
``plot_paper.py``.

Usage (after ``run.sh``; a GPU helps, and the reference solve to T = 1 is the expensive part):
    python experiments/allen_cahn/benchmark/evaluate_test.py
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_lr_sweep import ARMS                                 # prefix -> arm, same rules as the plots

from tensormesh.dataset import WaveMultiFrequency
from tensorpils.baselines import DeepONetModel, MollifiedModel, ZeroBoundaryModel
from tensorpils.meshing import grid_to_node, node_to_grid, structured_quad_mesh
from tensorpils.models import FNOModel
from tensorpils.physics import ACProblem, apply_zero_boundary


def arm_of(prefix):
    for key, _, match in ARMS:
        if match(prefix + ".json"):
            return key
    return None


def discover(root):
    """[{seed, arm, rec, ckpt}] for every final run; runs without a checkpoint are reported and skipped."""
    runs = []
    for path in sorted(glob.glob(os.path.join(root, "seed*", "results", "*.json"))):
        seed = int(re.search(r"seed(\d+)", path).group(1))
        with open(path) as fh:
            rec = json.load(fh)
        ckpt = os.path.join(os.path.dirname(os.path.dirname(path)), "checkpoints",
                            rec["prefix"] + "_best.pth")
        if not os.path.exists(ckpt):
            print(f"  skip (no checkpoint): seed {seed} {rec['prefix']}")
            continue
        runs.append({"seed": seed, "arm": arm_of(rec["prefix"]), "rec": rec, "ckpt": ckpt})
    return runs


def test_initial_conditions(seed, total, n_train, n_val, n_test, K, r, points, chunk):
    """The test split's ICs exactly as ACDataset builds them: one draw of ``total`` coefficient sets
    under ``torch.manual_seed(seed)``, evaluated in chunks and cast to float32, then the nested
    index range [n_train + n_val, total)."""
    torch.manual_seed(seed)
    l_a = torch.rand(total, K, K) * 2 - 1
    u0 = torch.cat([WaveMultiFrequency(a=l_a[s:s + chunk], r=r).initial_condition(points).float()
                    for s in range(0, total, chunk)], dim=0)
    return u0[n_train + n_val: n_train + n_val + n_test]


def build_model(rec, ckpt, nx, ny, device):
    """The trained architecture, rebuilt from the checkpoint's config; ``strict=True`` so a wrong
    rebuild raises rather than evaluating random weights."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)         # our own trusted files
    cfg, sd = ck["model_config"], ck["model_state_dict"]
    sd.pop("_metadata", None)                                           # neuraloperator bookkeeping, as in load_checkpoint
    if rec["prefix"].startswith("deeponet"):
        model = DeepONetModel(**cfg)
    else:
        model = FNOModel(**cfg)
        if any(k.startswith("model.") for k in sd):                    # PINO's hard-BC wrapper
            model = (MollifiedModel if "mollifier" in sd else ZeroBoundaryModel)(model, nx, ny)
    if "grid_size" in cfg:                                              # DeepONet records its grid
        assert tuple(cfg["grid_size"]) == (nx, ny), f"--grid_resolution {nx} != checkpoint {cfg['grid_size']}"
    model.load_state_dict(sd, strict=True)
    return model.to(device).float().eval(), ck.get("epoch")


@torch.no_grad()
def rollout_errors(model, u0, ref, steps, prob, M64, nx, ny, batch):
    """Free rollout from the exact IC, as RolloutTrainer._rollout evaluates. Returns per-sample
    per-step ``num[s, k] = ||e||_M^2`` and ``den[s, k] = ||u_ref||_M^2`` for k = 0..steps (fp64)."""
    mask = prob.boundary_mask.to(u0.device)

    def proj(g):
        return node_to_grid(apply_zero_boundary(grid_to_node(g, nx, ny), mask), nx, ny)

    n = u0.shape[0]
    num = torch.zeros(n, steps + 1, dtype=torch.float64)
    den = torch.zeros(n, steps + 1, dtype=torch.float64)
    for lo in range(0, n, batch):
        hi = min(lo + batch, n)
        frame = proj(node_to_grid(u0[lo:hi], nx, ny))                     # exact IC, projected
        for k in range(steps + 1):
            if k > 0:
                frame = proj(model(frame.unsqueeze(1)).squeeze(1))        # [b, H, W]
            u = ref[lo:hi, k].double()
            e = grid_to_node(frame, nx, ny).double() - u
            num[lo:hi, k] = (e * prob._spmm(M64, e)).sum(-1).clamp_min(0).cpu()
            den[lo:hi, k] = (u * prob._spmm(M64, u)).sum(-1).clamp_min(0).cpu()
    return num, den


def listf(x):
    """JSON-safe float list (NaN/inf kept as NaN, which Python's json reads back)."""
    return [float(v) for v in np.asarray(x, dtype=float)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/allen_cahn/benchmark/final")
    ap.add_argument("--out", default=None, help="default: <root>/test_eval.json")
    ap.add_argument("--arms", nargs="+", default=None, choices=[k for k, _, _ in ARMS],
                    help="evaluate only these arms (default: all). The reference solve dominates "
                         "the cost, so re-evaluating one arm after a re-run is much cheaper.")
    ap.add_argument("--merge", action="store_true",
                    help="merge into an existing --out file: entries of the evaluated arms are "
                         "replaced, all others are carried over unchanged.")
    ap.add_argument("--grid_resolution", type=int, default=129,
                    help="nodes per side of the training grid (not recorded in the results JSON)")
    ap.add_argument("--long_steps", type=int, default=100,
                    help="rollout horizon for each arm's best seed (100 = T 1 at dt 0.01)")
    ap.add_argument("--ic_r", type=float, default=0.5, help="IC spectral decay (--ac_r)")
    ap.add_argument("--newton_tol", type=float, default=1e-8, help="as --ac_newton_tol")
    ap.add_argument("--newton_max", type=int, default=20, help="as --ac_newton_max")
    ap.add_argument("--ref_chunk", type=int, default=64, help="as --ac_ref_chunk")
    ap.add_argument("--batch", type=int, default=32, help="rollout batch size")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.root, "test_eval.json")

    runs = discover(args.root)
    if args.arms:
        runs = [r for r in runs if r["arm"] in args.arms]
        print(f"evaluating arms {sorted(set(r['arm'] for r in runs))} only "
              f"({len(runs)} runs); best seed is still chosen within each arm")
    if not runs:
        raise SystemExit(f"no results with checkpoints under {args.root}")
    r0 = runs[0]["rec"]
    n_train, n_val, n_test = (int(x) for x in re.search(r"samples-(\d+)-(\d+)-(\d+)", r0["prefix"]).groups())
    a, eps, dt, K, R = r0["a"], r0["eps"], r0["dt"], r0["K"], r0["rollout_steps"]
    total = n_train + n_val + n_test
    for run in runs:                                                     # one regime per directory
        rr = run["rec"]
        assert (rr["a"], rr["eps"], rr["dt"], rr["K"], rr["rollout_steps"]) == (a, eps, dt, K, R), rr["prefix"]
    if (n_train + n_val) % args.ref_chunk:
        print(f"WARNING: test split does not start on a Newton-chunk boundary ({n_train + n_val} % "
              f"{args.ref_chunk}); references may differ from the dataset's in the last digits")

    # Best seed per arm, on validation.
    best_seed = {}
    for run in runs:
        v = run["rec"]["best_val_st_rel_l2"]
        if math.isfinite(v) and (run["arm"] not in best_seed or v < best_seed[run["arm"]][1]):
            best_seed[run["arm"]] = (run["seed"], v)
    print("best seed per arm (validation):", {k: s for k, (s, _) in best_seed.items()})

    nx = ny = args.grid_resolution
    mesh = structured_quad_mesh(nx=nx, ny=ny)
    prob = ACProblem(mesh).to(args.device)
    M64 = prob.M.to(dtype=torch.float64, device=args.device)
    out = {"config": {"a": a, "eps": eps, "dt": dt, "K": K, "rollout_steps": R, "n_test": n_test,
                      "grid_resolution": nx,
                      "long_steps": args.long_steps, "ic_r": args.ic_r}, "runs": []}

    for seed in sorted({run["seed"] for run in runs}):
        seed_runs = [run for run in runs if run["seed"] == seed]
        for run in seed_runs:
            run["long"] = best_seed.get(run["arm"], (None,))[0] == seed
        steps = args.long_steps if any(run["long"] for run in seed_runs) else R

        t0 = time.time()
        u0 = test_initial_conditions(seed, total, n_train, n_val, n_test, K, args.ic_r,
                                     mesh.points, args.ref_chunk).to(args.device)
        ref = prob.fem_reference(u0, a=a, eps=eps, dt=dt, n_steps=steps,
                                 newton_tol=args.newton_tol, newton_max=args.newton_max,
                                 chunk=args.ref_chunk)                                # [n, steps+1, N]
        print(f"seed {seed}: reference {n_test} x {steps} steps in {time.time() - t0:.0f}s", flush=True)

        for run in seed_runs:
            t0 = time.time()
            model, ep = build_model(run["rec"], run["ckpt"], nx, ny, args.device)
            n_steps = steps if run["long"] else R
            num, den = rollout_errors(model, u0, ref, n_steps, prob, M64, nx, ny, args.batch)
            with np.errstate(all="ignore"):
                num, den = num.numpy(), den.numpy()
                st010 = np.sqrt(num[:, :R + 1].sum(1) / den[:, :R + 1].sum(1))
                st110 = np.sqrt(num[:, 1:R + 1].sum(1) / den[:, 1:R + 1].sum(1))
                pooled = float(np.sqrt(num[:, 1:R + 1].sum() / den[:, 1:R + 1].sum()))
                step_rel = np.sqrt(num / den)                               # [n, steps+1]
            fin = np.isfinite(step_rel)
            step_mean = np.where(fin.any(0), np.nansum(np.where(fin, step_rel, 0), 0)
                                 / np.maximum(fin.sum(0), 1), np.nan)
            ref_st = run["rec"].get("test_st_rel_l2")
            print(f"  {run['arm']:<11s} (ckpt epoch {ep}{', long' if run['long'] else ''}): "
                  f"mean per-sample st rel L2 [0..{R}] {np.nanmean(st010):.4f}  "
                  f"non-finite samples {int((~np.isfinite(st010)).sum())}  "
                  f"pooled check {pooled:.5f} vs results {ref_st:.5f}  ({time.time() - t0:.0f}s)",
                  flush=True)
            out["runs"].append({
                "arm": run["arm"], "seed": seed, "prefix": run["rec"]["prefix"],
                "best_seed_for_arm": run["long"], "ckpt_epoch": ep,
                "best_val_st_rel_l2": run["rec"]["best_val_st_rel_l2"],
                "test_st_rel_l2_results_json": ref_st, "pooled_st_rel_1_10": pooled,
                "n_samples": int(num.shape[0]), "steps": n_steps,
                "st_rel_0_10": listf(st010), "st_rel_1_10": listf(st110),
                "nonfinite_samples": [int(i) for i in np.flatnonzero(~np.isfinite(st010))],
                "step_rel_mean": listf(step_mean),
                "step_rel_n_nonfinite": [int(c) for c in (~fin).sum(0)],
            })
            del model
        del ref

    if args.merge and os.path.exists(out_path):
        with open(out_path) as fh:
            old = json.load(fh)
        done = {r["arm"] for r in out["runs"]}
        kept = [r for r in old.get("runs", []) if r["arm"] not in done]
        print(f"merging into {out_path}: {len(out['runs'])} new runs, {len(kept)} carried over "
              f"({sorted({r['arm'] for r in kept})})")
        out["runs"] = kept + out["runs"]
        out["config"] = {**old.get("config", {}), **out["config"]}
    out["runs"].sort(key=lambda r: ([k for k, _, _ in ARMS].index(r["arm"]), r["seed"]))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
