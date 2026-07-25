r"""Is the monolithic V-cycle's conditioning really mesh-independent?

The point of the monolithic preconditioner is a change of *regime*, and that claim is only
meaningful across resolutions:

===============================================  ==========================  ==============
loss                                             Gauss--Newton matrix        claim
===============================================  ==========================  ==============
``½ rᵀPr``, block ``P``   (norm-equivalent)      ``K P K``                   ``O(h^-2)``
``½‖Pr‖²``, monolithic ``P`` (approx. inverse)   ``(PK)ᵀ(PK)``               ``O(1)``
===============================================  ==========================  ==============

So this script measures, per grid, (i) the V-cycle contraction rate as a stationary iteration,
(ii) ``kappa_2(PK)`` and (iii) ``kappa((PK)ᵀPK)`` — the actual conditioning the applied loss
sees — in float64, and fits the exponent in ``h``. Flat means the note's monolithic route
delivers what it promises.

Note two facts that make the measurement subtly different from the block case:

* ``P`` is **not symmetric** here (the V-cycle only nearly is), so the relevant number is the
  2-norm condition number of ``PK``, obtained as ``sqrt(kappa((PK)ᵀPK))`` — no symmetric
  eigenproblem for ``PK`` itself is available.
* The V-cycle output is projected onto the admissible subspace, so ``P`` already maps
  ``S -> S`` and the restriction ``ZᵀPZ`` loses nothing.

Usage:
    python experiments/stokes/monolithic/measure_monolithic.py --grids 17 33 65
    python experiments/stokes/monolithic/measure_monolithic.py --grids 33 --scan

``--scan`` sweeps the smoother strength at one grid instead (Uzawa sweeps x Chebyshev degree),
which is how the default ``(nu, deg) = (4, 8)`` was chosen.
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "conditioning"))
from measure_conditioning import admissible_basis, _cond_sym                # noqa: E402

from tensorpils.meshing import structured_quad9_mesh                        # noqa: E402
from tensorpils.physics import StokesProblem                                # noqa: E402
from tensorpils.preconditioners.stokes import StokesBlockPreconditioner     # noqa: E402
from tensorpils.preconditioners.stokes_monolithic import (                  # noqa: E402
    StokesMonolithicMultigrid)

torch.set_grad_enabled(False)


def dense_operator(op, n: int, batch: int = 256) -> torch.Tensor:
    """Materialize ``P`` ``[n, n]`` column-wise. ``op`` is applied to rows, so transpose at the end."""
    rows = []
    for lo in range(0, n, batch):
        hi = min(lo + batch, n)
        E = torch.zeros(hi - lo, n, dtype=torch.float64)
        E[torch.arange(hi - lo), torch.arange(lo, hi)] = 1.0
        rows.append(op(E))
    return torch.cat(rows, dim=0).T                     # column j = P e_j


def build(prob, levels, nu, cd, omega, ratio):
    return StokesMonolithicMultigrid(prob, n_levels=levels, n_pre=nu, n_post=nu,
                                     cheb_degree=cd, cheb_ratio=ratio,
                                     schur_omega=omega).double()


def measure_one(prob, K_S, Z, mg) -> dict:
    hist = mg.cycle_report(n_cycles=8)
    P = dense_operator(mg, prob.n_dofs)
    asym = (P - P.T).abs().max().item() / max(P.abs().max().item(), 1e-300)
    PK = (Z.T @ P @ Z) @ K_S
    del P
    gn, lo, hi = _cond_sym(PK.T @ PK)                   # GN of ½‖Pr‖² = (PK)ᵀ(PK)
    return {"contraction": hist[-1] ** (1.0 / len(hist)), "residual_hist": hist,
            "P_asymmetry": asym, "cond_2_PK": math.sqrt(gn), "cond_applied_GN": gn}


def setup(grid: int, mu: float):
    n_p = (grid + 1) // 2
    prob = StokesProblem(structured_quad9_mesh(nx=n_p, ny=n_p), nx_p=n_p, ny_p=n_p, mu=mu)
    Z = admissible_basis(prob)
    K = prob.K.to_dense().double()
    K_S = Z.T @ K @ Z
    return prob, Z, 0.5 * (K_S + K_S.T)


def slope(xs, ys):
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(xs)
    mx, my = sum(lx) / n, sum(ly) / n
    den = sum((a - mx) ** 2 for a in lx)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grids", type=int, nargs="+", default=[17, 33, 65])
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--uzawa", type=int, default=4, help="nu_1 = nu_2")
    ap.add_argument("--cheb_degree", type=int, default=8)
    ap.add_argument("--cheb_ratio", type=float, default=30.0)
    ap.add_argument("--schur_omega", type=float, default=1.0,
                    help="For the Uzawa *iteration* this must stay in (0,1]; 1.0 is the optimum.")
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--scan", action="store_true",
                    help="Sweep (uzawa sweeps x Chebyshev degree) at the first grid instead.")
    ap.add_argument("--out_dir", default="output/stokes/monolithic")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.scan:
        grid = args.grids[0]
        prob, Z, K_S = setup(grid, args.mu)
        print(f"=== smoother scan, velocity grid {grid}^2, omega={args.schur_omega:g} ===")
        print(f"{'nu':>3} {'deg':>4} | {'A-applies':>9} {'contraction':>11} "
              f"{'kappa2(PK)':>10} {'kappa((PK)TPK)':>14}")
        out = []
        for nu in (1, 2, 3, 4):
            for cd in (2, 4, 6, 8):
                mg = build(prob, args.levels, nu, cd, args.schur_omega, args.cheb_ratio)
                r = measure_one(prob, K_S, Z, mg)
                r.update(grid=grid, uzawa=nu, cheb_degree=cd, a_applies=4 * nu * cd)
                out.append(r)
                print(f"{nu:>3} {cd:>4} | {4*nu*cd:>9} {r['contraction']:11.4f} "
                       f"{r['cond_2_PK']:10.3f} {r['cond_applied_GN']:14.4e}")
        path = f"{args.out_dir}/smoother_scan_gr{grid}.json"
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"-> {path}")
        return

    rows = []
    for grid in sorted(args.grids):
        prob, Z, K_S = setup(grid, args.mu)
        mg = build(prob, args.levels, args.uzawa, args.cheb_degree, args.schur_omega,
                   args.cheb_ratio)
        rec = measure_one(prob, K_S, Z, mg)
        # the block preconditioner at the same grid, for the side-by-side
        blk = StokesBlockPreconditioner(prob, mg_levels=args.levels, schur_omega=0.5).double()
        Pb = dense_operator(blk, prob.n_dofs)
        Pb_S = Z.T @ Pb @ Z
        del Pb
        rec["cond_weighted_GN"], *_ = _cond_sym(K_S @ (0.5 * (Pb_S + Pb_S.T)) @ K_S)
        rec.update(grid=grid, pgrid=(grid + 1) // 2, h=2.0 / (grid - 1), m=Z.shape[1],
                   levels=args.levels, uzawa=args.uzawa, cheb_degree=args.cheb_degree,
                   cheb_ratio=args.cheb_ratio, schur_omega=args.schur_omega, mu=args.mu)
        rows.append(rec)
        path = f"{args.out_dir}/mono_gr{grid}.json"
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=2)
        print(f"[gr={grid}] contraction={rec['contraction']:.4f}  "
              f"asym(P)={rec['P_asymmetry']:.2e}  kappa2(PK)={rec['cond_2_PK']:.3f}  "
              f"kappa((PK)^T PK)={rec['cond_applied_GN']:.4e}   "
              f"[block weighted kappa(KPK)={rec['cond_weighted_GN']:.4e}]  -> {path}")

    if len(rows) > 1:
        inv_h = [1.0 / r["h"] for r in rows]
        print("\nfitted exponent s in kappa ~ h^-s:")
        for key in ("cond_2_PK", "cond_applied_GN", "cond_weighted_GN", "contraction"):
            print(f"  {key:18s} s = {slope(inv_h, [r[key] for r in rows]):+.3f}")
        print("\n" + " | ".join(f"{r['grid']}^2: {r['cond_applied_GN']:.3g}" for r in rows))


if __name__ == "__main__":
    main()
