r"""Measure ``kappa(K P_t K)`` along the Stokes blend family — the collapse plot's x-axis.

``P_t = (1-t)·alpha·I + t·P_block`` interpolates the two losses whose conditioning the theory
separates: ``t=0`` is the bare least-squares loss (``kappa = kappa(K^2) = O(h^-4)``) and ``t=1``
the norm-weighted preconditioned one (``O(h^-2)``). Training gives error-vs-``t``; this script
gives conditioning-vs-``t``, and the two together give error-vs-conditioning without any
theoretical step in between.

Reuses the float64 machinery of ``../conditioning/measure_conditioning.py`` (single source of
truth for the admissible-subspace reduction), so it must be run from the repository root.

Usage:
    python experiments/stokes/blend_sweep/measure_blend_cond.py --grid 65
    python experiments/stokes/blend_sweep/measure_blend_cond.py --grid 33 --strengths 0 0.9 1

Writes ``output/stokes/blend_sweep/cond_blend_gr<grid>.json``, which ``plot_blend_sweep.py`` joins
onto the training runs by ``t``.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "conditioning"))
from measure_conditioning import (admissible_basis, dense_preconditioner,  # noqa: E402
                                  _cond_sym)

from tensorpils.meshing import structured_quad9_mesh                        # noqa: E402
from tensorpils.physics import StokesProblem                                # noqa: E402
from tensorpils.preconditioners.stokes import (StokesBlockPreconditioner,   # noqa: E402
                                               StokesBlendPreconditioner)

torch.set_grad_enabled(False)

# t values spaced logarithmically in (1-t): the identity term dominates until 1-t ~ 1/kappa(P),
# so a uniform grid in t would put every interesting point in the last 5%.
DEFAULT_T = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.997, 0.999, 1.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grid", type=int, default=65, help="Velocity grid (odd).")
    ap.add_argument("--strengths", type=float, nargs="+", default=DEFAULT_T)
    ap.add_argument("--mg_levels", type=int, default=4)
    ap.add_argument("--schur_omega", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--out_dir", default="output/stokes/blend_sweep")
    args = ap.parse_args()

    n_p = (args.grid + 1) // 2
    prob = StokesProblem(structured_quad9_mesh(nx=n_p, ny=n_p), nx_p=n_p, ny_p=n_p, mu=args.mu)
    Z = admissible_basis(prob)
    K = prob.K.to_dense().double()
    K_S = Z.T @ K @ Z
    K_S = 0.5 * (K_S + K_S.T)
    del K
    base = StokesBlockPreconditioner(prob, mg_levels=args.mg_levels,
                                     schur_omega=args.schur_omega).double()

    rec = {"grid": args.grid, "pgrid": n_p, "h": 1.0 / (n_p - 1), "mu": args.mu,
           "mg_levels": args.mg_levels, "schur_omega": args.schur_omega,
           "m": Z.shape[1], "entries": []}
    print(f"=== blend conditioning, velocity grid {args.grid}^2, m={Z.shape[1]} ===")
    print(f"{'t':>7} {'alpha':>11} {'kappa(P_t)':>11} {'ratio(P_tK)':>12} {'kappa(KP_tK)':>13}")
    for t in args.strengths:
        pc = StokesBlendPreconditioner(base, strength=t)
        P, asym = dense_preconditioner(pc, prob.n_dofs)
        P_S = Z.T @ P @ Z
        P_S = 0.5 * (P_S + P_S.T)
        del P
        cP, *_ = _cond_sym(P_S)
        L = torch.linalg.cholesky(P_S)                # SPD for every t: convex combination
        ev = torch.linalg.eigvalsh(L.T @ K_S @ L).abs()
        cKPK, *_ = _cond_sym(K_S @ P_S @ K_S)
        e = {"t": t, "alpha": pc.alpha, "cond_P": cP, "cond_KPK": cKPK,
             "ratio_PK": ev.max().item() / max(ev.min().item(), 1e-300),
             "P_asymmetry": asym}
        rec["entries"].append(e)
        print(f"{t:>7.4g} {pc.alpha:11.4e} {cP:11.4e} {e['ratio_PK']:12.2f} {cKPK:13.4e}")

    os.makedirs(args.out_dir, exist_ok=True)
    path = f"{args.out_dir}/cond_blend_gr{args.grid}.json"
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
