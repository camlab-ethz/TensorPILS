r"""Scan the Schur relaxation ``omega`` and find where it minimizes the pls conditioning.

The note introduces ``omega`` inside the *symmetric Uzawa smoother*, where it must satisfy
``omega in (0,1]`` for the smoothing iteration to converge, and suggests ``0.5``. In the
**norm-weighted** pls loss ``½ rᵀPr`` there is no iteration: ``omega`` only rescales the pressure
block of the norm relative to the velocity block, so it is unconstrained above. It trades

    kappa(P) ~ omega        against        max|lam(PK)|/min|lam(PK)| ~ 1/omega,

and since ``kappa(KPK) <~ kappa(P) * ratio^2`` (Lemma 1), the product has a genuine interior
minimum. Finding it is free — no extra work per training step, one scalar.

Usage:
    python experiments/stokes/conditioning/omega_scan.py --grids 33 65
    python experiments/stokes/conditioning/omega_scan.py --grids 17 --omegas 1 4 16 64

Writes ``output/stokes/conditioning/omega_scan_gr<grid>.json`` and a markdown table on stdout.
"""

import argparse
import json
import os

import torch

from measure_conditioning import admissible_basis, dense_preconditioner, _cond_sym
from tensorpils.meshing import structured_quad9_mesh
from tensorpils.physics import StokesProblem
from tensorpils.preconditioners.stokes import StokesBlockPreconditioner

torch.set_grad_enabled(False)


def scan(grid: int, omegas, mg_levels: int, smooth: int, mu: float) -> dict:
    n_p = (grid + 1) // 2
    prob = StokesProblem(structured_quad9_mesh(nx=n_p, ny=n_p), nx_p=n_p, ny_p=n_p, mu=mu)
    Z = admissible_basis(prob)
    K = prob.K.to_dense().double()
    K_S = Z.T @ K @ Z
    K_S = 0.5 * (K_S + K_S.T)
    del K

    rec = {"grid": grid, "pgrid": n_p, "h": 1.0 / (n_p - 1), "mg_levels": mg_levels,
           "smooth": smooth, "mu": mu, "omegas": [], "cond_P": [], "ratio_PK": [],
           "cond_KPK": []}
    print(f"\n=== velocity grid {grid}^2   mg_levels={mg_levels}  pre/post={smooth}/{smooth} ===")
    print(f"{'omega':>8} | {'kappa(P)':>10} {'ratio(PK)':>10} {'kappa(KPK)':>11}")
    for om in omegas:
        pc = StokesBlockPreconditioner(prob, mg_levels=mg_levels, mg_pre_smooth=smooth,
                                       mg_post_smooth=smooth, schur_omega=om).double()
        P, _ = dense_preconditioner(pc, prob.n_dofs)
        P_S = Z.T @ P @ Z
        P_S = 0.5 * (P_S + P_S.T)
        del P
        cP, *_ = _cond_sym(P_S)
        L = torch.linalg.cholesky(P_S)
        ev = torch.linalg.eigvalsh(L.T @ K_S @ L).abs()
        ratio = ev.max().item() / max(ev.min().item(), 1e-300)
        cKPK, *_ = _cond_sym(K_S @ P_S @ K_S)
        print(f"{om:>8.2f} | {cP:10.3e} {ratio:10.2f} {cKPK:11.3e}")
        rec["omegas"].append(om)
        rec["cond_P"].append(cP)
        rec["ratio_PK"].append(ratio)
        rec["cond_KPK"].append(cKPK)

    i = min(range(len(omegas)), key=lambda j: rec["cond_KPK"][j])
    rec["best_omega"] = omegas[i]
    rec["best_cond_KPK"] = rec["cond_KPK"][i]
    rec["gain_vs_half"] = (rec["cond_KPK"][omegas.index(0.5)] / rec["cond_KPK"][i]
                           if 0.5 in omegas else None)
    g = rec["gain_vs_half"]
    print(f"  -> best omega = {rec['best_omega']:g}   kappa(KPK) = {rec['best_cond_KPK']:.3e}"
          + (f"   ({g:.1f}x better than omega=0.5)" if g else ""))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grids", type=int, nargs="+", default=[33, 65])
    ap.add_argument("--omegas", type=float, nargs="+",
                    default=[0.5, 1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--mg_levels", type=int, default=4)
    ap.add_argument("--smooth", type=int, default=2)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--out_dir", default="output/stokes/conditioning")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for grid in args.grids:
        rec = scan(grid, list(args.omegas), args.mg_levels, args.smooth, args.mu)
        path = f"{args.out_dir}/omega_scan_gr{grid}.json"
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=2)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
