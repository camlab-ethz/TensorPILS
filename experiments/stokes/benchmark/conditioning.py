"""The condition-number row of the Stokes preconditioner ablation table.

Condition numbers of the loss Hessians on the obstacle mesh: kappa(K^2) for the unpreconditioned
least-squares loss L_LS, and kappa(K P K) for L_PLS with the block preconditioner
P = diag(A^-1, omega_S / diag(M_p)) at every Schur weight of the ablation.

Measured on the **admissible subspace**: Dirichlet velocity DOFs removed and the constant-pressure
mode deflated by projection, since B^T 1 = 0 makes that mode a gauge and not ill-conditioning --
left in, it reports an infinite condition number for every omega alike. The velocity block uses
the exact A^-1 that the algebraic V-cycle approximates, which is what the theory is about.

At chara_length 0.035 (8463 free DOF) this gives

    bare least squares     kappa(K^2)   = 1.204e+11
    block P, omega_S=0.5   3.663e+06
    block P, omega_S=4     1.025e+06    <- minimum
    block P, omega_S=16    1.870e+06    <- the value validation selects
    block P, omega_S=32    3.156e+06
    block P, omega_S=64    5.759e+06
    block P, omega_S=256   2.143e+07

Dense eigenvalue problems: a few minutes and ~2 GB on a CPU at the default mesh. Several
``--mesh_hs`` values measure the exponent of the growth under refinement.

    python experiments/stokes/benchmark/conditioning.py
"""
import argparse
import time

import numpy as np

from tensorpils.meshing import obstacle_mesh
from tensorpils.physics import StokesProblem


def admissible(mesh, mu=1.0):
    prob = StokesProblem(mesh, mu=mu)
    K = prob.K.to_scipy_coo().tocsr().astype(np.float64)
    free = np.flatnonzero(~prob.dirichlet_mask.numpy())
    is_p = free >= prob.off_p
    z = np.zeros(len(free))
    z[is_p] = 1.0
    z /= np.linalg.norm(z)
    D = np.eye(len(free)) - np.outer(z, z)
    Mp = prob.M_p.to_scipy_coo().tocsr().astype(np.float64).diagonal()
    return D @ K[free][:, free].toarray() @ D, is_p, D, Mp[free[is_p] - prob.off_p]


def kappa(M):
    ev = np.abs(np.linalg.eigvalsh(M))
    ev = ev[ev > ev.max() * 1e-13]
    return ev.max() / ev.min()


def mean_edge_length(mesh):
    pts = mesh.points[:, :2].numpy()
    tri = mesh.cells["triangle6"].numpy()[:, :3]
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    return float(np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh_hs", type=float, nargs="+", default=[0.035],
                    help="Gmsh target edge lengths (chara_length) to measure")
    ap.add_argument("--mesh_cache_dir", default="output/meshes")
    ap.add_argument("--omegas", type=float, nargs="+", default=[0.5, 4, 16, 32, 64, 256])
    args = ap.parse_args()

    rows = []
    for h in args.mesh_hs:
        t0 = time.time()
        mesh = obstacle_mesh(chara_length=h, cache_dir=args.mesh_cache_dir)
        K, is_p, D, mp = admissible(mesh)
        nu = int((~is_p).sum())
        k_bare = kappa(K @ K)
        print(f"h={h:g} (mean edge {mean_edge_length(mesh):.4g}): {mesh.n_points} P2 nodes, "
              f"{K.shape[0]} free DOF", flush=True)
        print(f"    bare least squares     kappa(K^2) = {k_bare:.4g}", flush=True)
        A_inv = np.linalg.inv(K[:nu, :nu])
        best = (np.inf, None)
        for om in args.omegas:
            P = np.zeros_like(K)
            P[:nu, :nu] = A_inv
            P[nu:, nu:] = np.diag(om / mp)
            k = kappa(K @ (D @ P @ D) @ K)
            print(f"    block P, omega_S={om:<5g}  kappa(KPK) = {k:.4g}   "
                  f"({k_bare / k:.0f}x better)", flush=True)
            if k < best[0]:
                best = (k, om)
        rows.append((mean_edge_length(mesh), k_bare, best[0]))
        print(f"    minimum at omega_S={best[1]:g}   [{time.time() - t0:.0f}s]", flush=True)

    if len(rows) > 1:
        e = np.log([r[0] for r in rows])
        for lab, idx in (("bare  kappa(K^2) ", 1), ("prec. kappa(KPK)", 2)):
            print(f"fit against the mean edge length: {lab} ~ h^"
                  f"{np.polyfit(e, np.log([r[idx] for r in rows]), 1)[0]:+.2f}")


if __name__ == "__main__":
    main()
