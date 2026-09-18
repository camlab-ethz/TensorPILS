"""kappa of the Stokes least-squares Hessian on the unstructured operator.

The paper asserts kappa(K P K) = O(h^-2) against O(h^-4) for the bare residual, and the Poisson
experiment backs the corresponding claim by reporting kappa(H_t) beside the error. This does the
same for the obstacle mesh, so the Schur-weight sweep of tab:stokes_ablation carries a measured
conditioning row rather than a theoretical assertion.

Measured on the **admissible subspace**: Dirichlet velocity DOFs removed and the constant-pressure
mode deflated by projection, since B^T 1 = 0 makes that mode a gauge and not ill-conditioning --
left in, it reports an infinite condition number for every omega alike. The velocity block uses
the exact A^-1 that the algebraic V-cycle approximates, which is what the theory is about.

Result (2026-09-18), chara_length 0.035, 8463 free DOF:

    bare least squares   kappa(K^2)   = 1.204e+11
    block P, omega_S=0.5   3.663e+06     32866x
    block P, omega_S=4     1.025e+06    117431x   <- minimum
    block P, omega_S=16    1.870e+06     64356x   <- the value validation selects
    block P, omega_S=32    3.156e+06     38146x
    block P, omega_S=64    5.759e+06     20904x
    block P, omega_S=256   2.143e+07      5617x

The minimum is at omega_S = 4, not at the 16 that validation selects, though the two differ by
1.8x in kappa and by less than a seed in error and both sit inside the empirical plateau. NOTE
this corrects a claim carried over from the structured Q2/Q1 operator, whose kappa optimum is
near 16; that number does not transfer to P2/P1 on this geometry.

Run:  clrun -p cpu -t 60 -- python experiments/stokes/unstructured/conditioning.py
"""
import argparse

import numpy as np

from tensorpils.meshing import obstacle_mesh
from tensorpils.physics import UnstructuredStokesProblem


def admissible(mesh, mu=1.0):
    prob = UnstructuredStokesProblem(mesh, mu=mu)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh_h", type=float, default=0.035)
    ap.add_argument("--mesh_cache",
                    default="/cluster/scratch/shiwen/TensorPILS_cache/meshes/obst_h0.035.msh")
    ap.add_argument("--omegas", type=float, nargs="+", default=[0.5, 4, 16, 32, 64, 256])
    args = ap.parse_args()

    K, is_p, D, mp = admissible(obstacle_mesh(chara_length=args.mesh_h,
                                              cache_path=args.mesh_cache))
    print(f"free DOF on the admissible subspace: {K.shape[0]}")
    k_bare = kappa(K @ K)
    print(f"bare least squares   kappa(K^2)   = {k_bare:.4g}")
    nu = int((~is_p).sum())
    A_inv = np.linalg.inv(K[:nu, :nu])
    for om in args.omegas:
        P = np.zeros_like(K)
        P[:nu, :nu] = A_inv
        P[nu:, nu:] = np.diag(om / mp)
        k = kappa(K @ (D @ P @ D) @ K)
        print(f"  block P, omega_S={om:<6g} kappa(KPK) = {k:10.4g}   gain {k_bare / k:8.0f}x")


if __name__ == "__main__":
    main()
