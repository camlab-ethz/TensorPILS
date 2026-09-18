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
import os
import time

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


def mesh_for(h, cache_dir, cache_path=None, tol=0.25):
    """Load or build the obstacle mesh at ``chara_length = h`` -- and prove it is that mesh.

    ``obstacle_mesh`` treats ``cache_path`` as a plain filename: the ONLY test is whether the file
    exists, and chara_length / obstacle / cx / cy / r are never compared against what is stored.
    So a cache path from one resolution silently serves every other, and an h-sweep run naively
    against a single cached file fits an exponent to three copies of the same matrix.

    Two defences. The path is derived from ``h`` unless one is given explicitly, and the realised
    mean edge length of whatever came back is checked against the request.
    """
    path = cache_path or os.path.join(cache_dir, f"obst_h{h:g}.msh")
    mesh = obstacle_mesh(chara_length=h, cache_path=path)
    pts = mesh.points[:, :2].numpy()
    tri = mesh.cells["triangle6"].numpy()[:, :3]
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    realised = float(np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1).mean())
    if abs(realised - h) > tol * h:
        raise SystemExit(
            f"mesh at {path} has mean edge length {realised:.4g}, but chara_length={h:g} was "
            f"requested (>{tol:.0%} off). The cache is keyed by filename only, so this is almost "
            f"certainly a stale path from another resolution.")
    return mesh, realised


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh_hs", type=float, nargs="+",
                    default=[0.06, 0.05, 0.04, 0.035, 0.03, 0.025],
                    help="chara_length values to sweep. Deliberately wider than the cached "
                         "0.035-0.025: a factor of 1.4 in h cannot separate h^-2 from h^-4, and "
                         "the coarse end is nearly free since DOF ~ h^-2.")
    ap.add_argument("--cache_dir",
                    default="/cluster/scratch/shiwen/TensorPILS_cache/meshes")
    ap.add_argument("--mesh_cache", default=None,
                    help="explicit path, for a single h only; normally leave unset so the path "
                         "is derived from --mesh_hs and cannot go stale")
    ap.add_argument("--omegas", type=float, nargs="+", default=[0.5, 4, 16, 32, 64, 256])
    args = ap.parse_args()
    if args.mesh_cache and len(args.mesh_hs) > 1:
        raise SystemExit("--mesh_cache pins one file; it cannot serve several --mesh_hs")

    rows = []
    for h in args.mesh_hs:
        t0 = time.time()
        mesh, realised = mesh_for(h, args.cache_dir, args.mesh_cache)
        K, is_p, D, mp = admissible(mesh)
        nu = int((~is_p).sum())
        k_bare = kappa(K @ K)
        A_inv = np.linalg.inv(K[:nu, :nu])
        best = (np.inf, None)
        for om in args.omegas:
            P = np.zeros_like(K)
            P[:nu, :nu] = A_inv
            P[nu:, nu:] = np.diag(om / mp)
            k = kappa(K @ (D @ P @ D) @ K)
            if k < best[0]:
                best = (k, om)
        rows.append((h, mesh.n_points, K.shape[0], k_bare, best[0], best[1]))
        print(f"h={h:<6g} edge={realised:.4g}  {mesh.n_points:5d} P2 nodes  {K.shape[0]:5d} free DOF"
              f"   kappa(K^2)={k_bare:10.4g}   best kappa(KPK)={best[0]:10.4g} at omega_S={best[1]:<5g}"
              f"   gain {k_bare / best[0]:7.0f}x   [{time.time() - t0:.0f}s]", flush=True)

    if len(rows) < 2:
        return
    hs = np.array([r[0] for r in rows])
    print("\nlocal slope  dlog kappa / dlog h  (consecutive meshes)")
    for i in range(1, len(rows)):
        L = np.log(hs[i] / hs[i - 1])
        print(f"  h {hs[i-1]:.3g} -> {hs[i]:.3g} :  bare {np.log(rows[i][3]/rows[i-1][3])/L:+.2f}"
              f"   preconditioned {np.log(rows[i][4]/rows[i-1][4])/L:+.2f}")
    for lab, idx in (("bare  kappa(K^2) ", 3), ("prec. kappa(KPK)", 4)):
        e = np.polyfit(np.log(hs), np.log([r[idx] for r in rows]), 1)[0]
        print(f"global fit: {lab} ~ h^{e:+.2f}")
    print("optimal omega_S vs h:", [(r[0], r[5]) for r in rows])


if __name__ == "__main__":
    main()
