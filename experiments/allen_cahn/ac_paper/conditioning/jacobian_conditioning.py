r"""Condition numbers of the Allen–Cahn convex–concave Newton Jacobian, bare and preconditioned.

FEM only -- no network. Regime of experiments/allen_cahn/ac_paper: 128^2 nodes, a=1, eps=32,
dt=0.01, K=4, r=0.5, seed 42. For the FEM reference trajectory u^0..u^T of one initial condition
(the same one the dataset builds), on the interior DOFs:

    J(u) = M/dt + a^2 A + 3 eps^2 M diag(u^2)      Newton Jacobian of the convex-concave residual
    J0   = (1/dt + 3 eps^2) M + a^2 A              J at u^2 = 1, the operator the pls arm inverts

We report the 2-norm condition numbers sigma_max / sigma_min of

    J(u^n)            bare
    P_mg J(u^n)       P_mg = one V-cycle on J0, built exactly as ACTrainer builds it for pls
    J0^{-1} J(u^n)    the exact version of the same preconditioner, for reference

J is nonsymmetric (M diag(u^2) scales columns), hence singular values rather than eigenvalues;
they are also what the least-squares loss sees, its Gauss-Newton Hessian being (PJ)^T (PJ).
Extreme singular values come from ARPACK on the normal operators. sigma_min is always computed
by inverse iteration (largest eigenvalue of B^{-1}B^{-T}): J and J0 are inverted by sparse LU, the
V-cycle by CG -- P_mg is SPD and as well conditioned as J0, so CG converges in a few dozen steps.
(Plain Lanczos for the smallest eigenvalue stalls on the clustered spectrum of P_mg J.)

    python experiments/allen_cahn/ac_paper/conditioning/jacobian_conditioning.py
"""

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from tensormesh.dataset import WaveMultiFrequency
from tensorpils.meshing import structured_quad_mesh
from tensorpils.physics import ACProblem
from tensorpils.preconditioners import build_preconditioner


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # physical regime (defaults = ac_paper/lr_sweep.sbatch)
    p.add_argument("--grid", type=int, default=128, help="nodes per side (h = 1/(grid-1))")
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=32.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("-k", "--k", type=int, default=4)
    p.add_argument("--r", type=float, default=0.5, help="IC spectral decay (--ac_r)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_pool", type=int, default=1024 + 128 + 256,
                   help="size of the dataset's IC draw (n_train+n_val+n_test); sample i here is "
                        "sample i of that draw, so sample 0 is the first training sample")
    p.add_argument("--samples", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, nargs="+", default=[0, 5, 10],
                   help="trajectory frames n at which J(u^n) is evaluated")
    # multigrid (defaults = the CLI's --mg_*)
    p.add_argument("--mg_levels", type=int, default=4)
    p.add_argument("--mg_pre_smooth", type=int, default=2)
    p.add_argument("--mg_post_smooth", type=int, default=2)
    p.add_argument("--mg_omega", type=float, default=2.0 / 3.0)
    p.add_argument("--tol", type=float, default=1e-10, help="ARPACK tolerance")
    p.add_argument("--out", type=str, default=None, help="optional JSON dump of all numbers")
    return p.parse_args()


# ------------------------------------------------------------------ setup
def initial_conditions(mesh, args):
    """The dataset's ICs: same seed, same draw, same float32 cast (tensorpils.data.ACDataset)."""
    torch.manual_seed(args.seed)
    l_a = torch.rand(args.n_pool, args.k, args.k) * 2 - 1
    l_a = l_a[args.samples]
    return WaveMultiFrequency(a=l_a, r=args.r).initial_condition(mesh.points).float()


def interior_blocks(prob, idx):
    M = prob.M.to_scipy_coo().tocsr()[idx][:, idx].astype(np.float64)
    A = prob.A.to_scipy_coo().tocsr()[idx][:, idx].astype(np.float64)
    return M.tocsc(), A.tocsc()


def jacobian(M, A, u, a, eps, dt):
    """J(u) = M/dt + a^2 A + M diag(3 eps^2 u^2), exactly as ACProblem.fem_reference assembles it."""
    return (M / dt + (a * a) * A + M @ sp.diags(3.0 * eps * eps * u * u)).tocsc()


def interior_vcycle(mg, idx, n_nodes):
    """P_mg restricted to the interior: embed with a zero boundary, one V-cycle, take the interior."""
    idx_t = torch.from_numpy(idx)

    def apply(X):
        X2 = X.reshape(len(idx), -1)
        full = torch.zeros(X2.shape[1], n_nodes, dtype=torch.float64)
        full[:, idx_t] = torch.from_numpy(np.ascontiguousarray(X2.T))
        with torch.no_grad():
            Y = mg(full)
        return Y[:, idx_t].numpy().T.reshape(X.shape)
    return apply


def cg_inverse(P, n, rtol=1e-13):
    """x -> P^{-1} x for the SPD V-cycle P, by CG to rtol (tight: ARPACK needs an exact-ish inverse)."""
    op = spla.LinearOperator((n, n), matvec=P, dtype=np.float64)

    def solve(x):
        y, info = spla.cg(op, x, rtol=rtol, atol=0.0, maxiter=10 * n)
        if info != 0:
            raise RuntimeError(f"CG on the V-cycle did not converge (info={info})")
        return y
    return solve


# ------------------------------------------------------------------ singular values
def extreme_singular_values(n, mv, rmv, inv, rinv, tol=1e-10):
    """(sigma_min, sigma_max) of an n x n operator B given B x, B^T x, B^{-1} x and B^{-T} x."""
    lin = lambda f: spla.LinearOperator((n, n), matvec=f, dtype=np.float64)
    eig = lambda op, which: spla.eigsh(op, k=1, which=which, tol=tol, ncv=40, maxiter=20 * n,
                                       return_eigenvectors=False)[0]
    s_max = np.sqrt(eig(lin(lambda x: rmv(mv(x))), "LA"))
    s_min = 1.0 / np.sqrt(eig(lin(lambda x: inv(rinv(x))), "LA"))    # eig of B^{-1}B^{-T} = 1/sigma^2
    return float(s_min), float(s_max)


def conditioning(J, J0, J0_lu, P, P_inv, tol):
    """kappa_2 of J, P_mg J and J0^{-1} J."""
    n = J.shape[0]
    Jt = J.T.tocsc()
    lu = spla.splu(J)
    res = {}
    res["J"] = extreme_singular_values(
        n, lambda x: J @ x, lambda x: Jt @ x,
        lambda x: lu.solve(x), lambda x: lu.solve(x, trans="T"), tol)
    # P_mg is symmetric (symmetric Jacobi pre/post sweeps, R = P^T, exact coarse solve): checked in main
    res["P_mg J"] = extreme_singular_values(
        n, lambda x: P(J @ x), lambda x: Jt @ P(x),
        lambda x: lu.solve(P_inv(x)), lambda x: P_inv(lu.solve(x, trans="T")), tol)
    res["J0^-1 J"] = extreme_singular_values(       # J0 symmetric, so J0^{-T} = J0^{-1}
        n, lambda x: J0_lu.solve(J @ x), lambda x: Jt @ J0_lu.solve(x),
        lambda x: lu.solve(J0 @ x), lambda x: J0 @ lu.solve(x, trans="T"), tol)
    return {k: {"sigma_min": s0, "sigma_max": s1, "kappa": s1 / s0} for k, (s0, s1) in res.items()}


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    G, a, eps, dt = args.grid, args.a, args.eps, args.dt
    c = 1.0 / dt + 3.0 * eps * eps
    mesh = structured_quad_mesh(nx=G, ny=G)
    prob = ACProblem(mesh)
    idx = np.flatnonzero(~mesh.boundary_mask.bool().numpy())
    M, A = interior_blocks(prob, idx)
    print(f"grid {G}^2 (h=1/{G - 1}, {len(idx)} interior DOFs)  a={a:g} eps={eps:g} dt={dt:g}  "
          f"K={args.k} r={args.r:g} seed={args.seed}  c=1/dt+3eps^2={c:g}")

    # J0 and its two inverses: exact (sparse LU) and the pls arm's V-cycle
    J0 = ((a * a) * A + c * M).tocsc()
    J0_lu = spla.splu(J0)
    mg = build_preconditioner("multigrid", prob, (G, G), mg_levels=args.mg_levels,
                              mg_pre_smooth=args.mg_pre_smooth, mg_post_smooth=args.mg_post_smooth,
                              mg_omega=args.mg_omega, mg_a2=a * a, mg_c=c).double()
    mg.report()
    P = interior_vcycle(mg, idx, G * G)
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((2, len(idx)))
    asym = abs(x @ P(y) - y @ P(x)) / (np.linalg.norm(x) * np.linalg.norm(P(y)))
    print(f"V-cycle symmetry check |x'Py - y'Px| / (|x||Py|) = {asym:.1e}", flush=True)
    P_inv = cg_inverse(P, len(idx))

    # reference trajectories (float64 solve, same Newton as the dataset)
    u0 = initial_conditions(mesh, args).double()
    t0 = time.time()
    traj = prob.fem_reference(u0, a=a, eps=eps, dt=dt, n_steps=max(args.steps)).numpy()
    print(f"reference trajectories: {time.time() - t0:.1f}s", flush=True)

    def row(sample, step, u, J):
        t0 = time.time()
        r = {"sample": sample, "step": step, "frac_|u|>0.9": float(np.mean(np.abs(u) > 0.9)),
             "max_|u|": float(np.abs(u).max()), **conditioning(J, J0, J0_lu, P, P_inv, args.tol)}
        print(f"  sample {sample} step {step}: {time.time() - t0:.1f}s", flush=True)
        return r

    rows = [row(None, "u^2=1 (J0)", np.ones(len(idx)), J0)]
    for s, sample in enumerate(args.samples):
        for n in args.steps:
            u = traj[s, n, idx]
            rows.append(row(sample, n, u, jacobian(M, A, u, a, eps, dt)))

    hdr = f"{'sample':>6} {'step':>10} {'|u|>0.9':>8} {'max|u|':>7} | " \
          f"{'kappa(J)':>10} {'kappa(P_mg J)':>13} {'kappa(J0^-1 J)':>14}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{str(r['sample'] if r['sample'] is not None else '-'):>6} {str(r['step']):>10} "
              f"{r['frac_|u|>0.9']:>8.3f} {r['max_|u|']:>7.3f} | {r['J']['kappa']:>10.4g} "
              f"{r['P_mg J']['kappa']:>13.4g} {r['J0^-1 J']['kappa']:>14.4g}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "rows": rows}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
