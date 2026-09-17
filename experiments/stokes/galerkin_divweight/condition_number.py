"""Does a single scalar on the continuity rows precondition the Stokes least-squares problem?

Marius Zeinhofer's question, 2026-09-17: the FEM bare-least-squares arm trains only once the
continuity rows are weighted, so is that weight a preconditioner? As a norm weight it is
W = diag(I, w I) -- block diagonal, velocity block the identity, pressure block one number --
and the objective is 1/2 ||W(Kc - b)||^2, whose Gauss-Newton matrix is K W^2 K.  So the
quantity is kappa(K W^2 K) = kappa(WK)^2.

Two things have to be separated, and only a resolution sweep separates them: whether the scalar
improves the *constant* or the *rate*.  Run at 5 resolutions, with w retuned at every one of
them (a fixed w would conflate the two).  For contrast, the ideal block preconditioner
P = diag(A^-1/mu, omega*mu/diag(M_p)) -- exact A^-1, i.e. what the V-cycle approximates -- is
scored in the same framework as 1/2 r^T P r, Gauss-Newton matrix K P K.

Conditioning is measured on the **admissible subspace**: Dirichlet velocity DOFs are removed,
and the constant-pressure mode is deflated by projection rather than counted.  B^T 1 = 0 makes
that mode a gauge, not ill-conditioning -- leaving it in reports an infinite condition number
for every w alike and answers nothing.

Result (2026-09-17), fine log-spaced w grid, dofs 123 -> 9027:

    n_p  dofs   kappa(GN) w=1    best kappa(GN)   at w    gain
      5   123        4.67e+07        7.354e+04      40     635x
      9   531        7.35e+08        3.527e+05      69    2084x
     17  2211        1.19e+10        2.483e+06      91    4792x
     25  5043        6.04e+10        1.063e+07     103    5677x
     33  9027        1.91e+11        3.229e+07     133    5913x

    local slope  dlog(kappa)/dlog(h):
        w = 1      : -3.98  -4.02  -4.01  -4.00      <- converged, the known O(h^-4)
        w = best   : -2.26  -2.82  -3.59  -3.86      <- drifting to -4.00
        ideal P    : -1.82  -2.00  -2.00             <- converged, a genuine O(h^-2)

So the scalar does NOT correct the ill-conditioning: the rate stays O(h^-4) and the gain
saturates (increments 3.3x, 2.3x, 1.18x, 1.04x) at ~5900x on the Gauss-Newton matrix, i.e. ~77x
on kappa(WK).  The velocity block's A^-1 is what buys the order reduction; the pressure scalar
buys a bounded constant.  Two further signs it is not h-independent: the optimal w drifts
40 -> 133 over h = 1/8 -> 1/64 (~h^-0.58), against the h-independent omega ~ 16 recorded for the
full block P; and the *conditioning* optimum (w ~ 133) is not the *training* optimum (w = 30).

A bounded constant is still worth having.  At the resolution the paper trains at (65^2 velocity,
the n_p=33 row) it is the difference between 43.97 % velocity error at w=1 and 1.84 % at w=30.

Run:  clrun -p cpu -t 120 -- python experiments/stokes/galerkin_divweight/condition_number.py
"""
import time

import numpy as np

from tensorpils.meshing import structured_quad9_mesh
from tensorpils.physics import StokesProblem

# w retuned per resolution, so each grid brackets that resolution's optimum.
W_GRID = {5:  np.unique(np.round(np.logspace(0.5, 2.5, 21))),
          9:  np.unique(np.round(np.logspace(1.0, 2.8, 16))),
          17: np.unique(np.round(np.logspace(1.4, 2.8, 11))),
          25: np.unique(np.round(np.logspace(1.7, 2.8, 8))),
          33: np.unique(np.round(np.logspace(1.9, 2.8, 5)))}
OMEGAS = [0.5, 4, 16, 64, 256]
IDEAL_P_UP_TO = 25          # dense A^-1; 33 costs more than it tells us


def admissible(n_p):
    """K restricted to free velocity DOFs, with the constant-pressure gauge deflated away."""
    prob = StokesProblem(structured_quad9_mesh(nx=n_p, ny=n_p), nx_p=n_p, ny_p=n_p, mu=1.0)
    K = prob.K.to_scipy_coo().tocsr().astype(np.float64)
    free = np.flatnonzero(~prob.dirichlet_mask.numpy())
    is_p = free >= prob.off_p
    z = np.zeros(len(free))
    z[is_p] = 1.0
    z /= np.linalg.norm(z)
    D = np.eye(len(free)) - np.outer(z, z)
    M_p = prob.M_p.to_scipy_coo().tocsr().astype(np.float64).diagonal()
    return D @ K[free][:, free].toarray() @ D, is_p, D, M_p[free[is_p] - prob.off_p]


def kappa(M):
    ev = np.abs(np.linalg.eigvalsh(M))
    ev = ev[ev > ev.max() * 1e-13]
    return ev.max() / ev.min()


def main():
    rows = []
    for n_p, ws in W_GRID.items():
        t0 = time.time()
        K, is_p, D, mp_diag = admissible(n_p)
        k_unweighted = kappa(K @ K)
        k_best, w_best = min((kappa(K @ np.diag(np.where(is_p, w * w, 1.0)) @ K), w) for w in ws)

        k_ideal, om_best = float("nan"), None
        if n_p <= IDEAL_P_UP_TO:
            nu = int((~is_p).sum())
            A_inv = np.linalg.inv(K[:nu, :nu])
            for om in OMEGAS:
                P = np.zeros_like(K)
                P[:nu, :nu] = A_inv
                P[nu:, nu:] = np.diag(om / mp_diag)
                k = kappa(K @ (D @ P @ D) @ K)
                if not (k >= k_ideal):      # nan-safe
                    k_ideal, om_best = k, om

        h = 1.0 / (2 * n_p - 2)
        rows.append((n_p, h, K.shape[0], k_unweighted, k_best, w_best, k_ideal))
        print(f"n_p={n_p:>3}  h={h:.5f}  dofs={K.shape[0]:>5}  "
              f"kappa(GN) w=1 {k_unweighted:10.4g}   best {k_best:10.4g} at w={w_best:>5.0f} "
              f"({k_unweighted / k_best:6.1f}x)   ideal P {k_ideal:10.4g} at om={om_best}"
              f"   [{time.time() - t0:.0f}s]", flush=True)

    print("\nlocal slope  dlog kappa(GN) / dlog h")
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        L = np.log(b[1] / a[1])
        line = f"  h {a[1]:.5f}->{b[1]:.5f} :  w=1 {np.log(b[3] / a[3]) / L:+.2f}   best-w {np.log(b[4] / a[4]) / L:+.2f}"
        if np.isfinite(a[6]) and np.isfinite(b[6]):
            line += f"   ideal-P {np.log(b[6] / a[6]) / L:+.2f}"
        print(line)
    print("\noptimal w vs resolution:", [(r[0], int(r[5])) for r in rows],
          "  <- an h-independent preconditioner would not move")


if __name__ == "__main__":
    main()
