"""Graphical abstract, part 1: what ill-conditioning does to a first-order optimizer.

The experiment strips the setting down to its skeleton: **no neural operator at all**. We take
the Q1 stiffness matrix ``A`` of the Dirichlet Poisson problem on a structured ``N x N`` grid,
form the physics-informed least-squares loss

    L(u) = 1/2 || A u - b ||^2,        b = M f   (consistent FE load)

over a plain coefficient vector ``u``, and minimise it with Adam. Because the Hessian is
``A^T A = A^2``, the conditioning is ``kappa(A)^2 ~ h^-4`` -- at ``N=64`` that is ``6.5e5``. The
point of the figure is that this alone, with no network in the loop, already defeats Adam.

What is recorded (to a single ``.npz``):

* the Adam trajectory projected onto the plane of the extreme eigenvectors of ``A``
  (``v_max``, ``v_min``) -- the plane in which the loss contours are most stretched;
* the relative error against the float64 reference solution, per recorded step;
* the full modal spectrum of the error, ``|<e_k, v_pq>|``, for every mode -- which shows
  *which* frequencies Adam fixes and which it cannot touch;
* the same relative-error history for plain gradient descent at its optimal step size.

Two implementation notes that make this cheap and exact:

1. On a uniform grid the Q1 stiffness is ``A = K (x) Mt + Mt (x) K`` (1D stiffness/mass), and
   the 1D sine vectors diagonalise both, so the **2D discrete sine modes are exact eigenvectors
   of A** with closed-form eigenvalues. No eigensolver is needed and all modal projections come
   from one DST. The same fact is what ``preconditioners/spectral.py`` exploits. It is verified
   numerically at startup rather than assumed.
2. Adam runs in **float32** (the realistic training regime) but every reported error is computed
   in **float64** against a float64 direct solve, so the metric is not itself polluted.

Usage (see run.sh for the settings of Figure 1):
    python experiments/graphical_abstract/run_landscape.py \
        --out output/graphical_abstract/landscape.npz
"""

import argparse
import os

import numpy as np
import scipy.sparse.linalg as spla
import torch
from scipy.fft import dstn

from tensorpils.meshing import structured_quad_mesh
from tensorpils.physics import PoissonProblem
from tensormesh.dataset import PoissonMultiFrequency


# --------------------------------------------------------------------------- assembly
def assemble(grid: int, K: int, seed: int):
    """Interior stiffness/mass blocks, consistent load, and the float64 reference solution."""
    mesh = structured_quad_mesh(nx=grid, ny=grid)
    prob = PoissonProblem(mesh)
    A = prob.A.to_scipy_coo().tocsr()
    M = prob.M.to_scipy_coo().tocsr()
    mask = prob.boundary_mask.numpy().astype(bool)
    idx = np.flatnonzero(~mask)

    A_ii = A[idx][:, idx].astype(np.float64)
    M_ii = M[idx][:, idx].astype(np.float64)

    # Same right-hand-side distribution as every other Poisson experiment in the paper.
    torch.manual_seed(seed)
    coeffs = torch.rand(1, K, K) * 2 - 1
    f = PoissonMultiFrequency(a=coeffs, r=-0.5).source_term(mesh.points).numpy()[0]
    b = (M @ f)[idx]                                    # consistent load b = M f, interior

    u_star = spla.spsolve(A_ii.tocsc(), b)              # float64 ground truth
    return A_ii, M_ii, b, u_star


def sine_eigenvalues(grid: int) -> np.ndarray:
    """Closed-form eigenvalues of the interior Q1 stiffness, indexed ``[p-1, q-1]``.

    1D stiffness ``K = (1/h) tridiag(-1,2,-1)`` and mass ``Mt = (h/6) tridiag(1,4,1)`` share the
    sine eigenvectors, with ``theta_p = p*pi*h``; the 2D operator ``K (x) Mt + Mt (x) K`` then has
    ``lambda_pq = k_p m_q + m_p k_q``.
    """
    n_i = grid - 2                                      # interior points per dimension
    h = 1.0 / (grid - 1)
    theta = np.pi * np.arange(1, n_i + 1) * h           # p*pi*h,  p = 1..n_i
    k = (1.0 / h) * (2.0 - 2.0 * np.cos(theta))         # 1D stiffness eigenvalues
    m = (h / 6.0) * (4.0 + 2.0 * np.cos(theta))         # 1D mass eigenvalues
    return np.outer(k, m) + np.outer(m, k)              # [n_i, n_i]


def mode_vector(p: int, q: int, n_i: int) -> np.ndarray:
    """Unit-norm interior sine mode ``sin(p pi x) sin(q pi y)``, flattened row-major."""
    i = np.arange(1, n_i + 1)
    s = np.sin(np.pi * p * i / (n_i + 1))
    t = np.sin(np.pi * q * i / (n_i + 1))
    v = np.outer(s, t).ravel()
    return v / np.linalg.norm(v)


def verify_eigenbasis(A_ii, lam, n_i, rtol=1e-10):
    """Assert the sine modes really are eigenvectors -- the assumption the whole file rests on."""
    worst = 0.0
    for (p, q) in [(1, 1), (1, n_i), (n_i, 1), (n_i, n_i), (7, 13), (n_i // 2, n_i // 3)]:
        v = mode_vector(p, q, n_i)
        r = A_ii @ v - lam[p - 1, q - 1] * v
        worst = max(worst, np.linalg.norm(r) / abs(lam[p - 1, q - 1]))
    if worst > rtol:
        raise RuntimeError(f"sine modes are not eigenvectors of A (worst rel. residual {worst:.2e}); "
                           "the closed-form spectrum and the DST projection are both invalid here")
    return worst


# --------------------------------------------------------------------------- preconditioner
class MultigridOperator:
    """Apply the geometric-multigrid V-cycle to an *interior* vector, as a black box.

    Nothing here needs ``P`` as a matrix. ``GeometricMultigrid`` acts on full-grid vectors with
    zero Dirichlet entries, so we embed the interior vector, apply one V-cycle, and read the
    interior back. With ``pre_smooth == post_smooth`` and a (symmetric) weighted-Jacobi smoother
    the V-cycle operator is symmetric -- verified in :meth:`check_symmetry` rather than assumed --
    so ``P^T = P`` and the preconditioned gradient is ``A P^2 (Au - b)``: two V-cycles and two
    sparse matvecs per step.
    """

    def __init__(self, grid: int, idx: np.ndarray, n_levels=4, pre=2, post=2,
                 omega: float = 2.0 / 3.0):
        import torch
        from tensorpils.preconditioners.multigrid import GeometricMultigrid
        self.torch = torch
        self.mg = GeometricMultigrid(nx_fine=grid, ny_fine=grid, n_levels=n_levels,
                                     pre_smooth=pre, post_smooth=post, omega=omega)
        self.idx = torch.from_numpy(idx.astype(np.int64))
        self.n_full = grid * grid

    def __call__(self, r_int: np.ndarray) -> np.ndarray:
        t = self.torch
        full = t.zeros(1, self.n_full, dtype=t.float32)
        full[0, self.idx] = t.from_numpy(np.ascontiguousarray(r_int, dtype=np.float32))
        return self.mg(full)[0, self.idx].numpy()

    def check_symmetry(self, n_int: int, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        x, y = rng.standard_normal(n_int), rng.standard_normal(n_int)
        a, b = float(self(x) @ y), float(x @ self(y))
        return abs(a - b) / max(abs(a), 1e-30)


# --------------------------------------------------------------------------- optimizers
def _grad(A32, u, b32, P=None):
    """Gradient of 1/2||P(Au-b)||^2.

    ``P=None`` gives the bare loss, whose gradient is ``A^T(Au-b) = A(Au-b)`` (A symmetric).
    Otherwise ``A P^T P (Au-b) = A P^2 (Au-b)``, using the V-cycle's symmetry.
    """
    r = A32 @ u - b32
    if P is not None:
        r = P(P(r))
    return A32 @ r


def run_adam(A_ii, b, u0, lr, steps, record=None, n_i=None, P=None, lr_min=None):
    """Adam in float32. Returns (records, final_u). ``record`` is a sorted array of step indices."""
    A32, b32 = A_ii.astype(np.float32), b.astype(np.float32)
    u = u0.astype(np.float32).copy()
    m = np.zeros_like(u)
    v = np.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8
    rec = {} if record is None else {int(s): None for s in record}
    for t in range(1, steps + 1):
        # Cosine annealing lr -> lr_min over the run, matching the schedule the trainer uses
        # elsewhere in the project (torch CosineAnnealingLR from --lr to --lr_min).
        lr_t = lr if lr_min is None else \
            lr_min + 0.5 * (lr - lr_min) * (1.0 + np.cos(np.pi * (t - 1) / steps))
        g = _grad(A32, u, b32, P)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        u -= np.float32(lr_t) * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        if t in rec:
            rec[t] = u.astype(np.float64).copy()
    return rec, u


def run_gd(A_ii, b, u0, eta, steps, record, P=None):
    """Plain gradient descent in float32, same recording contract as :func:`run_adam`."""
    A32, b32 = A_ii.astype(np.float32), b.astype(np.float32)
    u = u0.astype(np.float32).copy()
    rec = {int(s): None for s in record}
    for t in range(1, steps + 1):
        u -= np.float32(eta) * _grad(A32, u, b32, P)
        if t in rec:
            rec[t] = u.astype(np.float64).copy()
    return rec


# --------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=64, help="nodes per dimension (default 64)")
    ap.add_argument("--K", type=int, default=4, help="source-term frequency cutoff (default 4)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=100000, help="Adam/GD budget (default 1e5)")
    ap.add_argument("--sweep_steps", type=int, default=10000,
                    help="budget at which the learning rate is tuned (default 1e4)")
    ap.add_argument("--n_records", type=int, default=200,
                    help="how many iterates to store when --record log (default 200)")
    ap.add_argument("--stride", type=int, default=1,
                    help="with --record every, keep every Nth step. Uniform subsampling, so the "
                         "matrix x-axis stays evenly spaced; needed to reach 1e5 steps without a "
                         "1.5 GB array (1e5 x 3844 float32).")
    ap.add_argument("--record", choices=["log", "every"], default="log",
                    help="log: n_records log-spaced iterates (default). every: store all steps "
                         "1..steps, for the full-resolution matrix view. Costs "
                         "steps*n_modes*4 bytes per optimizer (4096x3844 ~ 63 MB each).")
    ap.add_argument("--init", choices=["zero", "random"], default="zero",
                    help="u0 = 0 (default; the error is then -u*, i.e. entirely smooth -- the "
                         "modes a first-order method cannot fix) or a random vector scaled to "
                         "100%% relative error (seeds every mode, so the modal spectrum shows "
                         "the high-pass-filter behaviour).")
    ap.add_argument("--lr", type=float, default=None,
                    help="fix Adam's learning rate instead of sweeping for it. Use this to compare "
                         "the same lr across preconditioned and bare runs -- the swept optimum "
                         "differs between them, so a swept comparison is not like-for-like.")
    ap.add_argument("--lr_min", type=float, default=None,
                    help="if set, cosine-anneal Adam's learning rate from --lr to this over the "
                         "run (constant lr otherwise). Constant-lr Adam ends in a limit cycle of "
                         "radius ~0.011*lr; decay is what practitioners actually use.")
    ap.add_argument("--skip_gd", action="store_true",
                    help="skip the gradient-descent arm (it has no lr to match, and costs the same "
                         "as Adam under preconditioning)")
    ap.add_argument("--mg_omega", type=float, default=2.0 / 3.0,
                    help="damping of the weighted-Jacobi smoother. The default 2/3 is the *1D "
                         "finite-difference* optimum; for Q1 in 2D the high-frequency spectrum of "
                         "D^-1 A is [3/4, 3/2], so the optimal value is 8/9 (smoothing factor 1/3 "
                         "instead of 1/2).")
    ap.add_argument("--precond", choices=["none", "multigrid"], default="none",
                    help="none (default) minimises 1/2||Au-b||^2; multigrid minimises "
                         "1/2||P(Au-b)||^2 with P one geometric-multigrid V-cycle, used purely "
                         "as a matvec.")
    ap.add_argument("--out", default="output/graphical_abstract/landscape.npz")
    args = ap.parse_args()

    A_ii, M_ii, b, u_star = assemble(args.grid, args.K, args.seed)
    n_i = args.grid - 2
    n = A_ii.shape[0]

    lam2d = sine_eigenvalues(args.grid)
    worst = verify_eigenbasis(A_ii, lam2d, n_i)
    lam = lam2d.ravel()
    lam_max, lam_min = lam.max(), lam.min()
    pq_max = np.unravel_index(np.argmax(lam2d), lam2d.shape)
    pq_min = np.unravel_index(np.argmin(lam2d), lam2d.shape)
    print(f"grid {args.grid}^2   interior dofs {n}   sine-eigenbasis check {worst:.2e}")
    print(f"lambda_max = {lam_max:.4f} (mode {tuple(int(x)+1 for x in pq_max)})   "
          f"lambda_min = {lam_min:.4e} (mode {tuple(int(x)+1 for x in pq_min)})")
    print(f"kappa(A) = {lam_max/lam_min:.1f}   kappa(A^2) = {(lam_max/lam_min)**2:.3e}   "
          f"contour aspect = sqrt(kappa(A^2)) = {lam_max/lam_min:.0f}:1")

    P = None
    if args.precond == "multigrid":
        mask = np.ones(args.grid * args.grid, bool)
        mesh_idx = np.flatnonzero(~structured_quad_mesh(nx=args.grid, ny=args.grid)
                                  .boundary_mask.numpy().astype(bool))
        P = MultigridOperator(args.grid, mesh_idx, omega=args.mg_omega)
        sym = P.check_symmetry(n)
        print(f"preconditioner: multigrid V-cycle  omega={args.mg_omega:.4f}  "
              f"symmetry residual {sym:.2e}")
        if sym > 1e-5:
            raise RuntimeError("V-cycle is not symmetric; grad = A P^2 r is then wrong")

    nrm_star = np.sqrt(u_star @ (M_ii @ u_star))

    def rel(u):
        e = u - u_star
        return np.sqrt(max(e @ (M_ii @ e), 0.0)) / nrm_star

    # Both inits start at exactly 100% relative error but are *not* equally hard, and the
    # difference is the mechanism itself: `zero` puts the whole initial error in the smooth modes
    # (e0 = -u*, and u* = A^-1 b decays like lambda^-1), which are precisely the ones a first-order
    # method cannot reduce; `random` spreads it over every mode, most of which are stiff and get
    # crushed in a few hundred steps. `zero` is the realistic and much harder case.
    if args.init == "zero":
        u0 = np.zeros(n)
    else:
        rng = np.random.default_rng(args.seed)
        e0 = rng.standard_normal(n)
        e0 *= nrm_star / np.sqrt(e0 @ (M_ii @ e0))
        u0 = u_star + e0
    print(f"init = {args.init};  initial relative error {rel(u0):.4f}")

    # ---- tune the learning rate in good faith, at a fixed budget (unless fixed by hand) ----
    grid_lr = np.geomspace(1e-4, 3e-1, 11)
    if args.lr is not None:
        best_lr, scores = float(args.lr), [float("nan")] * len(grid_lr)
        print(f"\nlearning rate fixed at {best_lr:.2e} (no sweep)")
    else:
        print(f"\ntuning lr over {len(grid_lr)} values at {args.sweep_steps} steps:")
        scores = []
        for lr in grid_lr:
            _, u_end = run_adam(A_ii, b, u0, lr, args.sweep_steps, P=P)
            r = rel(u_end.astype(np.float64))
            scores.append(r if np.isfinite(r) else np.inf)   # a diverged run must not win
            print(f"   lr = {lr:8.2e}   rel = {scores[-1]:.4e}")
        best_lr = float(grid_lr[int(np.argmin(scores))])
        print(f"best lr = {best_lr:.2e}  (rel = {min(scores):.4e} at {args.sweep_steps} steps)")
        if int(np.argmin(scores)) in (0, len(grid_lr) - 1):
            print("   WARNING: optimum sits at the edge of the sweep -- widen the range")

    # ---- production run, recording log-spaced iterates ----
    if args.record == "every":
        rec_steps = np.arange(args.stride, args.steps + 1, args.stride, dtype=np.int64)
    else:
        rec_steps = np.unique(np.geomspace(1, args.steps, args.n_records).astype(np.int64))
    sched = "" if args.lr_min is None else f" -> {args.lr_min:.2e} (cosine)"
    print(f"\nAdam: {args.steps} steps at lr = {best_lr:.2e}{sched} ...")
    rec_adam, _ = run_adam(A_ii, b, u0, best_lr, args.steps, record=rec_steps, P=P,
                           lr_min=args.lr_min)

    if args.skip_gd:
        eta = float("nan")
        rec_gd = {int(k): np.zeros_like(u_star) for k in rec_steps}
        print("GD:   skipped (--skip_gd)")
    elif P is None:
        eta = 2.0 / (lam_max ** 2 + lam_min ** 2)       # optimal step for the bare quadratic
    else:
        # Preconditioning changes the Hessian to A P^2 A, whose spectrum we do not have in closed
        # form, so tune eta the same way we tune lr rather than guess.
        grid_eta = np.geomspace(1e-2, 1.8, 9)
        scores_g = []
        for e in grid_eta:
            r = rel(run_gd(A_ii, b, u0, e, args.sweep_steps, [args.sweep_steps], P=P)[args.sweep_steps])
            scores_g.append(r if np.isfinite(r) else np.inf)
        eta = float(grid_eta[int(np.argmin(scores_g))])
        print(f"tuned GD eta = {eta:.2e}  (rel = {min(scores_g):.4e})")
        if int(np.argmin(scores_g)) == len(grid_eta) - 1:
            print("   WARNING: eta optimum at the top of the sweep -- may diverge at full budget")
    if not args.skip_gd:
        print(f"GD:   {args.steps} steps at eta = {eta:.3e} ...")
        rec_gd = run_gd(A_ii, b, u0, eta, args.steps, rec_steps, P=P)

    # ---- project onto the eigenbasis ----
    v_max = mode_vector(int(pq_max[0]) + 1, int(pq_max[1]) + 1, n_i)
    v_min = mode_vector(int(pq_min[0]) + 1, int(pq_min[1]) + 1, n_i)

    alpha, beta, err_adam, err_gd, spec = [], [], [], [], []
    alpha_gd, beta_gd, spec_gd = [], [], []
    for s in rec_steps:
        e = rec_adam[int(s)] - u_star
        alpha.append(e @ v_max)
        beta.append(e @ v_min)
        # GD follows the classical mode-wise rates, so it is the trajectory for which the
        # extreme-eigenvector plane is actually the right diagnostic (Adam's per-coordinate
        # normalisation crosses the soft direction in ~10 steps and then wanders isotropically).
        e_gd = rec_gd[int(s)] - u_star
        alpha_gd.append(e_gd @ v_max)
        beta_gd.append(e_gd @ v_min)
        # One orthonormal DST-I gives every modal amplitude at once.
        spec.append(np.abs(dstn(e.reshape(n_i, n_i), type=1, norm="ortho")).ravel())
        spec_gd.append(np.abs(dstn(e_gd.reshape(n_i, n_i), type=1, norm="ortho")).ravel())
        err_adam.append(rel(rec_adam[int(s)]))
        err_gd.append(rel(rec_gd[int(s)]))
    e_init = u0 - u_star
    spec0 = np.abs(dstn(e_init.reshape(n_i, n_i), type=1, norm="ortho")).ravel()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        steps=rec_steps, alpha=np.array(alpha), beta=np.array(beta),
        alpha_gd=np.array(alpha_gd), beta_gd=np.array(beta_gd),
        alpha0=e_init @ v_max, beta0=e_init @ v_min,
        err_adam=np.array(err_adam), err_gd=np.array(err_gd),
        spec=np.array(spec, dtype=np.float32), spec0=spec0.astype(np.float32),
        spec_gd=np.array(spec_gd, dtype=np.float32),
        lam=lam, lam_max=lam_max, lam_min=lam_min,
        best_lr=best_lr, eta_gd=eta, grid=args.grid, n_int=n,
        sweep_lr=grid_lr, sweep_rel=np.array(scores), sweep_steps=args.sweep_steps,
        precond=args.precond, record_mode=args.record, skip_gd=args.skip_gd,
        mg_omega=args.mg_omega,
        lr_min=(np.nan if args.lr_min is None else args.lr_min),
    )
    print(f"\nwrote {args.out}")
    print(f"  Adam rel error: {err_adam[0]:.4f} (start) -> {err_adam[-1]:.4f} "
          f"({args.steps} steps);  GD -> {err_gd[-1]:.4f}")


if __name__ == "__main__":
    main()
