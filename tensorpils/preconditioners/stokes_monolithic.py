r"""Monolithic multigrid for the Taylor-Hood Stokes system, with an inexact symmetric Uzawa smoother.

This is the *other* preconditioner the methodology note describes (§"The Monolithic Approach"), and
the reason to want it is a change of regime rather than a constant factor. The block-diagonal
:class:`~tensorpils.preconditioners.stokes.StokesBlockPreconditioner` is only **norm-equivalent** to
:math:`\mathcal K^{-1}`, so it can only be used as a norm weight, and the resulting Gauss--Newton
conditioning is :math:`\kappa(\mathcal K\mathcal P\mathcal K)=\mathcal O(h^{-2})`. A monolithic
V-cycle is a genuine **approximate inverse**, :math:`\mathcal P\approx\mathcal K^{-1}`, which unlocks
the *applied* form

.. math::
    L_{\textup{PLS}}(\theta) = \tfrac12\|\mathcal P\mathcal Kc(\theta,f) - \mathcal Pb(f)\|^2,
    \qquad
    G(\theta) = J_\theta^\top(\mathcal{PK})^\top(\mathcal{PK})J_\theta,

whose conditioning is :math:`\mathcal O(1)` — mesh-independent — exactly as for Poisson. Note the
form is the one that is a *dead end* for a block-diagonal :math:`\mathcal P`: what makes it viable
here is only that :math:`\mathcal{PK}\approx I`.

Structure (following the note, deviations flagged):

* **Hierarchy.** Levels coarsen the pressure grid by two, ``n_p -> (n_p+1)//2``; the velocity grid
  is ``2n_p-1`` at every level, so *both* fields nest by a factor of two simultaneously. One
  hierarchy, one prolongation per block.
* **Transfer operators** are block-diagonal on the product space,
  :math:`\Pi=\operatorname{diag}(\Pi_u, \Pi_p)`, with :math:`\mathcal R = \Pi^\top` (Galerkin
  condition). Both blocks are bilinear interpolation on their own nested grid; the velocity block
  is ``kron(Pi_u, I_2)`` for the node-major/component-minor DOF order. *Deviation:* the note's
  :math:`\Pi_u` interpolates the quadratic basis, whereas we reuse the bilinear operator on the Q2
  node set — the same approximation the scalar V-cycle already makes for the velocity block, where
  it measures :math:`\kappa(PA)\approx3`.
* **Coarse operators are re-discretized**, not formed as :math:`\mathcal R\mathcal A\Pi`. For a
  saddle point this is the safer choice: re-discretization keeps every level an inf-sup stable
  Taylor-Hood pair, while the Galerkin projection of a nested pair through bilinear transfers
  carries no such guarantee. It is also what the scalar multigrid in this package does.
* **Smoother:** inexact symmetric Uzawa, three sequential steps per sweep exactly as in the note,
  with :math:`\hat A^{-1}` a Chebyshev-accelerated Jacobi polynomial (degree ``cheb_degree``;
  ``0`` falls back to weighted Jacobi) and
  :math:`\hat S^{-1}=\omega\mu\operatorname{diag}(M_p)^{-1}`. Here the note's restriction
  :math:`\omega\in(0,1]` **does** apply — this is an iteration, not a norm.
* **Coarsest level** is solved exactly with a dense factorization of the operator with the velocity
  Dirichlet rows eliminated and one pressure DOF pinned (:math:`\mathcal K` is singular on the
  constant pressure mode, :math:`B^\top\mathbf 1=0`).

Unlike the block preconditioner, this operator **projects its output** onto the admissible subspace
(zero Dirichlet velocity, zero-mean pressure). For the weighted form that would be cosmetic, since
the residual vanishes there anyway; for the applied form it is not, because anything the V-cycle
leaves on a constrained DOF would be counted by :math:`\|\mathcal Pr\|^2`.

Every operation is a sparse mat--vec or an element-wise op, so autograd flows through the whole
V-cycle and the loss is differentiable in the network output.
"""

import numpy as np
import torch

from tensormesh import LaplaceElementAssembler, MassElementAssembler
from ..meshing import structured_quad9_mesh, structured_quad_mesh
from .base import Preconditioner
from .multigrid import _build_2d_prolongation

__all__ = ["StokesMonolithicMultigrid"]


def _coo(dense: np.ndarray):
    """Dense ``[m, n]`` numpy -> ``(indices [2, nnz], values [nnz])`` float32 torch tensors."""
    t = torch.from_numpy(np.ascontiguousarray(dense)).float().to_sparse_coo().coalesce()
    return t.indices(), t.values()


class StokesMonolithicMultigrid(Preconditioner):
    r"""Monolithic Stokes V-cycle, :math:`\mathcal P \approx \mathcal K^{-1}`.

    Parameters
    ----------
    problem : physics.StokesProblem
        Fine level; supplies ``mu``, the DOF layout and the pressure gauge weight.
    n_levels : int
        Number of levels including the fine one. Coarsening stops at a ``3x3`` pressure grid.
    n_pre, n_post : int
        Uzawa sweeps before / after the coarse-grid correction (:math:`\nu_1,\nu_2` in the note).
    cheb_degree : int
        Chebyshev-Jacobi polynomial degree for :math:`\hat A^{-1}`. ``0`` uses ``n_jacobi`` steps of
        weighted Jacobi instead.
    cheb_ratio : float
        Lower end of the targeted spectrum, ``a = lambda_max / cheb_ratio``. Chebyshev smoothers
        damp ``[a, lambda_max]`` and deliberately leave the rest to the coarse grid.
    schur_omega : float
        :math:`\omega\in(0,1]` in :math:`\hat S^{-1}=\omega\mu\operatorname{diag}(M_p)^{-1}`.
    jacobi_omega : float
        Relaxation for the weighted-Jacobi fallback.
    quadrature_order : int
        Passed to the per-level assembly.
    """

    def __init__(self, problem, n_levels: int = 4, n_pre: int = 1, n_post: int = 1,
                 cheb_degree: int = 2, cheb_ratio: float = 30.0, schur_omega: float = 0.5,
                 n_jacobi: int = 2, jacobi_omega: float = 2.0 / 3.0,
                 quadrature_order: int = 5):
        super().__init__()
        from ..physics import StokesProblem                      # local: avoid a circular import

        self.mu = float(problem.mu)
        self.n_pre, self.n_post = int(n_pre), int(n_post)
        self.cheb_degree = int(cheb_degree)
        self.cheb_ratio = float(cheb_ratio)
        self.schur_omega = float(schur_omega)
        self.n_jacobi = int(n_jacobi)
        self.jacobi_omega = float(jacobi_omega)

        # ---- 1. level dimensions (pressure grid halves; velocity is 2*n_p-1) ----
        dims = [problem.nx_p]
        for _ in range(n_levels - 1):
            nxt = max(3, (dims[-1] + 1) // 2)
            if nxt == dims[-1]:
                break
            dims.append(nxt)
        self.dims = dims
        self.n_levels = len(dims)

        # ---- 2. per-level operators ----
        for L, n_p in enumerate(dims):
            prob = problem if L == 0 else StokesProblem(
                structured_quad9_mesh(nx=n_p, ny=n_p), nx_p=n_p, ny_p=n_p, mu=self.mu,
                quadrature_order=quadrature_order)
            self._register_level(L, prob, quadrature_order)
            self.register_buffer(f"dmask_{L}", prob.dirichlet_mask.clone())
            self.register_buffer(f"w_p_{L}", prob.m_p_lumped.clone().float())

        # ---- 3. block-diagonal prolongations ----
        for L in range(self.n_levels - 1):
            self._register_transfer(L, dims[L], dims[L + 1])

    # ------------------------------------------------------------------ build
    def _register_level(self, L: int, prob, ngp: int):
        """Store the monolithic operator, the scalar velocity stiffness and the mass diagonals."""
        n_p = prob.nx_p
        K = prob.K.to_dense().double().cpu().numpy()
        dmask = prob.dirichlet_mask.cpu().numpy().astype(bool)
        K[dmask, :] = 0.0
        K[:, dmask] = 0.0
        K[dmask, dmask] = 1.0
        idx, val = _coo(K)
        self.register_buffer(f"K_indices_{L}", idx)
        self.register_buffer(f"K_values_{L}", val)
        self.register_buffer(f"n_u_{L}", torch.tensor(prob.n_u))
        self.register_buffer(f"n_p_{L}", torch.tensor(prob.n_p))
        self.register_buffer(f"off_p_{L}", torch.tensor(prob.off_p))

        # scalar Q2 stiffness = the per-component velocity block / mu, boundary-eliminated
        A = prob.A.to_dense().double().cpu().numpy()
        bmask = prob.boundary_mask.cpu().numpy().astype(bool)
        A[bmask, :] = 0.0
        A[:, bmask] = 0.0
        A[bmask, bmask] = 1.0 / max(self.mu, 1e-30)          # so mu*A has a unit diagonal there
        A = self.mu * A
        idx, val = _coo(A)
        self.register_buffer(f"A_indices_{L}", idx)
        self.register_buffer(f"A_values_{L}", val)
        self.register_buffer(f"A_diag_{L}", torch.from_numpy(np.diag(A).copy()).float())
        self.register_buffer(f"bmask_{L}", prob.boundary_mask.clone())

        # lambda_max(D^-1 A) for the Chebyshev interval, by power iteration
        self.register_buffer(f"lam_max_{L}", torch.tensor(self._power_lam_max(A)))

        # diag(M_p) at this level, for the Schur surrogate
        p_mesh = structured_quad_mesh(nx=n_p, ny=n_p)
        M_p = MassElementAssembler.from_mesh(p_mesh, quadrature_order=ngp)(p_mesh.points)
        self.register_buffer(f"mp_diag_{L}",
                             M_p.to_dense().double().diagonal().float().contiguous())

        if L == self.n_levels - 1:                            # coarsest: exact solve
            Kc = K.copy()
            pin = int(prob._pressure_pin)
            Kc[pin, :] = 0.0
            Kc[:, pin] = 0.0
            Kc[pin, pin] = 1.0
            self.register_buffer("K_inv_coarsest",
                                 torch.from_numpy(np.linalg.inv(Kc)).float())
            self.register_buffer("pin_coarsest", torch.tensor(pin))

    @staticmethod
    def _power_lam_max(A: np.ndarray, n_iter: int = 80) -> float:
        """Largest eigenvalue of ``D^-1 A`` by power iteration (symmetric-similar, so real)."""
        d = np.diag(A).copy()
        d[d == 0.0] = 1.0
        rng = np.random.default_rng(0)
        x = rng.standard_normal(A.shape[0])
        lam = 1.0
        for _ in range(n_iter):
            x = x / max(np.linalg.norm(x), 1e-300)
            x = (A @ x) / d
            lam = float(np.linalg.norm(x))
        return lam

    def _register_transfer(self, L: int, n_p_f: int, n_p_c: int):
        """Block-diagonal monolithic prolongation, assembled directly in COO (never densified)."""
        nu_f, nu_c = 2 * n_p_f - 1, 2 * n_p_c - 1            # velocity grids (also nested by 2)
        Pv = _build_2d_prolongation(nu_f, nu_f, nu_c, nu_c)  # [n_u_f, n_u_c]
        Pp = _build_2d_prolongation(n_p_f, n_p_f, n_p_c, n_p_c)
        iv, vv = _coo(Pv)
        ip, vp = _coo(Pp)
        del Pv, Pp

        # velocity DOFs are node-major / component-minor: dof = 2*node + comp
        rows = torch.cat([2 * iv[0] + c for c in (0, 1)])
        cols = torch.cat([2 * iv[1] + c for c in (0, 1)])
        vals = torch.cat([vv, vv])
        off_f, off_c = 2 * (nu_f * nu_f), 2 * (nu_c * nu_c)
        rows = torch.cat([rows, ip[0] + off_f])
        cols = torch.cat([cols, ip[1] + off_c])
        vals = torch.cat([vals, vp])

        self.register_buffer(f"Pi_indices_{L}", torch.stack([rows, cols]))
        self.register_buffer(f"Pi_values_{L}", vals)
        self.register_buffer(f"Pi_shape_{L}",
                             torch.tensor([off_f + n_p_f * n_p_f, off_c + n_p_c * n_p_c]))

    # ------------------------------------------------- sparse reconstruction
    def _n_dofs(self, L: int) -> int:
        return int(self.get_buffer(f"off_p_{L}")) + int(self.get_buffer(f"n_p_{L}"))

    def _K(self, L: int):
        n = self._n_dofs(L)
        return torch.sparse_coo_tensor(self.get_buffer(f"K_indices_{L}"),
                                       self.get_buffer(f"K_values_{L}"), (n, n))

    def _A(self, L: int):
        n = int(self.get_buffer(f"n_u_{L}"))
        return torch.sparse_coo_tensor(self.get_buffer(f"A_indices_{L}"),
                                       self.get_buffer(f"A_values_{L}"), (n, n))

    def _Pi(self, L: int):
        shp = self.get_buffer(f"Pi_shape_{L}")
        return torch.sparse_coo_tensor(self.get_buffer(f"Pi_indices_{L}"),
                                       self.get_buffer(f"Pi_values_{L}"),
                                       (int(shp[0]), int(shp[1])))

    @staticmethod
    def _mm(mat, x: torch.Tensor) -> torch.Tensor:
        """``x @ matᵀ`` for batched rows ``[B, n]`` with a sparse ``mat``."""
        return torch.sparse.mm(mat, x.transpose(-1, -2)).transpose(-1, -2)

    # ------------------------------------------------------------ components
    def _split(self, L: int, c: torch.Tensor):
        off = int(self.get_buffer(f"off_p_{L}"))
        n_u = int(self.get_buffer(f"n_u_{L}"))
        return c[..., :off].reshape(*c.shape[:-1], n_u, 2), c[..., off:]

    def _join(self, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.cat([u.reshape(*u.shape[:-2], u.shape[-2] * 2), p], dim=-1)

    def _A_hat_inv(self, L: int, r_u: torch.Tensor) -> torch.Tensor:
        r"""Approximate :math:`A^{-1}r_u` on ``[B, n_u, 2]`` — Chebyshev-Jacobi or weighted Jacobi.

        The Chebyshev semi-iteration is run on the interval
        ``[lambda_max/cheb_ratio, lambda_max]`` of ``D^-1 A``: a smoother only needs the top of the
        spectrum, and targeting the bottom too would waste the polynomial's degrees of freedom on
        modes the coarse grid resolves.
        """
        A_sp = self._A(L)
        d = self.get_buffer(f"A_diag_{L}").to(r_u.dtype).unsqueeze(-1)     # [n_u, 1]

        def A_apply(e):
            return torch.stack([self._mm(A_sp, e[..., 0]), self._mm(A_sp, e[..., 1])], dim=-1)

        if self.cheb_degree <= 0:
            e = torch.zeros_like(r_u)
            for _ in range(self.n_jacobi):
                e = e + self.jacobi_omega * (r_u - A_apply(e)) / d
            return e

        b = float(self.get_buffer(f"lam_max_{L}"))
        a = b / self.cheb_ratio
        theta, delta = 0.5 * (b + a), 0.5 * (b - a)
        sigma = theta / delta
        rho = 1.0 / sigma
        e = torch.zeros_like(r_u)
        dvec = r_u / d / theta
        for _ in range(self.cheb_degree - 1):
            e = e + dvec
            res = r_u - A_apply(e)
            rho_new = 1.0 / (2.0 * sigma - rho)
            dvec = rho_new * rho * dvec + (2.0 * rho_new / delta) * (res / d)
            rho = rho_new
        return e + dvec

    def _uzawa(self, L: int, x: torch.Tensor, rhs: torch.Tensor, n_sweeps: int) -> torch.Tensor:
        r"""``n_sweeps`` inexact symmetric Uzawa sweeps for ``K x = rhs`` at level ``L``.

        One sweep is the note's three steps:
        ``u += Â⁻¹(f − Au − Bᵀp)``, then ``p += Ŝ⁻¹(Bu − g)``, then ``u += Â⁻¹(f − Au − Bᵀp)``.
        Each residual is one monolithic sparse mat--vec, since the momentum rows of ``K`` give
        ``Au + Bᵀp`` and the continuity rows give ``Bu``.
        """
        K_sp = self._K(L)
        s_inv = (self.schur_omega * self.mu
                 / self.get_buffer(f"mp_diag_{L}").to(x.dtype).clamp_min(1e-30))
        f_u, g_p = self._split(L, rhs)

        for _ in range(n_sweeps):
            r_u, _ = self._split(L, rhs - self._mm(K_sp, x))                # f − Au − Bᵀp
            u, p = self._split(L, x)
            u = u + self._A_hat_inv(L, r_u)
            x = self._join(u, p)

            _, Bu = self._split(L, self._mm(K_sp, x))                       # continuity rows = Bu
            p = p + s_inv * (Bu - g_p)
            x = self._join(u, p)

            r_u, _ = self._split(L, rhs - self._mm(K_sp, x))
            u = u + self._A_hat_inv(L, r_u)
            x = self._join(u, p)
        return x

    # ------------------------------------------------------------- V-cycle
    def _v_cycle_rec(self, L: int, rhs: torch.Tensor) -> torch.Tensor:
        if L == self.n_levels - 1:
            pin = int(self.get_buffer("pin_coarsest"))
            rhs = rhs.clone()
            rhs[..., pin] = 0.0                 # the pinned pressure DOF carries no equation
            return rhs @ self.K_inv_coarsest.to(rhs.dtype).transpose(0, 1)

        x = torch.zeros_like(rhs)
        x = self._uzawa(L, x, rhs, self.n_pre)
        defect = rhs - self._mm(self._K(L), x)
        Pi = self._Pi(L)
        d_c = self._mm(Pi.transpose(0, 1), defect)              # restrict, R = Pi^T
        e_c = self._v_cycle_rec(L + 1, d_c)
        x = x + self._mm(Pi, e_c)                               # prolong + correct
        return self._uzawa(L, x, rhs, self.n_post)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """One V-cycle: ``P r ≈ K⁻¹ r``, on ``[B, n_dofs]`` (or ``[n_dofs]``).

        The output is projected onto the admissible subspace — zero on Dirichlet velocity DOFs,
        zero mean in the lumped-``M_p`` inner product on the pressure block. Both are required for
        the applied form ``½‖Pr‖²``, which would otherwise pay for components the discrete problem
        does not contain.
        """
        squeeze = (r.dim() == 1)
        if squeeze:
            r = r.unsqueeze(0)
        out = self._v_cycle_rec(0, r)

        out = out.masked_fill(self.get_buffer("dmask_0").to(out.device), 0.0)
        off = int(self.get_buffer("off_p_0"))
        w = self.get_buffer("w_p_0").to(dtype=out.dtype, device=out.device)
        p = out[..., off:]
        p = p - (p * w).sum(dim=-1, keepdim=True) / w.sum()
        out = torch.cat([out[..., :off], p], dim=-1)
        return out.squeeze(0) if squeeze else out

    # ------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def cycle_report(self, n_cycles: int = 6, batch: int = 2, seed: int = 0):
        r"""Residual reduction of the V-cycle used as a stationary iteration for ``K e = r``.

        Returns the list of relative residual norms after each cycle. The average factor is the
        contraction rate: below ~0.3 the operator is a genuine approximate inverse and the applied
        loss form is justified; near 1 it is not, and the weighted form should be used instead.

        The test residual must lie in the **range** of ``K``, or no iteration could drive the
        residual to zero and the measurement would report the inconsistency rather than the
        smoother. The continuity rows of any ``K``-image satisfy :math:`\mathbf 1^\top Bu = 0`
        (plain sum, since :math:`\mathbf 1^\top B = 0`), which is what a real training residual
        satisfies automatically — so the random test residual is projected to plain zero *sum*, not
        to the mass-weighted zero *mean* that gauges a pressure solution.
        """
        n = self._n_dofs(0)
        g = torch.Generator(device="cpu").manual_seed(seed)
        dev = self.get_buffer("K_values_0").device
        dtype = self.get_buffer("K_values_0").dtype
        r = torch.randn(batch, n, generator=g, dtype=dtype).to(dev)
        r = r.masked_fill(self.get_buffer("dmask_0").to(dev), 0.0)
        off = int(self.get_buffer("off_p_0"))
        p = r[..., off:]
        r = torch.cat([r[..., :off], p - p.mean(dim=-1, keepdim=True)], dim=-1)

        K_sp = self._K(0)
        x = torch.zeros_like(r)
        r0 = r.norm(dim=-1)
        hist = []
        for _ in range(n_cycles):
            x = x + self.forward(r - self._mm(K_sp, x))
            hist.append(float(((r - self._mm(K_sp, x)).norm(dim=-1) / r0).mean()))
        return hist

    def report(self) -> None:
        smoother = (f"Chebyshev-Jacobi deg={self.cheb_degree} (ratio {self.cheb_ratio:g})"
                    if self.cheb_degree > 0
                    else f"weighted Jacobi x{self.n_jacobi} (omega={self.jacobi_omega:.3f})")
        print(f"[stokes monolithic MG] P ~ K^-1   levels={self.n_levels}  "
              f"pressure dims={self.dims}  Uzawa pre/post={self.n_pre}/{self.n_post}  "
              f"A_hat^-1: {smoother}  schur_omega={self.schur_omega:g}  mu={self.mu:g}")
        hist = self.cycle_report()
        rates = [hist[0]] + [hist[i] / max(hist[i - 1], 1e-300) for i in range(1, len(hist))]
        print("  V-cycle relative residual: " + "  ".join(f"{v:.3e}" for v in hist))
        print(f"  contraction per cycle    : {sum(rates)/len(rates):.4f} "
              f"(mean of {len(rates)})")
