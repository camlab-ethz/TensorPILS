r"""Block-diagonal preconditioner for the Taylor-Hood Stokes saddle-point system.

Implements the classical norm-equivalent preconditioner

.. math::
    \mathcal P = \operatorname{diag}\big(\hat A^{-1},\ \hat S^{-1}\big),

with :math:`\hat A^{-1}` a geometric-multigrid V-cycle on the velocity block and
:math:`\hat S^{-1} = \omega\,\mu\,\operatorname{diag}(M_p)^{-1}` the lumped pressure-mass
surrogate for the inverse Schur complement :math:`S = BA^{-1}B^\top \simeq \mu^{-1}M_p`.

This is the preconditioner of the ``pls`` Stokes loss
:math:`L = \tfrac12\|\mathcal Kc - f\|_{\mathcal P}^2 = \tfrac12 r^\top\mathcal P r`,
whose Gauss--Newton matrix is :math:`J^\top\mathcal K\mathcal P\mathcal K J`. Note the
loss uses :math:`\mathcal P` as a **norm weight**, not as an applied operator: for a saddle
point system :math:`\tfrac12\|\mathcal P r\|^2` would square the conditioning back to
:math:`O(h^{-4})` (Lemma "saddle point preconditioners"), whereas the weighted form stays
at :math:`O(h^{-2})`. That distinction is the whole point, so :class:`StokesBlockPreconditioner`
is deliberately **not** interchangeable with the Poisson preconditioners in ``pls``.

Being a norm weight, :math:`\mathcal P` must be symmetric positive definite. It is: the
pressure block is a positive diagonal, and the V-cycle operator with matching pre/post
smoothing and :math:`R = P^\top` is symmetric with :math:`\lambda_{\min}\approx 0.25 > 0`
(see ``tests/test_stokes_operator.py::test_block_preconditioner_is_spd``).

Two structural facts let the *scalar* Poisson multigrid serve the velocity block unchanged:

* the velocity block is :math:`\mu` times the Q2 scalar stiffness and decouples across
  components, so one V-cycle per component divided by :math:`\mu` is :math:`\hat A^{-1}`;
* the Q2 velocity nodes form a uniform grid, so the Q1 multigrid hierarchy built on that
  node set is spectrally equivalent to the Q2 operator — measured
  :math:`\kappa(PA) \approx 3` against :math:`\kappa(A) \approx 10^2`. A preconditioner only
  needs spectral equivalence, not to be the exact inverse.
"""

import torch

from .base import Preconditioner
from .multigrid import GeometricMultigrid

__all__ = ["StokesBlockPreconditioner", "StokesBlendPreconditioner"]


class StokesBlockPreconditioner(Preconditioner):
    r"""``P = diag(mg/mu, omega*mu/diag(M_p))`` acting on a monolithic residual.

    Parameters
    ----------
    problem : physics.StokesProblem
        Supplies the grids, ``mu``, the lumped pressure mass, and the DOF layout.
    mg_levels, mg_pre_smooth, mg_post_smooth, mg_omega
        Velocity V-cycle settings (see :class:`GeometricMultigrid`).
    schur_omega : float
        Relaxation :math:`\omega\in(0,1]` on the Schur surrogate. ``0.5`` is a safe
        default; raise it toward ``1`` while the loss stays stable.
    """

    def __init__(self, problem, mg_levels: int = 4, mg_pre_smooth: int = 2,
                 mg_post_smooth: int = 2, mg_omega: float = 2.0 / 3.0,
                 schur_omega: float = 0.5):
        super().__init__()
        nx, ny = problem.grid_size
        self.mg = GeometricMultigrid(
            nx_fine=nx, ny_fine=ny, n_levels=mg_levels,
            pre_smooth=mg_pre_smooth, post_smooth=mg_post_smooth, omega=mg_omega,
        )
        self.mu = float(problem.mu)
        self.schur_omega = float(schur_omega)
        self.n_u = problem.n_u
        self.n_p = problem.n_p
        self.off_p = problem.off_p
        self.register_buffer("m_p_lumped", problem.m_p_lumped.clone())

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Apply ``P`` to a monolithic residual ``[B, n_dofs]`` (or ``[n_dofs]``)."""
        squeeze = (r.dim() == 1)
        if squeeze:
            r = r.unsqueeze(0)
        r_u = r[..., :self.off_p].reshape(-1, self.n_u, 2)
        r_p = r[..., self.off_p:]

        # velocity: one V-cycle per component; A = mu * A_scalar  =>  A^-1 = mg / mu
        pu = torch.stack([self.mg(r_u[..., 0]), self.mg(r_u[..., 1])], dim=-1) / self.mu
        # pressure: S^-1 ~ mu * M_p^-1, lumped
        w = self.m_p_lumped.to(dtype=r.dtype, device=r.device)
        pp = self.schur_omega * self.mu * r_p / w

        out = torch.cat([pu.reshape(r.shape[0], self.off_p), pp], dim=-1)
        return out.squeeze(0) if squeeze else out

    def report(self) -> None:
        print(f"[stokes block precond] P = diag(mg/mu, {self.schur_omega:g}*mu/diag(M_p))  "
              f"mu={self.mu:g}  n_levels={self.mg.n_levels}  "
              f"pre/post={self.mg.pre_smooth}/{self.mg.post_smooth}  dims={self.mg.dims}")


class StokesBlendPreconditioner(Preconditioner):
    r"""Convex blend ``P_t = (1-t)·alpha·I + t·P_block`` — one knob from bare to preconditioned.

    The Stokes analogue of :class:`~tensorpils.preconditioners.spectral.SpectralPreconditioner`'s
    ``blend`` family. Both endpoints are exact and need no special-casing:

    * ``t=0`` gives ``P = alpha·I``, so ``½ rᵀPr = ½·alpha·‖r‖²`` — the bare least-squares
      (``galerkin``) loss up to a positive constant, which changes nothing about the conditioning
      and, under Adam, essentially nothing about the step size either;
    * ``t=1`` gives exactly :class:`StokesBlockPreconditioner`.

    Unlike the Poisson blend there is no ``A^{-1}`` endpoint to interpolate toward: no
    block-diagonal operator approximates :math:`\mathcal K^{-1}` (that is the whole point of the
    weighted-norm form), so the far end of the knob is the block preconditioner itself. What the
    sweep buys is the *conditioning axis*: :math:`\kappa(\mathcal K\mathcal P_t\mathcal K)` moves
    continuously from :math:`\mathcal O(h^{-4})` at ``t=0`` to :math:`\mathcal O(h^{-2})` at
    ``t=1``, so the error-versus-conditioning collapse can be measured rather than argued.

    ``alpha`` is :math:`\lambda_{\max}(\mathcal P_{\textup{block}})`, estimated once by power
    iteration confined to the admissible subspace (zero Dirichlet velocity, zero-mean pressure).
    Scaling the identity end by it matches the *spectral radius* of the two endpoints, so the loss
    magnitude — hence the effective step size at a fixed learning rate — stays comparable along the
    sweep instead of jumping by orders of magnitude at ``t`` near 0. A convex combination of SPD
    operators is SPD, so the weighted norm stays a norm for every ``t``.

    Parameters
    ----------
    base : StokesBlockPreconditioner
        The ``t=1`` endpoint.
    strength : float in [0, 1]
        The blend parameter ``t``.
    n_power : int
        Power iterations for ``alpha`` (``0`` keeps ``alpha=1``).
    """

    def __init__(self, base: StokesBlockPreconditioner, strength: float = 1.0,
                 n_power: int = 60, seed: int = 0):
        super().__init__()
        if not (0.0 <= float(strength) <= 1.0):
            raise ValueError(f"strength must be in [0, 1], got {strength}")
        self.base = base
        self.strength = float(strength)
        self.alpha = self._power_iterate(base, n_power, seed) if n_power > 0 else 1.0

    @staticmethod
    def _power_iterate(base: StokesBlockPreconditioner, n_iter: int, seed: int) -> float:
        r"""``lambda_max`` of ``P_block`` by power iteration on the admissible subspace.

        ``P_block`` is SPD, so plain power iteration converges to the largest eigenvalue. The
        iterate is projected each step (Dirichlet velocity DOFs zeroed, pressure block set to
        zero mean) so the estimate describes ``P`` where the loss actually acts.
        """
        w = base.m_p_lumped
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(1, base.off_p + base.n_p, generator=g,
                        dtype=w.dtype).to(w.device)
        lam = 1.0
        with torch.no_grad():
            for _ in range(n_iter):
                x = x / x.norm().clamp_min(1e-30)
                x = base(x)
                p = x[:, base.off_p:]
                x[:, base.off_p:] = p - (p * w).sum(-1, keepdim=True) / w.sum()
                lam = float(x.norm())
        return lam

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Apply ``P_t`` to a monolithic residual ``[B, n_dofs]`` (or ``[n_dofs]``)."""
        if self.strength >= 1.0:
            return self.base(r)
        out = ((1.0 - self.strength) * self.alpha) * r
        if self.strength > 0.0:                      # t=0 skips the V-cycle entirely
            out = out + self.strength * self.base(r)
        return out

    def report(self) -> None:
        print(f"[stokes blend precond] P_t = (1-t)*alpha*I + t*P_block   t={self.strength:.3f}  "
              f"alpha={self.alpha:.4e}"
              + ("   (t=0: identical to the bare galerkin loss up to the constant alpha)"
                 if self.strength == 0.0 else ""))
        self.base.report()
