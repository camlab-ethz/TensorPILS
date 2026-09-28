r"""Block-diagonal preconditioner for the Taylor-Hood Stokes saddle-point system.

Implements the classical norm-equivalent preconditioner

.. math::
    \mathcal P = \operatorname{diag}\big(\hat A^{-1},\ \hat S^{-1}\big),

with :math:`\hat A^{-1}` one multigrid V-cycle per velocity component (on the unstructured mesh an
algebraic one, :class:`~tensorpils.preconditioners.AMGXPreconditioner`) and
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
pressure block is a positive diagonal, and a V-cycle with matching pre/post smoothing is
symmetric positive definite.

The velocity block is :math:`\mu` times the P2 scalar stiffness and decouples across
components, so one *scalar* V-cycle per component divided by :math:`\mu` is :math:`\hat A^{-1}`.
"""

import torch

from .base import Preconditioner

__all__ = ["StokesBlockPreconditioner"]


class StokesBlockPreconditioner(Preconditioner):
    r"""``P = diag(mg/mu, omega*mu/diag(M_p))`` acting on a monolithic residual.

    Parameters
    ----------
    problem : physics.StokesProblem
        Supplies ``mu``, the lumped pressure mass, and the DOF layout.
    velocity_precond : Preconditioner
        :math:`\hat A^{-1}` for the *scalar* velocity stiffness, ``[B, n_u] -> [B, n_u]``, e.g.
        :class:`~tensorpils.preconditioners.AMGXPreconditioner` on ``problem.A``. It must be SPD,
        since ``P`` is used as a *norm*; the block structure around it is mesh-agnostic.
    schur_omega : float
        Weight :math:`\omega > 0` on the Schur surrogate. As a norm weight it is a free
        parameter; the paper selects it on the validation set (``omega = 16``).
    """

    def __init__(self, problem, velocity_precond, schur_omega: float = 0.5):
        super().__init__()
        self.mg = velocity_precond
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
        print(f"[stokes block precond] P = diag(A^-1/mu, {self.schur_omega:g}*mu/diag(M_p))  "
              f"mu={self.mu:g}  velocity block: {type(self.mg).__name__}")
        if hasattr(self.mg, "report"):
            self.mg.report()
