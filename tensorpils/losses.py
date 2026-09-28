"""Loss functions for neural-operator training.

All FEM-based losses act on **node** values and route through the problem classes of
:mod:`tensorpils.physics` (which wrap TensorMesh's assembled matrices). They **sum over nodes
and average over the batch**, so the gradient magnitude is O(1) rather than O(1/N).

Poisson (:func:`build_loss`):

* :class:`DataLoss`             — supervised MSE on the grid (labels required)
* :class:`GalerkinLoss`         — ``½‖A u − b‖²``, the unpreconditioned least-squares loss ``L_LS``
* :class:`PreconditionedLSLoss` — ``½‖P(A u − b)‖²`` with ``P ≈ A⁻¹``, the loss ``L_PLS``

Allen–Cahn (:func:`build_ac_loss`): :class:`ACGalerkinLoss`, the step residual over an
autoregressive rollout, bare or preconditioned.

Stokes (:func:`build_stokes_loss`): :class:`StokesGalerkinLoss` (bare) and
:class:`StokesPLSLoss` (block preconditioner used as a norm weight).
"""

from typing import Optional

import torch
import torch.nn as nn

from .physics import PoissonProblem, ACProblem, StokesProblem
from .preconditioners import Preconditioner

__all__ = [
    "DataLoss", "GalerkinLoss", "PreconditionedLSLoss", "build_loss",
    "ACGalerkinLoss", "build_ac_loss",
    "StokesGalerkinLoss", "StokesPLSLoss", "build_stokes_loss",
]


class DataLoss(nn.Module):
    """Supervised MSE on the regular grid (boundary included). ``u_pred``, ``u_true``: ``[B, H, W]``."""

    def forward(self, u_pred: torch.Tensor, u_true: torch.Tensor,
                f_node: torch.Tensor = None) -> torch.Tensor:
        return ((u_pred - u_true) ** 2).mean(dim=0).sum()


class GalerkinLoss(nn.Module):
    """``0.5 ‖A u − b‖²`` on node values ``[B, N]`` (no labels needed)."""

    def __init__(self, problem: PoissonProblem):
        super().__init__()
        self.problem = problem

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        residual = self.problem.residual(u_pred_node, f_node)
        return 0.5 * (residual ** 2).sum(dim=-1).mean()


class PreconditionedLSLoss(nn.Module):
    r"""Preconditioned least-squares loss :math:`L = \tfrac12\|P(Au-b)\|^2`.

    ``P`` is a preconditioner ``P ≈ A⁻¹`` (a multigrid V-cycle, or the exact spectral
    blend). The squared-norm form has Hessian ``AᵀPᵀPA ≈ I`` in the node values
    (mesh-independent conditioning) and gradient ``≈ J_θᵀ(u − u⋆)``, recovering supervised
    training dynamics without labels.
    """

    def __init__(self, problem: PoissonProblem, precond: Preconditioner):
        super().__init__()
        self.problem = problem
        self.precond = precond

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        r = self.problem.residual(u_pred_node, f_node)
        if r.dim() == 1:
            r = r.unsqueeze(0)
        Pr = self.precond(r)                    # ≈ A⁻¹ r
        return 0.5 * (Pr * Pr).sum(dim=-1).mean()


def build_loss(loss_type: str, problem: PoissonProblem,
               precond: Optional[Preconditioner] = None):
    """Factory for the Poisson loss criterion: ``data``, ``galerkin`` or ``pls``."""
    if loss_type == "data":
        return DataLoss()
    if loss_type == "galerkin":
        return GalerkinLoss(problem)
    if loss_type == "pls":
        if precond is None:
            raise ValueError("loss_type='pls' requires a preconditioner")
        return PreconditionedLSLoss(problem, precond)
    raise ValueError(f"Unknown loss type {loss_type!r}")


# ============================ Allen–Cahn losses ============================

class ACGalerkinLoss(nn.Module):
    r"""Least-squares weak-form residual loss for Allen–Cahn (no labels).

    Given a node-space trajectory ``seq`` of shape ``[B, L, N]`` — the ground-truth seed frame
    followed by the model's autoregressive predictions — this penalises the boundary-masked
    convex–concave step residual (:meth:`ACProblem.residual`) over every consecutive pair
    :math:`(u^k, u^{k+1})` as ``½‖R‖²`` (sum over nodes, mean over batch), averaged over the
    pairs. The previous frame :math:`u^k` is **detached**, so each term is a one-step function of
    the network's newest output: together with the trainer's detached rollout inputs this is the
    pushforward training of the paper.

    If ``precond`` is given, the loss is the **preconditioned** least-squares residual
    :math:`\tfrac12\|P R\|^2` with :math:`P\approx J_0^{-1}`, :math:`J_0=a^2A+cM` the frozen
    (``u²=1``) Newton Jacobian, :math:`c = 1/\Delta t + 3\epsilon^2`.
    """

    def __init__(self, problem: ACProblem, a: float, eps: float, dt: float,
                 precond: Optional[Preconditioner] = None):
        super().__init__()
        self.problem = problem
        self.a = a
        self.eps = eps
        self.dt = dt
        self.precond = precond

    def forward(self, seq_node: torch.Tensor) -> torch.Tensor:
        L = seq_node.shape[1]
        if L < 2:
            raise ValueError(f"AC residual needs at least 2 frames, got L={L}")
        total = seq_node.new_zeros(())
        for k in range(L - 1):
            r = self.problem.residual(seq_node[:, k].detach(), seq_node[:, k + 1],
                                      self.a, self.eps, self.dt)
            if self.precond is not None:
                r = self.precond(r)                                   # ½‖P R‖², P ≈ (a²A+cM)⁻¹
            total = total + 0.5 * (r ** 2).sum(dim=-1).mean()         # sum nodes, mean batch
        return total / (L - 1)


def build_ac_loss(problem: ACProblem, a: float, eps: float, dt: float,
                  precond: Optional[Preconditioner] = None):
    """Factory for the Allen–Cahn physics criterion. The supervised data term is a plain
    trajectory MSE handled by the trainer; only the label-free physics loss is assembled here."""
    return ACGalerkinLoss(problem, a=a, eps=eps, dt=dt, precond=precond)


# ============================== Stokes losses ==============================

class StokesGalerkinLoss(nn.Module):
    r"""Unpreconditioned least-squares saddle-point residual ``½‖Kc − b‖²`` (no labels).

    For the indefinite Stokes operator :math:`\mathcal K` the Gauss--Newton matrix is
    :math:`J^\top\mathcal K^2 J`, and since :math:`\mathcal K` is symmetric its squared
    eigenvalues give :math:`\kappa = O(h^{-4})`, so it is expected *not* to train — exactly as
    the bare ``galerkin`` loss stalls for Poisson.
    """

    def __init__(self, problem: StokesProblem):
        super().__init__()
        self.problem = problem

    def forward(self, u_node: torch.Tensor, p_node: torch.Tensor,
                f_node: torch.Tensor) -> torch.Tensor:
        r = self.problem.residual(u_node, p_node, f_node)
        return 0.5 * (r ** 2).sum(dim=-1).mean()          # sum dofs, mean batch


class StokesPLSLoss(nn.Module):
    r"""Preconditioned least-squares loss ``½ rᵀ P r`` with ``r = K c − b``.

    ``P = diag(Â⁻¹, Ŝ⁻¹)`` (see :class:`~tensorpils.preconditioners.stokes.
    StokesBlockPreconditioner`) is used as a **norm weight**, giving

    .. math::
        \nabla L = J^\top[\mathcal K\mathcal P\mathcal K c - \mathcal K\mathcal P b],
        \qquad G(\theta) = J^\top \mathcal K\mathcal P\mathcal K J,

    whose conditioning is governed by :math:`\kappa(\mathcal P)`, i.e. :math:`O(h^{-2})`
    — versus :math:`O(h^{-4})` for :class:`StokesGalerkinLoss`.

    The ``½‖P r‖²`` variant is deliberately *not* offered: for a saddle-point system it
    squares the conditioning back up and works for Poisson only because there ``P ≈ A⁻¹``
    genuinely approximates the inverse, which no block-diagonal preconditioner does for
    :math:`\mathcal K`.
    """

    def __init__(self, problem: StokesProblem, precond: Preconditioner):
        super().__init__()
        self.problem = problem
        self.precond = precond

    def forward(self, u_node: torch.Tensor, p_node: torch.Tensor,
                f_node: torch.Tensor) -> torch.Tensor:
        r = self.problem.residual(u_node, p_node, f_node)
        Pr = self.precond(r)
        return 0.5 * (r * Pr).sum(dim=-1).mean()          # ½ rᵀPr, mean over batch


def build_stokes_loss(loss_type: str, problem: StokesProblem,
                      precond: Optional[Preconditioner] = None):
    """Factory for the label-free Stokes criterion: ``galerkin`` (``½‖Kc − b‖²``) or ``pls``
    (``½ rᵀPr``, which needs the block ``precond``). The supervised ``data`` loss is an FE-norm
    MSE handled by the trainer, so it has no entry here."""
    if loss_type == "galerkin":
        return StokesGalerkinLoss(problem)
    if loss_type == "pls":
        if precond is None:
            raise ValueError("Stokes loss_type='pls' requires a preconditioner")
        return StokesPLSLoss(problem, precond)
    raise ValueError(f"unknown Stokes loss {loss_type!r}; expected 'galerkin' or 'pls'")
