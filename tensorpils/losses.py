"""Physics-informed loss functions for neural-operator training.

All FEM-based losses act on **node** values ``[B, N]`` and route through
:class:`PoissonProblem` (which wraps TensorMesh's assembled matrices):

* :class:`DataLoss`               — supervised MSE on the grid (labels required)
* :class:`GalerkinLoss`           — ``‖A u − b‖²`` weak-form residual
* :class:`DeepRitzLoss`           — energy ``∫(½|∇u|² − f u)`` (+ optional BC penalty)
* :class:`PreconditionedLSLoss`   — ``½‖P(A u − b)‖²`` with ``P ≈ A⁻¹`` (multigrid V-cycle)
* :class:`PreconditionedDeepRitzLoss` — Deep Ritz with the gradient preconditioned by ``P``

Register a new loss by adding a class and a branch in :func:`build_loss`.
"""

from typing import Optional

import torch
import torch.nn as nn

from .physics import PoissonProblem, apply_zero_boundary
from .multigrid import GeometricMultigrid

__all__ = [
    "DataLoss", "GalerkinLoss", "DeepRitzLoss",
    "PreconditionedLSLoss", "PreconditionedDeepRitzLoss", "build_loss",
]


class DataLoss(nn.Module):
    """Supervised MSE on the regular grid. ``u_pred``, ``u_true``: ``[B, H, W]``.

    ``bc_mode='penalty'`` (default): MSE over the full grid (boundary included).
    ``bc_mode='hard'``: MSE over interior nodes only (the outer frame is dropped);
    boundary outputs are left unconstrained at train time and projected to 0 at eval.
    """

    def __init__(self, bc_mode: str = "penalty"):
        super().__init__()
        if bc_mode not in ("penalty", "hard"):
            raise ValueError(f"bc_mode must be 'penalty' or 'hard', got {bc_mode!r}")
        self.bc_mode = bc_mode

    def forward(self, u_pred: torch.Tensor, u_true: torch.Tensor,
                f_node: torch.Tensor = None) -> torch.Tensor:
        if self.bc_mode == "hard":
            return ((u_pred[..., 1:-1, 1:-1] - u_true[..., 1:-1, 1:-1]) ** 2).mean()
        return ((u_pred - u_true) ** 2).mean()


class GalerkinLoss(nn.Module):
    """``‖A u − b‖²`` on node values ``[B, N]`` (no labels needed)."""

    def __init__(self, problem: PoissonProblem):
        super().__init__()
        self.problem = problem

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        residual = self.problem.residual(u_pred_node, f_node)
        return (residual ** 2).mean()


class DeepRitzLoss(nn.Module):
    """Deep Ritz energy on node values ``[B, N]``.

    ``bc_mode='penalty'`` (soft): ``E(u) + λ_bc · mean(u[∂Ω]²)``.
    ``bc_mode='hard'``: project ``u → 0`` on the boundary, then ``E(u_proj)`` (no penalty).
    """

    def __init__(self, problem: PoissonProblem, lambda_bc: float = 100.0,
                 bc_mode: str = "penalty"):
        super().__init__()
        if bc_mode not in ("penalty", "hard"):
            raise ValueError(f"bc_mode must be 'penalty' or 'hard', got {bc_mode!r}")
        self.problem = problem
        self.lambda_bc = lambda_bc
        self.bc_mode = bc_mode

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        mask = self.problem.boundary_mask
        if self.bc_mode == "hard":
            u_for_energy = apply_zero_boundary(u_pred_node, mask)
            return self.problem.energy(u_for_energy, f_node, reduce="mean")

        energy = self.problem.energy(u_pred_node, f_node, reduce="mean")
        if u_pred_node.dim() == 1:
            bc_loss = (u_pred_node[mask] ** 2).mean()
        else:
            bc_loss = (u_pred_node[:, mask] ** 2).mean()
        return energy + self.lambda_bc * bc_loss


class PreconditionedLSLoss(nn.Module):
    r"""Preconditioned least-squares loss :math:`L = \tfrac12\|P(Au-b)\|^2`.

    ``P`` is one geometric-multigrid V-cycle (``P ≈ A⁻¹``). The squared-norm form has
    c-space Hessian ``AᵀPᵀPA ≈ I`` (mesh-independent conditioning) and gradient
    ``≈ J_θᵀ(u − u⋆)``, recovering supervised training dynamics without labels.
    """

    def __init__(self, problem: PoissonProblem, mg: GeometricMultigrid):
        super().__init__()
        self.problem = problem
        self.mg = mg

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        r = self.problem.residual(u_pred_node, f_node)
        if r.dim() == 1:
            r = r.unsqueeze(0)
        Pr = self.mg.v_cycle(r)                 # ≈ A⁻¹ r
        return 0.5 * (Pr * Pr).mean()


class PreconditionedDeepRitzLoss(nn.Module):
    r"""Deep Ritz (hard-BC) with the energy gradient preconditioned by ``P ≈ A⁻¹``.

    Surrogate-gradient trick: ``L = ⟨u, (P r).detach()⟩`` so that ``∂L/∂u = P r``,
    i.e. the descent follows ``M r ≈ A⁻¹(Au − b) = u − u⋆`` (Newton/supervised direction).
    The reported value is the surrogate inner product, not the energy; model selection
    uses validation MSE, so this is harmless.
    """

    def __init__(self, problem: PoissonProblem, mg: GeometricMultigrid):
        super().__init__()
        self.problem = problem
        self.mg = mg

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        r = self.problem.residual(u_pred_node, f_node)
        u = u_pred_node
        if r.dim() == 1:
            r = r.unsqueeze(0)
            u = u.unsqueeze(0)
        Mr = self.mg.v_cycle(r).detach()        # preconditioned descent direction
        return (u * Mr).sum(dim=1).mean()       # surrogate: ∂/∂u = M r


def build_loss(loss_type: str, problem: PoissonProblem,
               lambda_bc: float, bc_mode: str = "penalty",
               mg: Optional[GeometricMultigrid] = None,
               precondition: bool = False):
    """Factory for the loss criterion. Add a branch here to register a new loss."""
    if loss_type == "data":
        return DataLoss(bc_mode=bc_mode)
    if loss_type == "galerkin":
        return GalerkinLoss(problem)
    if loss_type == "deepritz":
        if precondition:
            if mg is None:
                raise ValueError("preconditioned deepritz requires a GeometricMultigrid")
            return PreconditionedDeepRitzLoss(problem, mg)
        return DeepRitzLoss(problem, lambda_bc=lambda_bc, bc_mode=bc_mode)
    if loss_type == "pls":
        if mg is None:
            raise ValueError("loss_type='pls' requires a GeometricMultigrid")
        return PreconditionedLSLoss(problem, mg)
    raise ValueError(f"Unknown loss type {loss_type!r}")
