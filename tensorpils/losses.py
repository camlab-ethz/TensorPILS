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

from .physics import PoissonProblem, WaveProblem, ACProblem, apply_zero_boundary
from .preconditioners import Preconditioner

__all__ = [
    "DataLoss", "DataL2Loss", "DataH1Loss", "GalerkinLoss", "DeepRitzLoss",
    "PreconditionedLSLoss", "PreconditionedDeepRitzLoss", "build_loss",
    "WaveGalerkinLoss", "build_wave_loss",
    "ACGalerkinLoss", "build_ac_loss",
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
            return ((u_pred[..., 1:-1, 1:-1] - u_true[..., 1:-1, 1:-1]) ** 2).mean(dim=0).sum()
        return ((u_pred - u_true) ** 2).mean(dim=0).sum()


class DataL2Loss(nn.Module):
    r"""Supervised true-``L²`` loss ``½‖u−u★‖²_{L²} = ½ eᵀ M e`` on node values ``[B, N]``.

    Unlike :class:`DataLoss` (flat grid MSE), the error ``e = u_pred − u_true`` is measured in
    the finite-element ``L²`` norm via the mass matrix ``M`` — the same metric used to report
    validation error. Requires labels ``u_true``.
    """

    def __init__(self, problem: PoissonProblem):
        super().__init__()
        self.problem = problem

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        e = u_pred_node - u_true_node
        Me = self.problem._spmm(self.problem.M, e)      # M e
        return 0.5 * (e * Me).sum(dim=-1).mean()        # ½ mean_B(eᵀ M e)


class DataH1Loss(nn.Module):
    r"""Supervised ``H¹₀``-norm loss ``½‖u−u★‖²_{H¹₀} = ½ eᵀ A e`` on node values ``[B, N]``.

    Like :class:`DataL2Loss` but with the stiffness matrix ``A`` (``∫∇φ·∇φ``) in place of the
    mass matrix, measuring the error in the energy / ``H¹₀`` seminorm ``½∫|∇e|²``. The
    prediction is first projected to zero on the Dirichlet boundary, so ``e = Πu_pred − u★``
    lies in ``H₀¹`` where the seminorm is a genuine norm (Poincaré) — without this the
    constant/boundary mode is in the loss nullspace. Requires labels ``u_true``.
    """

    def __init__(self, problem: PoissonProblem):
        super().__init__()
        self.problem = problem

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        u_bc = apply_zero_boundary(u_pred_node, self.problem.boundary_mask)  # -> H¹₀
        e = u_bc - u_true_node
        Ae = self.problem._spmm(self.problem.A, e)      # A e
        return 0.5 * (e * Ae).sum(dim=-1).mean()        # ½ mean_B(eᵀ A e)


class GalerkinLoss(nn.Module):
    """``0.5 ‖A u − b‖²`` on node values ``[B, N]`` (no labels needed)."""

    def __init__(self, problem: PoissonProblem):
        super().__init__()
        self.problem = problem

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        residual = self.problem.residual(u_pred_node, f_node)
        return 0.5 * (residual ** 2).sum(dim=-1).mean()


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

    ``P`` is a preconditioner ``P ≈ A⁻¹`` (a multigrid V-cycle, or the exact spectral
    blend/power operator). The squared-norm form has c-space Hessian ``AᵀPᵀPA ≈ I``
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
        return 0.5 * (Pr * Pr).sum(dim=-1).mean()    # changed to sum, not mean.


class PreconditionedDeepRitzLoss(nn.Module):
    r"""Deep Ritz (hard-BC) with the energy gradient preconditioned by ``P ≈ A⁻¹``.

    Surrogate-gradient trick: ``L = ⟨u, (P r).detach()⟩`` so that ``∂L/∂u = P r``,
    i.e. the descent follows ``M r ≈ A⁻¹(Au − b) = u − u⋆`` (Newton/supervised direction).
    The reported value is the surrogate inner product, not the energy; model selection
    uses validation MSE, so this is harmless.
    """

    def __init__(self, problem: PoissonProblem, precond: Preconditioner):
        super().__init__()
        self.problem = problem
        self.precond = precond

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        r = self.problem.residual(u_pred_node, f_node)
        u = u_pred_node
        if r.dim() == 1:
            r = r.unsqueeze(0)
            u = u.unsqueeze(0)
        Mr = self.precond(r).detach()           # preconditioned descent direction
        return (u * Mr).sum(dim=1).mean()       # surrogate: ∂/∂u = M r


def build_loss(loss_type: str, problem: PoissonProblem,
               lambda_bc: float, bc_mode: str = "penalty",
               precond: Optional[Preconditioner] = None,
               precondition: bool = False):
    """Factory for the Poisson loss criterion. Add a branch here to register a new loss."""
    if loss_type == "data":
        return DataLoss(bc_mode=bc_mode)
    if loss_type == "data_l2":
        return DataL2Loss(problem)
    if loss_type == "data_h1":
        return DataH1Loss(problem)
    if loss_type == "galerkin":
        return GalerkinLoss(problem)
    if loss_type == "deepritz":
        if precondition:
            if precond is None:
                raise ValueError("preconditioned deepritz requires a preconditioner")
            return PreconditionedDeepRitzLoss(problem, precond)
        return DeepRitzLoss(problem, lambda_bc=lambda_bc, bc_mode=bc_mode)
    if loss_type == "pls":
        if precond is None:
            raise ValueError("loss_type='pls' requires a preconditioner")
        return PreconditionedLSLoss(problem, precond)
    raise ValueError(f"Unknown loss type {loss_type!r}")


# ============================ wave equation losses ============================

class WaveGalerkinLoss(nn.Module):
    r"""Central-difference weak-form residual loss for the wave equation (no labels).

    Given a node-space trajectory ``seq`` of shape ``[B, L, N]`` — the two ground-truth
    seed frames followed by the model's autoregressive predictions — this sums the squared
    boundary-masked residual over every consecutive triple
    :math:`(u^{k-1}, u^k, u^{k+1})`, weighted by ``discount**(k-1)`` so later (more error-prone)
    rollout steps can be down-weighted. See :meth:`WaveProblem.residual`.
    """

    def __init__(self, problem: WaveProblem, c: float, dt: float, discount: float = 1.0):
        super().__init__()
        self.problem = problem
        self.c = c
        self.dt = dt
        self.discount = discount

    def forward(self, seq_node: torch.Tensor) -> torch.Tensor:
        L = seq_node.shape[1]
        if L < 3:
            raise ValueError(f"wave residual needs at least 3 frames, got L={L}")
        total = seq_node.new_zeros(())
        wsum = 0.0
        for k in range(1, L - 1):
            r = self.problem.residual(seq_node[:, k - 1], seq_node[:, k],
                                      seq_node[:, k + 1], self.c, self.dt)
            w = self.discount ** (k - 1)
            total = total + w * (r ** 2).mean()
            wsum += w
        return total / wsum


def build_wave_loss(problem: WaveProblem, c: float, dt: float, discount: float = 1.0):
    """Factory for the wave physics (Galerkin) criterion. The supervised data term is a
    plain trajectory MSE handled by the trainer; only the residual loss is assembled here."""
    return WaveGalerkinLoss(problem, c=c, dt=dt, discount=discount)


# ============================ Allen–Cahn losses ============================

class ACGalerkinLoss(nn.Module):
    r"""Least-squares weak-form residual loss for Allen–Cahn (no labels).

    Given a node-space trajectory ``seq`` of shape ``[B, L, N]`` — the ground-truth seed frame
    followed by the model's autoregressive predictions — this penalises the boundary-masked step
    residual over every consecutive pair :math:`(u^k, u^{k+1})` as ``½‖R‖²`` (sum over nodes,
    mean over batch), weighted by ``discount**k`` so later (more error-prone) rollout steps can be
    down-weighted. Summing over nodes (rather than averaging) keeps the gradient magnitude
    :math:`O(1)` instead of :math:`O(1/N)`.

    ``integrator`` selects the residual form (see :meth:`ACProblem.residual`): ``"backward_euler"``
    (fully implicit) or ``"convex_concave"`` (Eyre split). Both use the ``/dt`` scaling, so the
    loss is ``dt``-independent in gradient scale.
    """

    def __init__(self, problem: ACProblem, a: float, eps: float, dt: float, discount: float = 1.0,
                 integrator: str = "backward_euler"):
        super().__init__()
        self.problem = problem
        self.a = a
        self.eps = eps
        self.dt = dt
        self.discount = discount
        self.integrator = integrator

    def forward(self, seq_node: torch.Tensor) -> torch.Tensor:
        L = seq_node.shape[1]
        if L < 2:
            raise ValueError(f"AC residual needs at least 2 frames, got L={L}")
        total = seq_node.new_zeros(())
        wsum = 0.0
        for k in range(L - 1):
            r = self.problem.residual(seq_node[:, k], seq_node[:, k + 1],
                                      self.a, self.eps, self.dt, integrator=self.integrator)
            w = self.discount ** k
            total = total + w * (0.5 * (r ** 2).sum(dim=-1).mean())   # ½‖R‖²: sum nodes, mean batch
            wsum += w
        return total / wsum


def build_ac_loss(problem: ACProblem, a: float, eps: float, dt: float, discount: float = 1.0,
                  integrator: str = "backward_euler"):
    """Factory for the Allen–Cahn physics (Galerkin least-squares) criterion. The supervised data
    term is a plain trajectory MSE handled by the trainer; only the residual loss is assembled
    here. ``integrator`` picks the residual scheme (``backward_euler`` | ``convex_concave``)."""
    return ACGalerkinLoss(problem, a=a, eps=eps, dt=dt, discount=discount, integrator=integrator)
