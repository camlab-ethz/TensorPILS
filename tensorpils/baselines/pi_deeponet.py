r"""PI-DeepONet baseline — strong-form PDE residual by **autodiff** through the trunk.

The other half of the baseline pair. PINO differentiates the prediction with finite
differences on a grid; PI-DeepONet differentiates it exactly, by backpropagating through
the trunk's coordinate input (Wang, Wang & Perdikaris, *Learning the solution operator of
parametric PDEs with physics-informed DeepONets*). The PDE and the reduction are otherwise
the same, so the two baselines differ in architecture and differentiation, not in objective.

**Why the collocation points default to the grid nodes.** Two reasons, one of them forced:

* *fairness* — every arm then sees the same number of points in the same places, so the
  comparison is not confounded by the discretisation budget;
* *necessity for Allen–Cahn* — its residual needs :math:`u^n` at each collocation point, and
  the reference trajectory comes from a FEM Newton solve, so it exists **only** at nodes.
  Off-node collocation would require interpolating the reference, which introduces an error
  that has nothing to do with the method under test.

For Poisson the closed form (``PoissonMultiFrequency.source_term``) is exact anywhere, so
genuinely random collocation is also available (``--pi_colloc random``).

**Per-sample derivatives.** ``coords`` is passed as ``[B, Q, 2]``, not ``[Q, 2]``. With the
shared form, ``grad(u.sum(), coords)`` would return :math:`\sum_b \partial u_b/\partial y`,
the batch *sum* of the gradients, and the per-sample residual could not be recovered. With
the per-sample form ``u[b, q]`` depends only on ``coords[b, q]``, so one ``grad`` call gives
exactly the per-sample derivatives.
"""

import torch
import torch.nn as nn

from .pino import rel_lp

__all__ = ["autodiff_laplacian", "PIDeepONetPoissonLoss", "PIDeepONetACLoss"]


def autodiff_laplacian(model, f_grid: torch.Tensor, coords: torch.Tensor):
    r"""Evaluate ``model`` at ``coords`` and return ``(u, laplacian)``, both ``[B, Q]``.

    ``coords`` must be a ``[B, Q, 2]`` tensor with ``requires_grad=True``. ``create_graph`` is
    kept on throughout so the result is still differentiable with respect to the model
    parameters — without it the residual would have no gradient and training would silently
    do nothing.
    """
    if coords.dim() != 3:
        raise ValueError(f"coords must be [B, Q, 2] for per-sample derivatives, got {tuple(coords.shape)}")
    if not coords.requires_grad:
        raise ValueError("coords must have requires_grad=True")

    u = model.forward_at(f_grid, coords)                                     # [B, Q]
    grad, = torch.autograd.grad(u.sum(), coords, create_graph=True)          # [B, Q, 2]
    lap = torch.zeros_like(u)
    for d in range(2):
        second, = torch.autograd.grad(grad[..., d].sum(), coords, create_graph=True)
        lap = lap + second[..., d]
    return u, lap


class PIDeepONetPoissonLoss(nn.Module):
    r"""Strong-form residual for :math:`-\Delta u = f` at collocation points.

    ``forward(model, f_grid, coords, f_at_coords)``. ``reduction`` matches
    :class:`~tensorpils.baselines.pino.PINOPoissonLoss` so the two baselines are reduced
    identically and only the differentiation differs.

    ``lambda_bc`` adds the soft boundary penalty of the original PI-DeepONet. It is 0 by
    default because the model is built with ``mollify=True`` (hard BC), matching PINO's
    mollifier — that removes one tuned hyperparameter from the baseline rather than adding
    one, and can only help it.
    """

    def __init__(self, reduction: str = "rel", p: int = 2, lambda_bc: float = 0.0):
        super().__init__()
        if reduction not in ("rel", "mse"):
            raise ValueError(f"reduction must be 'rel' or 'mse', got {reduction!r}")
        self.reduction = reduction
        self.p = p
        self.lambda_bc = lambda_bc

    def forward(self, model, f_grid, coords, f_at_coords,
                bc_coords=None) -> torch.Tensor:
        _, lap = autodiff_laplacian(model, f_grid, coords)
        lhs = -lap
        loss = (rel_lp(lhs, f_at_coords, p=self.p) if self.reduction == "rel"
                else ((lhs - f_at_coords) ** 2).mean())
        if self.lambda_bc > 0.0 and bc_coords is not None:
            u_b = model.forward_at(f_grid, bc_coords)
            loss = loss + self.lambda_bc * (u_b ** 2).mean()
        return loss


class PIDeepONetACLoss(nn.Module):
    r"""Strong-form Allen–Cahn step residual over an autoregressive rollout.

    Identical in form to :class:`~tensorpils.baselines.pino.PINOACLoss` — the same dropped
    mass matrix and the same :math:`A \leftrightarrow -\Delta` substitution — except that
    :math:`\Delta u^{n+1}` is the autodiff Laplacian rather than the finite-difference one.

    ``forward(seq_node, laps)`` where ``seq_node`` is ``[B, 1+R, N]`` (seed frame + predicted
    frames, at the collocation nodes) and ``laps`` is ``[B, R, N]``, the Laplacian of each
    predicted frame. The trainer supplies both from a single model evaluation per step.

    ``interior_mask`` (``[N]``, True on interior nodes) restricts the average to interior
    nodes. It is not optional in practice: the rollout has to evaluate the model at *every*
    node in order to feed the next frame back, while
    :class:`~tensorpils.baselines.pino.PINOACLoss` averages over the interior only, as the
    reference implementation does. Without the mask the two baselines' losses differ by the
    ratio of node counts alone — 6 % at ``65^2`` — and their learning rates would not be
    comparable.

    ``detach_coupling`` mirrors :class:`~tensorpils.losses.ACGalerkinLoss` and
    :class:`~tensorpils.baselines.pino.PINOACLoss`: it detaches the previous frame ``u^k``
    inside the residual, and the trainer sets it from ``--bptt_mode`` exactly as the FEM arm
    does. Without it, the baseline and the FEM arm would differ in what the gradient flows
    through as well as in the residual, and the comparison would not be attributable.
    """

    def __init__(self, a: float, eps: float, dt: float,
                 discount: float = 1.0, integrator: str = "convex_concave",
                 interior_mask: torch.Tensor = None, detach_coupling=None):
        super().__init__()
        if integrator not in ("backward_euler", "convex_concave"):
            raise ValueError(f"unknown integrator {integrator!r}")
        self.a, self.eps, self.dt = a, eps, dt
        self.discount = discount
        self.integrator = integrator
        self.detach_coupling = False if detach_coupling is None else detach_coupling
        self.register_buffer("interior_mask", interior_mask)

    def forward(self, seq_node: torch.Tensor, laps: torch.Tensor) -> torch.Tensor:
        a2 = self.a * self.a
        e2 = self.eps * self.eps
        total = seq_node.new_zeros(())
        for k in range(laps.shape[1]):
            uc = seq_node[:, k].detach() if self.detach_coupling else seq_node[:, k]
            un = seq_node[:, k + 1]
            reaction = e2 * ((uc if self.integrator == "convex_concave" else un) - un ** 3)
            r = (un - uc) / self.dt - a2 * laps[:, k] - reaction
            if self.interior_mask is not None:
                r = r[:, self.interior_mask]
            total = total + (self.discount ** k) * (r ** 2).mean()
        return total
