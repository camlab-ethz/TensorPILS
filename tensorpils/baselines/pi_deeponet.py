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

**Per-sample derivatives.** ``coords`` is passed as ``[B, Q, 2]``, not ``[Q, 2]``. With the
shared form, ``grad(u.sum(), coords)`` would return :math:`\sum_b \partial u_b/\partial y`,
the batch *sum* of the gradients, and the per-sample residual could not be recovered. With
the per-sample form ``u[b, q]`` depends only on ``coords[b, q]``, so one ``grad`` call gives
exactly the per-sample derivatives.
"""

import torch
import torch.nn as nn

from .pino import rel_lp

__all__ = ["autodiff_laplacian", "autodiff_stokes", "stokes_reduce", "PIDeepONetPoissonLoss",
           "PIDeepONetACLoss", "PIDeepONetStokesLoss"]


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
    r"""Strong-form residual for :math:`-\Delta u = f` at collocation points, plus an
    optional soft boundary penalty.

    ``forward(model, f_grid, coords, f_at_coords, bc_coords=None)``. ``reduction`` matches
    :class:`~tensorpils.baselines.pino.PINOPoissonLoss` so the two baselines are reduced
    identically and only the differentiation differs.

    ``lambda_bc`` weights the soft boundary penalty of the original PI-DeepONet. It is 0 when
    the model imposes the BC hard through the mollifier (``--pideeponet_bc mollifier``), which
    is the reference's choice. Under ``--pideeponet_bc zero`` — the counterpart of
    ``--pino_bc zero``, boundary nodes of the grid output set to zero as our own arms do — it
    is the *only* thing that couples the residual to the boundary condition: the autodiff
    Laplacian at interior points is blind to the boundary values and to any mask on them, so
    without the penalty the objective is invariant under adding harmonic functions. (PINO's
    finite-difference stencil sees the zeroed boundary ring; a pointwise derivative does not.)

    The penalty is expressed in the residual's own units, so ``lambda_bc = 1`` is a meaningful
    default rather than a tuning target:

    * ``rel`` — ``mean_b RMS(u_b|dOmega) / RMS(u_b|Omega)``, dimensionless like the residual's
      ``||Lu - f|| / ||f||``. The denominator is detached, so the term is a per-sample
      adaptively weighted boundary RMS: it drives the boundary values to zero relative to the
      solution's own scale and gives no incentive to inflate the interior. The normalisation
      is not cosmetic: at ``K=10`` the solutions have RMS ``~1.6e-3`` (``f`` ~0.8) while the
      relative residual is ``O(1)``, so a plain mean-square boundary penalty at unit weight
      would be five to six orders of magnitude too weak to matter.
    * ``mse`` — the mean square of the boundary values, the original PI-DeepONet form, paired
      with the plain mean-square residual.
    """

    def __init__(self, reduction: str = "rel", p: int = 2, lambda_bc: float = 0.0):
        super().__init__()
        if reduction not in ("rel", "mse"):
            raise ValueError(f"reduction must be 'rel' or 'mse', got {reduction!r}")
        self.reduction = reduction
        self.p = p
        self.lambda_bc = lambda_bc

    def boundary_term(self, model, f_grid, bc_coords, u_int: torch.Tensor) -> torch.Tensor:
        """The penalty alone (unweighted); ``u_int`` is ``[B, Q]`` at the interior collocation
        points and only sets the scale of the ``rel`` form."""
        u_b = model.forward_at(f_grid, bc_coords)                                # [B, Qb]
        if self.reduction == "mse":
            return (u_b ** 2).mean()
        rms_b = (u_b.pow(2).mean(dim=-1) + 1e-24).sqrt()                         # [B]
        rms_int = (u_int.detach().pow(2).mean(dim=-1) + 1e-24).sqrt()            # [B]
        return (rms_b / rms_int).mean()

    def forward(self, model, f_grid, coords, f_at_coords,
                bc_coords=None) -> torch.Tensor:
        u_int, lap = autodiff_laplacian(model, f_grid, coords)
        lhs = -lap
        loss = (rel_lp(lhs, f_at_coords, p=self.p) if self.reduction == "rel"
                else ((lhs - f_at_coords) ** 2).mean())
        if self.lambda_bc > 0.0 and bc_coords is not None:
            loss = loss + self.lambda_bc * self.boundary_term(model, f_grid, bc_coords, u_int)
        return loss


class PIDeepONetACLoss(nn.Module):
    r"""Strong-form Allen–Cahn step residual over an autoregressive rollout.

    Identical in form to :class:`~tensorpils.baselines.pino.PINOACLoss` — the same dropped
    mass matrix, the same :math:`A \leftrightarrow -\Delta` substitution and the same detached
    previous frame — except that :math:`\Delta u^{n+1}` is the autodiff Laplacian rather than
    the finite-difference one.

    ``forward(seq_node, laps)`` where ``seq_node`` is ``[B, 1+R, Q]`` (seed frame + predicted
    frames, at the interior collocation nodes) and ``laps`` is ``[B, R, Q]``, the Laplacian of
    each predicted frame. The trainer supplies both from a single model evaluation per step.
    """

    def __init__(self, a: float, eps: float, dt: float):
        super().__init__()
        self.a, self.eps, self.dt = a, eps, dt

    def forward(self, seq_node: torch.Tensor, laps: torch.Tensor) -> torch.Tensor:
        a2 = self.a * self.a
        e2 = self.eps * self.eps
        total = seq_node.new_zeros(())
        for k in range(laps.shape[1]):
            uc = seq_node[:, k].detach()
            un = seq_node[:, k + 1]
            reaction = e2 * (uc - un ** 3)
            r = (un - uc) / self.dt - a2 * laps[:, k] - reaction
            total = total + (r ** 2).mean()
        return total


# ------------------------------------------------------------------------------- Stokes

def stokes_reduce(r_mom: torch.Tensor, r_div: torch.Tensor, f: torch.Tensor,
                  reduction: str = "rel", p: int = 2, div_weight: float = 1.0) -> torch.Tensor:
    r"""Reduce a strong-form Stokes residual (the PI-DeepONet Stokes arm).

    ``r_mom`` ``[B, ..., 2]`` is the momentum residual :math:`-\mu\Delta u + \nabla p - f`,
    ``r_div`` ``[B, ...]`` the continuity residual :math:`\nabla\cdot u`, ``f`` ``[B, ..., 2]``
    the body force at the same points (any trailing layout; everything is flattened per sample).

    ``rel``: PINO's relative ratio on the **stacked** system, ``mean_b ‖(r_mom, w·r_div)‖ /
    ‖(f, 0)‖`` — the strong-form analogue of the FEM least-squares loss ``½‖Kc − b‖²`` with
    ``b = (M_u f, 0)``, in which momentum and continuity rows are likewise stacked with their
    natural units. ``mse``: ``mean(r_mom²) + w² mean(r_div²)``. ``div_weight`` (``w``) is the
    one knob a strong-form saddle point needs beyond Poisson: the two equations have different
    units (``div u ~ k u`` against ``f ~ mu k² u``), so it is tuned on validation like a
    learning rate.
    """
    b = r_mom.shape[0]
    rm = r_mom.reshape(b, -1)
    rd = div_weight * r_div.reshape(b, -1)
    if reduction == "mse":
        return (rm ** 2).mean() + (rd ** 2).mean()
    if reduction != "rel":
        raise ValueError(f"reduction must be 'rel' or 'mse', got {reduction!r}")
    # ``r_mom`` is already the residual (the loss subtracts f before calling), so the ratio is
    # formed directly -- NOT via rel_lp(residual, f), which would subtract f a second time and
    # make the minimiser the solution for the force 2f (a bug that showed up as exactly 100 %
    # velocity error; pinned by test_stokes_rel_reduction_is_the_ratio_of_residual_to_force).
    res = torch.linalg.vector_norm(torch.cat([rm, rd], dim=1), ord=p, dim=1)
    ref = torch.linalg.vector_norm(f.reshape(b, -1), ord=p, dim=1)
    return (res / ref.clamp_min(1e-12)).mean()



def autodiff_stokes(model, f_grid: torch.Tensor, coords: torch.Tensor, p_scale: float = 1.0):
    r"""Evaluate a 3-channel model at ``coords`` and return every derivative the Stokes
    residual needs, all ``[B, Q, ...]``:

    ``u`` ``[B, Q, 2]`` (raw velocity), ``p`` ``[B, Q]`` (pressure, already times
    ``p_scale``), ``lap_u`` ``[B, Q, 2]``, ``grad_p`` ``[B, Q, 2]`` and ``div_u`` ``[B, Q]``.

    Same contract as :func:`autodiff_laplacian`: ``coords`` is the per-sample ``[B, Q, 2]``
    form with ``requires_grad=True``, and ``create_graph`` stays on so the residual remains
    differentiable in the parameters. Three first-order and four second-order ``grad`` calls.
    """
    if coords.dim() != 3:
        raise ValueError(f"coords must be [B, Q, 2] for per-sample derivatives, got {tuple(coords.shape)}")
    if not coords.requires_grad:
        raise ValueError("coords must have requires_grad=True")
    out = model.forward_at(f_grid, coords)                                   # [B, Q, 3]
    if out.dim() != 3 or out.shape[-1] != 3:
        raise ValueError(f"a Stokes model must emit 3 channels at each query, got {tuple(out.shape)}")
    ux, uy, p = out[..., 0], out[..., 1], out[..., 2] * p_scale
    gx, = torch.autograd.grad(ux.sum(), coords, create_graph=True)           # [B, Q, 2]
    gy, = torch.autograd.grad(uy.sum(), coords, create_graph=True)
    gp, = torch.autograd.grad(p.sum(), coords, create_graph=True)
    lap = []
    for g in (gx, gy):
        acc = torch.zeros_like(ux)
        for d in range(2):
            second, = torch.autograd.grad(g[..., d].sum(), coords, create_graph=True)
            acc = acc + second[..., d]
        lap.append(acc)
    return dict(u=torch.stack([ux, uy], dim=-1), p=p, lap_u=torch.stack(lap, dim=-1),
                grad_p=gp, div_u=gx[..., 0] + gy[..., 1])


class PIDeepONetStokesLoss(nn.Module):
    r"""Strong-form Stokes residual at collocation points, differentiated by autodiff.

    Momentum and continuity residuals are stacked and reduced by :func:`stokes_reduce`.
    ``forward(model, f_grid, coords, f_at_coords, bc_coords=None)`` with ``f_at_coords``
    ``[B, Q, 2]``; ``f_grid`` is whatever the model's branch expects (the trainer hands it
    ``f / f_scale``, as :meth:`StokesTrainer._predict` does for every arm), and ``p_scale`` puts
    the third channel in physical units.

    ``lambda_bc`` is the velocity boundary penalty — the same relative form as the Poisson one
    (boundary RMS over the detached interior RMS, both components stacked) for the same reason:
    the autodiff Laplacian cannot see a boundary mask. The pressure has no boundary condition, only a gauge, which the trainer fixes by
    projection at evaluation and which ``∇p`` never sees.
    """

    def __init__(self, mu: float, reduction: str = "rel", p: int = 2, div_weight: float = 1.0,
                 lambda_bc: float = 0.0, p_scale: float = 1.0):
        super().__init__()
        if reduction not in ("rel", "mse"):
            raise ValueError(f"reduction must be 'rel' or 'mse', got {reduction!r}")
        self.mu = float(mu)
        self.reduction = reduction
        self.p = p
        self.div_weight = float(div_weight)
        self.lambda_bc = float(lambda_bc)
        self.p_scale = float(p_scale)

    def boundary_term(self, model, f_grid, bc_coords, u_int: torch.Tensor) -> torch.Tensor:
        """Unweighted velocity boundary penalty; ``u_int`` ``[B, Q, 2]`` sets the scale."""
        out_b = model.forward_at(f_grid, bc_coords)                                  # [B, Qb, 3]
        u_b = out_b[..., :2]
        if self.reduction == "mse":
            return (u_b ** 2).mean()
        rms_b = (u_b.pow(2).mean(dim=(-2, -1)) + 1e-24).sqrt()                       # [B]
        rms_int = (u_int.detach().pow(2).mean(dim=(-2, -1)) + 1e-24).sqrt()
        return (rms_b / rms_int).mean()

    def forward(self, model, f_grid, coords, f_at_coords, bc_coords=None) -> torch.Tensor:
        d = autodiff_stokes(model, f_grid, coords, p_scale=self.p_scale)
        r_mom = -self.mu * d["lap_u"] + d["grad_p"] - f_at_coords                    # [B, Q, 2]
        loss = stokes_reduce(r_mom, d["div_u"], f_at_coords, reduction=self.reduction,
                             p=self.p, div_weight=self.div_weight)
        if self.lambda_bc > 0.0 and bc_coords is not None:
            loss = loss + self.lambda_bc * self.boundary_term(model, f_grid, bc_coords, d["u"])
        return loss

