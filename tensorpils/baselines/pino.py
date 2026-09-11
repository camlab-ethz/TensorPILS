r"""PINO baseline — strong-form PDE residual by finite differences on the grid.

Faithful to the reference implementation (``neuraloperator/physics_informed``). That repo
uses **two** differentiation schemes and the choice tracks the boundary condition, not
taste:

===================================  ===============  =====================
example                              BC               differentiation
===================================  ===============  =====================
Burgers (``FDM_Burgers``)            periodic in x    FFT
Navier–Stokes (``FDM_NS_vorticity``) periodic         FFT
**Darcy** (``FDM_Darcy``)            **Dirichlet**    **central differences**
===================================  ===============  =====================

Every PDE in this repo is homogeneous Dirichlet on the unit square, so **Darcy is the
analogue and finite differences is the faithful choice**. Three further details are copied
from ``FDM_Darcy`` / ``train_2d.py``, each of which *helps* the baseline — skipping them
would be strawmanning:

1. the residual is evaluated on **interior points only** (boundary rows/columns sliced off);
2. the zero Dirichlet BC is imposed **hard**, as a property of the *model* rather than of the
   loss, so it is active at eval too. The reference multiplies by a mollifier
   :math:`\sin(\pi x)\sin(\pi y)` (:class:`MollifiedModel`); **we default instead to zeroing
   the boundary nodes** (:class:`ZeroBoundaryModel`), which is exactly what ``losses.py`` does
   to our own arms via ``apply_zero_boundary``. The mollifier would hand the baseline a smooth
   decay profile matched to this dataset that our arms never see, so zeroing is the comparable
   choice; the mollifier remains available via ``--pino_bc mollifier``;
3. the reduction is the **relative** :math:`L^p` ratio :math:`\|\mathcal{L}u-f\|/\|f\|`
   averaged over the batch, not our sum-over-nodes convention. This changes the gradient
   scale by orders of magnitude, which is why each baseline arm gets its own learning-rate
   sweep.

The Laplacian itself comes from ``neuralop.losses.differentiation.FiniteDiff`` — the
reference library's own utility (second-order central in the interior, pure ``torch`` ops
so it is differentiable). Using it rather than a reimplementation is deliberate: it removes
"you coded the baseline wrong" as a reading of any result.
"""

import math

import torch
import torch.nn as nn

try:
    from neuralop.losses.differentiation import FiniteDiff
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "neuralop >= 2.0 is required for the PINO baseline "
        "(neuralop.losses.differentiation.FiniteDiff)."
    ) from e

__all__ = ["mollifier_grid", "MollifiedModel", "boundary_mask_grid", "ZeroBoundaryModel",
           "rel_lp", "stokes_reduce", "PINOPoissonLoss", "PINOACLoss", "PINOStokesLoss"]


# --------------------------------------------------------------------------- helpers

def mollifier_grid(nx: int, ny: int, device=None, dtype=torch.float32,
                   scale: float = 1.0) -> torch.Tensor:
    r"""``sin(pi x) sin(pi y)`` on the ``[ny, nx]`` grid over the unit square.

    Multiplying a prediction by this imposes ``u = 0`` on ``\partial\Omega`` exactly. Grid
    layout follows the repo convention: axis ``-2`` is ``y`` (``H = ny``), axis ``-1`` is
    ``x`` (``W = nx``).

    ``scale`` is PINO's output-scaling constant (their Darcy uses ``0.001`` because their
    solutions are ``O(1e-3)``); ours are ``O(1)``, so the default is ``1.0``.
    """
    x = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)     # [W]
    y = torch.linspace(0.0, 1.0, ny, device=device, dtype=dtype)     # [H]
    m = scale * torch.sin(math.pi * y)[:, None] * torch.sin(math.pi * x)[None, :]
    # sin(pi * 1.0) is 8.7e-8 in float32, not 0, because pi is not exactly representable — so
    # the "hard" BC would leak at the far edges. Zero the boundary explicitly; the eval-time
    # projection then genuinely has nothing left to do.
    m[0, :] = 0.0
    m[-1, :] = 0.0
    m[:, 0] = 0.0
    m[:, -1] = 0.0
    return m


class MollifiedModel(nn.Module):
    r"""Wrap a grid model so its output is multiplied by ``sin(pi x) sin(pi y)``.

    This is exactly ``pred = model(data) * mollifier`` from PINO's ``train_2d.py``, and it
    is the model's job rather than the loss's: were it applied only inside the loss, the
    trained network and the evaluated network would differ. Wrapping keeps train and eval
    consistent for free, and makes the eval-time boundary projection a no-op (the output is
    already exactly zero there).
    """

    def __init__(self, model: nn.Module, nx: int, ny: int, scale: float = 1.0):
        super().__init__()
        self.model = model
        self.scale = scale
        self.register_buffer("mollifier", mollifier_grid(nx, ny, scale=scale))
        # Keep the checkpoint-reload contract of models.FNOModel.
        self.build_config = getattr(model, "build_config", None)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.model(f) * self.mollifier


def boundary_mask_grid(nx: int, ny: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """``[ny, nx]`` of ones with a zeroed outer ring — the grid form of the FEM Dirichlet mask."""
    m = torch.ones(ny, nx, device=device, dtype=dtype)
    m[0, :] = 0.0
    m[-1, :] = 0.0
    m[:, 0] = 0.0
    m[:, -1] = 0.0
    return m


class ZeroBoundaryModel(nn.Module):
    r"""Wrap a grid model so its boundary nodes are set to **exactly zero**.

    The BC treatment our own arms use, applied to the baseline. ``losses.py`` builds every FEM
    residual from ``apply_zero_boundary(u_pred_node, boundary_mask)`` — the network's boundary
    values are discarded and replaced by zero — and this is the grid-space form of the same
    operation, so PINO and the ``galerkin``/``pls`` arms now impose the Dirichlet condition
    identically and differ only in the residual itself.

    **Why this replaces the mollifier as the default.** Multiplying by
    :math:`\sin(\pi x)\sin(\pi y)` also imposes the BC exactly, and it is what PINO's reference
    does, but it injects a smooth global decay profile into the model — and for the multi-frequency
    sine dataset used here that profile is close to the solution's own structure, which is
    information our arms do not get. Zeroing the boundary ring assumes nothing about the interior
    and generalises to any mesh, whereas a mollifier has to be hand-built per geometry.

    Expect this to *cost* PINO accuracy relative to the mollified version. That is the point: the
    two are then comparable. :class:`MollifiedModel` is kept for reproducing the reference.
    """

    def __init__(self, model: nn.Module, nx: int, ny: int):
        super().__init__()
        self.model = model
        self.register_buffer("mask", boundary_mask_grid(nx, ny))
        # Keep the checkpoint-reload contract of models.FNOModel.
        self.build_config = getattr(model, "build_config", None)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.model(f) * self.mask


def rel_lp(pred: torch.Tensor, target: torch.Tensor, p: int = 2,
           eps: float = 1e-12) -> torch.Tensor:
    r"""PINO's ``LpLoss.rel(..., size_average=True)``: ``mean_b ||pred-target||_p / ||target||_p``.

    Reimplemented here (five lines) rather than imported, because ``neuralop`` 2.0's
    ``LpLoss`` additionally applies quadrature weights and a different default reduction —
    faithful to *that* library, but no longer to the PINO paper's loss. Flattening is over
    everything but the batch axis, as in the original.
    """
    b = pred.shape[0]
    diff = torch.linalg.vector_norm(pred.reshape(b, -1) - target.reshape(b, -1), ord=p, dim=1)
    norm = torch.linalg.vector_norm(target.reshape(b, -1), ord=p, dim=1)
    return (diff / norm.clamp_min(eps)).mean()


def _fd(nx: int, ny: int, device=None) -> FiniteDiff:
    """A non-periodic 2-D ``FiniteDiff`` for the unit square.

    ``FiniteDiff`` names axis ``-2`` "x" and axis ``-1`` "y"; this repo names axis ``-2``
    ``y`` and axis ``-1`` ``x``. The spacings are therefore passed in *its* order,
    ``(h_axis-2, h_axis-1) = (hy, hx)``, which is what makes the Laplacian correct on a
    non-square grid too (on the square grids used everywhere here the two are equal).
    """
    hy = 1.0 / (ny - 1)
    hx = 1.0 / (nx - 1)
    return FiniteDiff(dim=2, h=(hy, hx), periodic_in_x=False, periodic_in_y=False)


# --------------------------------------------------------------------------- Poisson

class PINOPoissonLoss(nn.Module):
    r"""PINO residual for :math:`-\Delta u = f`, mirroring ``darcy_loss``/``FDM_Darcy``.

    ``forward(u_pred_grid, f_grid)`` with ``u_pred_grid`` ``[B, H, W]`` and ``f_grid``
    ``[B, 1, H, W]`` or ``[B, H, W]``. The Laplacian is taken with central differences and
    the residual is reduced over **interior** points only, exactly as the reference does.

    ``reduction='rel'`` (default) is PINO's relative-``L²`` ratio ``‖−Δu − f‖/‖f‖``;
    ``'mse'`` is the plain mean square, provided so the loss can be put on the same footing
    as our ``½‖·‖²`` losses when that is the question being asked.
    """

    def __init__(self, grid_size, reduction: str = "rel", p: int = 2):
        super().__init__()
        if reduction not in ("rel", "mse"):
            raise ValueError(f"reduction must be 'rel' or 'mse', got {reduction!r}")
        nx, ny = grid_size
        self.grid_size = (nx, ny)
        self.reduction = reduction
        self.p = p
        self.fd = _fd(nx, ny)

    def forward(self, u_pred_grid: torch.Tensor, f_grid: torch.Tensor) -> torch.Tensor:
        if f_grid.dim() == 4:
            f_grid = f_grid.squeeze(1)                                   # [B, H, W]
        lap = self.fd.laplacian(u_pred_grid)                             # [B, H, W]
        lhs = -lap[..., 1:-1, 1:-1]
        rhs = f_grid[..., 1:-1, 1:-1]
        if self.reduction == "rel":
            return rel_lp(lhs, rhs, p=self.p)
        return ((lhs - rhs) ** 2).mean()


# ----------------------------------------------------------------------- Allen–Cahn

class PINOACLoss(nn.Module):
    r"""PINO residual for the Allen–Cahn step, over an autoregressive rollout.

    The strong form of the *same* time discretisation :class:`~tensorpils.physics.ACProblem`
    uses, so the only thing under test is weak-vs-strong. Where the FEM residual is

    .. math::
        M\frac{u^{n+1}-u^n}{\tau} + a^2 A u^{n+1} - M\,\rho ,

    this drops the mass matrix and replaces :math:`A \leftrightarrow -\Delta_h`:

    .. math::
        \frac{u^{n+1}-u^n}{\tau} - a^2 \Delta_h u^{n+1} - \rho ,
        \qquad
        \rho = \begin{cases}
          \epsilon^2\,(u^n - (u^{n+1})^3) & \text{convex\_concave} \\
          \epsilon^2\,(u^{n+1} - (u^{n+1})^3) & \text{backward\_euler}
        \end{cases}

    ``forward(seq_grid)`` takes ``[B, 1+R, H, W]`` — the seed frame followed by the ``R``
    predicted frames — and sums the interior residual over consecutive pairs, ``discount``-
    weighted like :class:`~tensorpils.losses.ACGalerkinLoss`.

    Reduction here is ``mse``: the Allen–Cahn step residual has no right-hand side to
    normalise against (PINO's own relative form needs a nonzero ``f``), and the plain mean
    square is the direct analogue of our ``½‖R‖²``.

    ``detach_coupling`` mirrors :class:`~tensorpils.losses.ACGalerkinLoss`: it detaches the
    previous frame ``u^k`` inside the residual. The trainer sets it from ``--bptt_mode`` exactly
    as the FEM arm does, so the two differ only in the residual and not in what the gradient
    flows through.
    """

    def __init__(self, grid_size, a: float, eps: float, dt: float,
                 discount: float = 1.0, integrator: str = "convex_concave",
                 detach_coupling=None):
        super().__init__()
        if integrator not in ("backward_euler", "convex_concave"):
            raise ValueError(f"unknown integrator {integrator!r}")
        nx, ny = grid_size
        self.grid_size = (nx, ny)
        self.a, self.eps, self.dt = a, eps, dt
        self.discount = discount
        self.integrator = integrator
        self.detach_coupling = False if detach_coupling is None else detach_coupling
        self.fd = _fd(nx, ny)

    def forward(self, seq_grid: torch.Tensor) -> torch.Tensor:
        a2 = self.a * self.a
        e2 = self.eps * self.eps
        total = seq_grid.new_zeros(())
        n_pairs = seq_grid.shape[1] - 1
        for k in range(n_pairs):
            uc = seq_grid[:, k].detach() if self.detach_coupling else seq_grid[:, k]
            un = seq_grid[:, k + 1]
            reaction = e2 * ((uc if self.integrator == "convex_concave" else un) - un ** 3)
            r = (un - uc) / self.dt - a2 * self.fd.laplacian(un) - reaction
            r = r[..., 1:-1, 1:-1]
            total = total + (self.discount ** k) * (r ** 2).mean()
        return total


# --------------------------------------------------------------------------- Stokes

def _ddx(fd: FiniteDiff, u: torch.Tensor) -> torch.Tensor:
    """d/dx in the repo's convention (axis ``-1``) — ``FiniteDiff`` calls that axis ``y``."""
    return fd.dy(u)


def _ddy(fd: FiniteDiff, u: torch.Tensor) -> torch.Tensor:
    """d/dy in the repo's convention (axis ``-2``) — ``FiniteDiff`` calls that axis ``x``."""
    return fd.dx(u)


def stokes_reduce(r_mom: torch.Tensor, r_div: torch.Tensor, f: torch.Tensor,
                  reduction: str = "rel", p: int = 2, div_weight: float = 1.0) -> torch.Tensor:
    r"""Reduce a strong-form Stokes residual, shared by the PINO and PI-DeepONet arms.

    ``r_mom`` ``[B, ..., 2]`` is the momentum residual :math:`-\mu\Delta u + \nabla p - f`,
    ``r_div`` ``[B, ...]`` the continuity residual :math:`\nabla\cdot u`, ``f`` ``[B, ..., 2]``
    the body force at the same points (any trailing layout; everything is flattened per sample).

    ``rel``: PINO's relative ratio on the **stacked** system, ``mean_b ‖(r_mom, w·r_div)‖ /
    ‖(f, 0)‖`` — the strong-form analogue of the FEM negative control ``½‖Kc − b‖²`` with
    ``b = (M_u f, 0)``, in which momentum and continuity rows are likewise stacked with their
    natural units. ``mse``: ``mean(r_mom²) + w² mean(r_div²)``. ``div_weight`` (``w``) is the
    one knob a strong-form saddle point needs beyond Poisson: the two equations have different
    units (``div u ~ k u`` against ``f ~ mu k² u``), and no PINO reference example fixes the
    ratio, so it is exposed and tuned on validation like a learning rate.
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


class PINOStokesLoss(nn.Module):
    r"""PINO residual for the stationary Stokes equations on the velocity grid.

    Strong form of the same system :class:`~tensorpils.physics.StokesProblem` assembles,

    .. math::
        -\mu\Delta u + \nabla p = f,\qquad \nabla\cdot u = 0,

    with every derivative taken by central differences on the fine (Q2-node) grid and the
    residual reduced over **interior** nodes only, as ``FDM_Darcy`` does. There is no PINO
    reference implementation for a saddle point, so the port is the Darcy recipe applied to
    each equation; the only genuinely new ingredient is the continuity weight of
    :func:`stokes_reduce`.

    ``forward(out_grid, f_grid)`` with ``out_grid`` ``[B, 3, H, W]`` the **physical** fields
    ``(u_x, u_y, p)`` — velocity already zero on the boundary ring (the trainer projects it,
    exactly as it does for our own arms) and pressure on the **full** fine grid (the FNO's
    third channel times ``p_scale``; the FEM arms read it on the ``[::2, ::2]`` Q1 subgrid, a
    strong-form residual wants it everywhere) — and ``f_grid`` ``[B, 2, H, W]`` the body force.
    The pressure gauge is irrelevant here: only ``∇p`` enters.
    """

    def __init__(self, grid_size, mu: float, reduction: str = "rel", p: int = 2,
                 div_weight: float = 1.0):
        super().__init__()
        if reduction not in ("rel", "mse"):
            raise ValueError(f"reduction must be 'rel' or 'mse', got {reduction!r}")
        nx, ny = grid_size
        self.grid_size = (nx, ny)
        self.mu = float(mu)
        self.reduction = reduction
        self.p = p
        self.div_weight = float(div_weight)
        self.fd = _fd(nx, ny)

    def residual(self, out_grid: torch.Tensor):
        """``(r_mom [B, H-2, W-2, 2], r_div [B, H-2, W-2])`` on the interior nodes, before
        subtracting ``f`` (so ``r_mom`` here is ``-mu lap u + grad p``)."""
        ux, uy, pr = out_grid[:, 0], out_grid[:, 1], out_grid[:, 2]
        fd = self.fd
        mom = torch.stack([-self.mu * fd.laplacian(ux) + _ddx(fd, pr),
                           -self.mu * fd.laplacian(uy) + _ddy(fd, pr)], dim=-1)
        div = _ddx(fd, ux) + _ddy(fd, uy)
        return mom[:, 1:-1, 1:-1], div[:, 1:-1, 1:-1]

    def forward(self, out_grid: torch.Tensor, f_grid: torch.Tensor) -> torch.Tensor:
        mom, div = self.residual(out_grid)
        f_int = f_grid.permute(0, 2, 3, 1)[:, 1:-1, 1:-1]                 # [B, H-2, W-2, 2]
        return stokes_reduce(mom - f_int, div, f_int, reduction=self.reduction, p=self.p,
                             div_weight=self.div_weight)

