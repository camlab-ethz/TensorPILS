r"""DeepONet — the second architecture, and the carrier for the PI-DeepONet baseline.

``neuralop`` ships no DeepONet (its zoo is fno/uno/gino/sfno/codano/…), so this is a direct
implementation of the standard unstacked form

.. math::
    \mathcal G_\theta(f)(y) \;=\; \sum_{k=1}^{p} b_k(f)\, t_k(y) \;+\; b_0 ,

with an MLP **branch** over the input function sampled on the sensor grid and an MLP
**trunk** over the query coordinate :math:`y=(x,y)`.

Two entry points, and the split is the whole point:

``forward(f_grid) -> [B, 1, H, W]``
    Queries the trunk at the grid nodes and returns a grid. This makes the model a **drop-in
    replacement for** :class:`~tensorpils.models.FNOModel`, so supervised training and every
    FEM loss in ``losses.py`` run through the unmodified
    :class:`~tensorpils.trainer.PoissonTrainer`. That is what lets us fill the "our loss, other
    architecture" cell for free.

``forward_at(f_grid, coords) -> [B, Q]``
    Evaluates at arbitrary coordinates. Because the trunk consumes coordinates directly, the
    PDE residual is available by autodiff — which is exactly how PI-DeepONet is trained, and
    what :mod:`.pi_deeponet` uses.

Two knobs exist so that a poor result is attributable to the *loss* rather than to a
handicapped baseline:

* ``trunk_fourier`` — random Fourier features on the trunk input. Plain MLPs have a strong
  spectral bias, and these datasets are multi-frequency (``K=4`` means modes up to 4x4), so
  without this a DeepONet underfits for reasons that have nothing to do with the objective.
  This is standard practice in the PI-DeepONet literature.
* ``f_scale`` — the branch sees ``f / f_scale``. Mirrors ``StokesDataset.f_scale`` already in
  this repo: an MLP on a raw ``O(10^2)`` input trains badly, while the FNO's lifting layer
  absorbs the scale on its own.
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

__all__ = ["DeepONetModel"]


def _mlp(sizes: Sequence[int], activation=nn.Tanh) -> nn.Sequential:
    """Plain MLP over ``sizes``; activation on every layer but the last."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class _FourierFeatures(nn.Module):
    r"""``y -> [sin(2*pi*B y), cos(2*pi*B y)]`` with a **fixed** random ``B``.

    Registered as a buffer, not a parameter: the features are a fixed embedding (Tancik et
    al.), and they travel with the checkpoint so a reloaded model reproduces its training
    embedding exactly.
    """

    def __init__(self, in_dim: int, n_features: int, scale: float = 4.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(in_dim, n_features, generator=g) * scale)
        self.out_dim = 2 * n_features

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * (y @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class DeepONetModel(nn.Module):
    r"""Unstacked DeepONet with a grid-valued ``forward`` (drop-in for ``FNOModel``).

    Parameters
    ----------
    grid_size : (nx, ny)
        Sensor grid and the grid ``forward`` returns. Node layout follows the repo
        convention: axis ``-2`` is ``y`` (``H = ny``), axis ``-1`` is ``x`` (``W = nx``).
    in_channels : int
        Channels of the input function (1 for Poisson ``f`` and for the Allen–Cahn seed
        frame). The branch consumes ``in_channels * nx * ny`` sensor values.
    p : int
        Number of basis functions (the branch/trunk latent width).
    width, depth : int
        Hidden width and number of Linear layers in each of branch and trunk.
    trunk_fourier : int
        If > 0, use that many random Fourier features on the trunk input.
    f_scale : float
        Branch input is divided by this.
    mollify : bool
        Hard zero-Dirichlet BC by multiplying every query by ``sin(pi x) sin(pi y)`` — the
        reference PI-DeepONet / PINO choice. Acts on ``forward_at`` and hence on ``forward``.
    out_channels : int
        Output fields. ``1`` for Poisson / Allen–Cahn; ``3`` for Stokes, ``(u_x, u_y, p)``.
    bc_channels : sequence of int, optional
        Channels the hard BC (``mollify`` / ``zero_boundary``) acts on; all by default, the
        velocity pair ``(0, 1)`` for Stokes.
    zero_boundary : bool
        The alternative hard BC, and the counterpart of PINO's ``ZeroBoundaryModel``: the
        **grid** output's boundary ring is set to exactly zero — the same operation
        ``losses.py`` applies to our own arms. It acts on ``forward`` only; a coordinate query
        (``forward_at``) returns the raw network value even on the boundary, so a boundary
        penalty can still see, and drive down, the boundary values. That is what makes it
        usable for the PI-DeepONet residual, whose autodiff Laplacian at interior points is
        blind to the boundary values and to any mask on them.
    """

    def __init__(self, grid_size, in_channels: int = 1, p: int = 128,
                 width: int = 256, depth: int = 4, trunk_fourier: int = 64,
                 fourier_scale: float = 4.0, f_scale: float = 1.0,
                 mollify: bool = False, zero_boundary: bool = False,
                 out_channels: int = 1, bc_channels: Optional[Sequence[int]] = None,
                 seed: int = 0, coords: Optional[torch.Tensor] = None):
        super().__init__()
        nx, ny = grid_size
        self.grid_size = None if coords is not None else (nx, ny)
        # An UNSTRUCTURED sensor set. The branch is an MLP over flattened sensor values and never
        # knew they were on a grid, so the only things that change are how many there are and
        # where the default query points sit. The trunk was always coordinate-based, which is why
        # a PI-DeepONet -- unlike PINO's finite differences -- carries over to an arbitrary mesh
        # at all. The grid ``forward`` is withdrawn; ``forward_at`` is the whole interface here.
        self.coords_given = coords is not None
        self.in_channels = in_channels
        self.out_channels = int(out_channels)
        self.p = p
        self.f_scale = float(f_scale)
        # Which output channels the hard BC (mollifier / boundary zeroing) applies to. All of
        # them by default; for Stokes the CLI passes (0, 1) -- the two velocity components --
        # because the pressure has no Dirichlet condition (only a gauge, fixed by projection).
        self.bc_channels = (tuple(range(self.out_channels)) if bc_channels is None
                            else tuple(int(c) for c in bc_channels))
        # Hard zero-Dirichlet BC by multiplying with sin(pi x) sin(pi y). Unlike the grid-based
        # MollifiedModel used for PINO this must be evaluated at the *query* coordinates, so it
        # lives inside forward_at — which also means the autodiff Laplacian differentiates the
        # mollified field, as it must.
        self.mollify = mollify
        if mollify and zero_boundary:
            raise ValueError("mollify and zero_boundary are alternative hard-BC mechanisms; pick one")
        self.zero_boundary = zero_boundary
        if zero_boundary:
            if coords is not None:
                raise ValueError(
                    "zero_boundary masks a grid's outer ring and has no unstructured form; on a "
                    "mesh the BC is the loss's boundary penalty (--pideeponet_bc zero) evaluated "
                    "at the mesh's own boundary nodes.")
            # Registered only when used, so checkpoints of mollified / plain models keep their
            # exact state_dict keys.
            mask = torch.ones(ny, nx)
            mask[0, :] = mask[-1, :] = 0.0
            mask[:, 0] = mask[:, -1] = 0.0
            self.register_buffer("boundary_mask", mask)

        # Multi-output: branch and trunk each emit C*p values, read as C independent groups of
        # p basis functions that share the hidden layers -- the standard "multiple-output
        # DeepONet" split, and for C=1 exactly the original network.
        C = self.out_channels
        n_points = coords.shape[0] if coords is not None else nx * ny
        n_sensors = in_channels * n_points
        self.branch = _mlp([n_sensors] + [width] * (depth - 1) + [C * p])

        if trunk_fourier > 0:
            self.embed = _FourierFeatures(2, trunk_fourier, scale=fourier_scale, seed=seed)
            trunk_in = self.embed.out_dim
        else:
            self.embed = None
            trunk_in = 2
        self.trunk = _mlp([trunk_in] + [width] * (depth - 1) + [C * p])
        self.bias = nn.Parameter(torch.zeros(C))

        # Default query coordinates: the mesh's own points when one was given, otherwise the
        # grid in the repo's row-major node order (row i -> y, col j -> x), so reshaping
        # [B, ny*nx] -> [B, ny, nx] lands each value on its own node.
        if coords is not None:
            self.register_buffer("grid_coords", torch.as_tensor(coords, dtype=torch.float32))
        else:
            xs = torch.linspace(0.0, 1.0, nx)
            ys = torch.linspace(0.0, 1.0, ny)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            self.register_buffer("grid_coords",
                                 torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1))

    # ------------------------------------------------------------------ core
    def _branch(self, f_grid: torch.Tensor) -> torch.Tensor:
        """Sensor values -> ``[B, C*p]``.

        Grid sensors come as ``[B, C, H, W]`` (or ``[C, H, W]`` unbatched); mesh sensors as
        ``[B, N, C]``. A 3-D tensor is ambiguous between the two, so the sensor set the model
        was built on decides — anything else silently flattens the batch into the branch input.
        """
        if not self.coords_given and f_grid.dim() == 3:
            f_grid = f_grid.unsqueeze(0)                    # [C, H, W] -> [1, C, H, W]
        return self.branch(f_grid.reshape(f_grid.shape[0], -1) / self.f_scale)

    def _trunk(self, coords: torch.Tensor) -> torch.Tensor:
        """``[..., 2]`` -> ``[..., p]``."""
        return self.trunk(self.embed(coords) if self.embed is not None else coords)

    def forward_at(self, f_grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        r"""Evaluate at arbitrary coordinates.

        ``coords`` is ``[Q, 2]`` (shared by the batch) or ``[B, Q, 2]`` (per-sample).
        Returns ``[B, Q]`` for a single output channel, ``[B, Q, C]`` otherwise.

        The per-sample form is what the autodiff residual needs: ``u[b, q]`` then depends
        only on ``coords[b, q]``, so a single ``grad`` of the summed output recovers the
        per-sample derivatives rather than their sum over the batch.
        """
        C, p = self.out_channels, self.p
        b = self._branch(f_grid).reshape(-1, C, p)                 # [B, C, p]
        t = self._trunk(coords)                                    # [Q, C*p] or [B, Q, C*p]
        t = t.reshape(*t.shape[:-1], C, p)                         # [Q, C, p] or [B, Q, C, p]
        # 1/sqrt(p): the dot product of two p-dimensional vectors grows like sqrt(p), so
        # without this the output magnitude at initialization depends on p (measured 5.5x the
        # target at p=32 and 12.4x at p=128). Starting an operator O(10x) too large costs the
        # baseline real epochs, which would show up as a loss effect that it is not.
        scale = p ** -0.5
        if t.dim() == 3:
            u = scale * torch.einsum("bcp,qcp->bqc", b, t) + self.bias        # [B, Q, C]
        else:
            u = scale * (b.unsqueeze(1) * t).sum(dim=-1) + self.bias         # [B, Q, C]
        if self.mollify:
            x, y = coords[..., 0], coords[..., 1]
            m = torch.sin(math.pi * x) * torch.sin(math.pi * y)
            # sin(pi * 1.0) is 8.7e-8 in float32, not 0, so the "hard" BC would leak at the far
            # edges. Snap the domain boundary to exactly zero. This is a measure-zero set, so
            # the autodiff Laplacian at interior collocation points is unaffected.
            on_bdry = (x <= 0) | (x >= 1) | (y <= 0) | (y >= 1)
            m = torch.where(on_bdry, torch.zeros_like(m), m)
            if m.dim() == 1:                                       # shared coords: [Q] -> [1, Q]
                m = m.unsqueeze(0)
            keep = torch.ones(C, device=u.device, dtype=u.dtype)
            keep[list(self.bc_channels)] = 0.0
            u = u * (keep + (1.0 - keep) * m.unsqueeze(-1))        # bc channels * m, others * 1
        return u.squeeze(-1) if C == 1 else u

    def forward(self, f_grid: torch.Tensor) -> torch.Tensor:
        """``[B, C_in, H, W]`` -> ``[B, C_out, H, W]`` — the ``FNOModel``-compatible signature."""
        if self.grid_size is None:
            raise RuntimeError(
                "this DeepONet was built on an unstructured sensor set, which has no grid to "
                "reshape to; query it with forward_at(f, coords).")
        squeeze = (f_grid.dim() == 3)
        if squeeze:
            f_grid = f_grid.unsqueeze(0)
        nx, ny = self.grid_size
        C = self.out_channels
        u = self.forward_at(f_grid, self.grid_coords)               # [B, ny*nx] or [B, ny*nx, C]
        u = u.reshape(-1, ny, nx, C).permute(0, 3, 1, 2)            # [B, C, ny, nx]
        if self.zero_boundary:
            keep = torch.ones(C, device=u.device, dtype=u.dtype)
            keep[list(self.bc_channels)] = 0.0
            mask = keep[:, None, None] + (1.0 - keep)[:, None, None] * self.boundary_mask
            u = u * mask
        return u.squeeze(0) if squeeze else u
