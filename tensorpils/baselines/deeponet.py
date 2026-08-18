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
    """

    def __init__(self, grid_size, in_channels: int = 1, p: int = 128,
                 width: int = 256, depth: int = 4, trunk_fourier: int = 64,
                 fourier_scale: float = 4.0, f_scale: float = 1.0,
                 mollify: bool = False, out_channels: int = 1, seed: int = 0):
        super().__init__()
        if out_channels != 1:
            raise NotImplementedError("DeepONetModel currently emits a single channel")
        nx, ny = grid_size
        self.grid_size = (nx, ny)
        self.in_channels = in_channels
        self.p = p
        self.f_scale = float(f_scale)
        # Hard zero-Dirichlet BC by multiplying with sin(pi x) sin(pi y). Unlike the grid-based
        # MollifiedModel used for PINO this must be evaluated at the *query* coordinates, so it
        # lives inside forward_at — which also means the autodiff Laplacian differentiates the
        # mollified field, as it must.
        self.mollify = mollify

        n_sensors = in_channels * nx * ny
        self.branch = _mlp([n_sensors] + [width] * (depth - 1) + [p])

        if trunk_fourier > 0:
            self.embed = _FourierFeatures(2, trunk_fourier, scale=fourier_scale, seed=seed)
            trunk_in = self.embed.out_dim
        else:
            self.embed = None
            trunk_in = 2
        self.trunk = _mlp([trunk_in] + [width] * (depth - 1) + [p])
        self.bias = nn.Parameter(torch.zeros(1))

        # Grid query coordinates, in the repo's row-major node order (row i -> y, col j -> x),
        # so reshaping [B, ny*nx] -> [B, ny, nx] lands each value on its own node.
        xs = torch.linspace(0.0, 1.0, nx)
        ys = torch.linspace(0.0, 1.0, ny)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("grid_coords", torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1))

    # ------------------------------------------------------------------ core
    def _branch(self, f_grid: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` (or ``[C, H, W]``) -> ``[B, p]``."""
        if f_grid.dim() == 3:
            f_grid = f_grid.unsqueeze(0)
        return self.branch(f_grid.reshape(f_grid.shape[0], -1) / self.f_scale)

    def _trunk(self, coords: torch.Tensor) -> torch.Tensor:
        """``[..., 2]`` -> ``[..., p]``."""
        return self.trunk(self.embed(coords) if self.embed is not None else coords)

    def forward_at(self, f_grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        r"""Evaluate at arbitrary coordinates.

        ``coords`` is ``[Q, 2]`` (shared by the batch) or ``[B, Q, 2]`` (per-sample).
        Returns ``[B, Q]``.

        The per-sample form is what the autodiff residual needs: ``u[b, q]`` then depends
        only on ``coords[b, q]``, so a single ``grad`` of the summed output recovers the
        per-sample derivatives rather than their sum over the batch.
        """
        b = self._branch(f_grid)                                   # [B, p]
        t = self._trunk(coords)                                    # [Q, p] or [B, Q, p]
        # 1/sqrt(p): the dot product of two p-dimensional vectors grows like sqrt(p), so
        # without this the output magnitude at initialization depends on p (measured 5.5x the
        # target at p=32 and 12.4x at p=128). Starting an operator O(10x) too large costs the
        # baseline real epochs, which would show up as a loss effect that it is not.
        scale = self.p ** -0.5
        if t.dim() == 2:
            u = scale * (b @ t.transpose(0, 1)) + self.bias        # [B, Q]
        else:
            u = scale * (b.unsqueeze(1) * t).sum(dim=-1) + self.bias   # [B, Q]
        if self.mollify:
            x, y = coords[..., 0], coords[..., 1]
            m = torch.sin(math.pi * x) * torch.sin(math.pi * y)
            # sin(pi * 1.0) is 8.7e-8 in float32, not 0, so the "hard" BC would leak at the far
            # edges. Snap the domain boundary to exactly zero. This is a measure-zero set, so
            # the autodiff Laplacian at interior collocation points is unaffected.
            on_bdry = (x <= 0) | (x >= 1) | (y <= 0) | (y >= 1)
            u = u * torch.where(on_bdry, torch.zeros_like(m), m)
        return u

    def forward(self, f_grid: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` -> ``[B, 1, H, W]`` — the ``FNOModel``-compatible signature."""
        squeeze = (f_grid.dim() == 3)
        if squeeze:
            f_grid = f_grid.unsqueeze(0)
        nx, ny = self.grid_size
        u = self.forward_at(f_grid, self.grid_coords)               # [B, ny*nx]
        u = u.reshape(-1, 1, ny, nx)
        return u.squeeze(0) if squeeze else u
