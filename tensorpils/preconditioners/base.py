"""Preconditioner interface.

A preconditioner is an ``nn.Module`` that maps a residual ``r`` to ``P r``, an
approximation of ``A^{-1} r`` for the Poisson stiffness ``A``. Inputs and outputs are
node fields on the **full** grid, shape ``[B, N_full]`` or ``[N_full]``, with the
boundary entries carrying the zero-embedding convention (see the ``preconditioner_notes``
write-up, "Interior space versus full-grid storage").

Both :class:`~tensorpils.preconditioners.multigrid.GeometricMultigrid` and
:class:`~tensorpils.preconditioners.spectral.SpectralPreconditioner` implement this
contract, so the losses and trainer depend only on ``precond(r)`` and ``precond.report()``.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

__all__ = ["Preconditioner"]


class Preconditioner(nn.Module, ABC):
    """Maps a residual ``r`` to ``P r ≈ A^{-1} r`` (differentiable)."""

    @abstractmethod
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Apply the preconditioner. ``r``: ``[B, N_full]`` or ``[N_full]``; same shape out."""
        raise NotImplementedError

    def report(self) -> None:
        """Print a one-line diagnostic. Overridden by concrete preconditioners."""
