"""Preconditioners for the discrete Poisson stiffness.

Public interface :class:`Preconditioner` (``precond(r) -> P r``), implemented by the
geometric-multigrid V-cycle and the exact spectral (blend / power) preconditioner.
Use :func:`build_preconditioner` to construct one from a config.
"""

from .base import Preconditioner
from .multigrid import GeometricMultigrid
from .spectral import SpectralPreconditioner
from .factory import build_preconditioner

__all__ = [
    "Preconditioner",
    "GeometricMultigrid",
    "SpectralPreconditioner",
    "build_preconditioner",
]
