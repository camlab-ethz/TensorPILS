"""Preconditioners for the discrete Poisson stiffness.

Public interface :class:`Preconditioner` (``precond(r) -> P r``), implemented by the
geometric-multigrid V-cycle and the exact spectral (blend / power) preconditioner.
Use :func:`build_preconditioner` to construct one from a config.

:class:`StokesBlockPreconditioner` also implements the interface but is built directly
from a :class:`~tensorpils.physics.StokesProblem` (it is a *composite* over two grids, not
a single-operator kind), and it is applied as a **norm weight** ``½ rᵀPr`` rather than as
``½‖Pr‖²`` — see its module docstring for why that distinction is essential for a saddle
point system.
"""

from .base import Preconditioner
from .multigrid import GeometricMultigrid
from .spectral import SpectralPreconditioner, SineSpectralPreconditioner
from .stokes import StokesBlockPreconditioner, StokesBlendPreconditioner
from .stokes_monolithic import StokesMonolithicMultigrid
from .factory import build_preconditioner

__all__ = [
    "Preconditioner",
    "GeometricMultigrid",
    "SpectralPreconditioner",
    "SineSpectralPreconditioner",
    "StokesBlockPreconditioner",
    "StokesBlendPreconditioner",
    "StokesMonolithicMultigrid",
    "build_preconditioner",
]
