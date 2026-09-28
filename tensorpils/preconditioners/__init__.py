"""Preconditioners ``P`` for the preconditioned least-squares losses.

Public interface :class:`Preconditioner` (``precond(r) -> P r``), implemented by the
geometric-multigrid V-cycle, the *algebraic* multigrid V-cycle (AmgX --- same contract, but
built from the matrix rather than the grid, so it also serves unstructured meshes), and the
exact spectral blend ``(1-t) I + t A^{-1}``. Use :func:`build_preconditioner` to construct one
of these from a config.

:class:`StokesBlockPreconditioner` also implements the interface but is built directly
from a :class:`~tensorpils.physics.StokesProblem` (it is a *composite* over the velocity and
pressure blocks, not a single-operator kind), and it is applied as a **norm weight**
``½ rᵀPr`` rather than as ``½‖Pr‖²`` — see its module docstring for why that distinction is
essential for a saddle point system.
"""

from .base import Preconditioner
from .multigrid import GeometricMultigrid
from .algebraic import AMGXPreconditioner
from .spectral import SpectralPreconditioner
from .stokes import StokesBlockPreconditioner
from .factory import build_preconditioner

__all__ = [
    "Preconditioner",
    "GeometricMultigrid",
    "AMGXPreconditioner",
    "SpectralPreconditioner",
    "StokesBlockPreconditioner",
    "build_preconditioner",
]
