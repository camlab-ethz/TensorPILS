"""Factory for preconditioners: pick multigrid or the spectral (blend/power) family."""

from typing import Optional

from .base import Preconditioner
from .multigrid import GeometricMultigrid
from .spectral import SpectralPreconditioner

__all__ = ["build_preconditioner"]


def build_preconditioner(kind: str, problem, grid_size, *,
                         mg_levels: int = 4, mg_pre_smooth: int = 2,
                         mg_post_smooth: int = 2, mg_omega: float = 2.0 / 3.0,
                         strength: float = 1.0,
                         device: Optional[str] = None) -> Preconditioner:
    """Build a preconditioner.

    Parameters
    ----------
    kind : {"multigrid", "blend", "power"}
        ``"multigrid"`` (default in the CLI) builds the geometric-multigrid V-cycle;
        ``"blend"`` / ``"power"`` build the exact spectral preconditioner.
    problem : PoissonProblem
        Source of the stiffness ``A`` and boundary mask (spectral kinds).
    grid_size : tuple(int, int)
        ``(nx, ny)`` of the fine grid (multigrid kind).
    mg_* : multigrid V-cycle settings (ignored by spectral kinds).
    strength : float in [0, 1]
        ``t`` (blend) or ``s`` (power); ignored by multigrid.
    device : optional
        If given, move the preconditioner there.
    """
    if kind == "multigrid":
        nx, ny = grid_size
        precond = GeometricMultigrid(
            nx_fine=nx, ny_fine=ny, n_levels=mg_levels,
            pre_smooth=mg_pre_smooth, post_smooth=mg_post_smooth, omega=mg_omega,
        )
    elif kind in ("blend", "power"):
        precond = SpectralPreconditioner.from_problem(problem, kind=kind, strength=strength)
    else:
        raise ValueError(f"Unknown preconditioner kind {kind!r} "
                         "(expected 'multigrid', 'blend', or 'power')")

    if device is not None:
        precond = precond.to(device)
    return precond
