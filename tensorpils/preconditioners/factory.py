"""Factory for preconditioners: pick multigrid or the spectral (blend/power) family."""

from typing import Optional

from .base import Preconditioner
from .multigrid import GeometricMultigrid
from .algebraic import AMGXPreconditioner
from .spectral import SpectralPreconditioner, SineSpectralPreconditioner

__all__ = ["build_preconditioner"]


def build_preconditioner(kind: str, problem, grid_size, *,
                         mg_levels: int = 4, mg_pre_smooth: int = 2,
                         mg_post_smooth: int = 2, mg_omega: float = 2.0 / 3.0,
                         mg_a2: float = 1.0, mg_c: float = 0.0,
                         strength: float = 1.0, method: str = "dense",
                         amg_sweeps: Optional[int] = None, amg_algorithm: str = "CLASSICAL",
                         amg_smoother: str = "BLOCK_JACOBI", amg_relaxation: float = 0.8,
                         device: Optional[str] = None) -> Preconditioner:
    """Build a preconditioner.

    Parameters
    ----------
    kind : {"multigrid", "amg", "blend", "power"}
        ``"multigrid"`` (default in the CLI) builds the geometric-multigrid V-cycle;
        ``"amg"`` builds the algebraic (AmgX) V-cycle, which uses only the assembled matrix
        and therefore carries over to unstructured meshes; ``"blend"`` / ``"power"`` build the
        exact spectral preconditioner.
    problem : PoissonProblem
        Source of the stiffness ``A`` and boundary mask (spectral kinds).
    grid_size : tuple(int, int)
        ``(nx, ny)`` of the fine grid (multigrid kind).
    mg_* : multigrid V-cycle settings (ignored by spectral kinds). ``mg_a2``/``mg_c`` set the level
        operator ``a²A + cM``: the defaults ``(1, 0)`` give the Poisson stiffness ``A``; a positive
        ``mg_c`` builds the screened-Poisson operator for the Allen–Cahn preconditioned LS loss.
    amg_* : algebraic V-cycle settings (ignored by every other kind). ``amg_sweeps`` is a
        single number used for *both* pre- and post-smoothing: the loss differentiates through
        ``P``, and the vjp is only another forward apply while the cycle stays symmetric, which
        needs matching sweep counts. It defaults to ``mg_pre_smooth`` (which must then equal
        ``mg_post_smooth``) so an existing sweep script transfers unchanged.
    strength : float in [0, 1]
        ``t`` (blend) or ``s`` (power); ignored by multigrid.
    method : {"dense", "sine"}
        Realization for the ``blend`` / ``power`` spectral kinds: ``"dense"`` (default) is the
        eigendecomposition; ``"sine"`` is the fast DST equivalent for a uniform grid (needed at
        128²/256²). Ignored by multigrid.
    device : optional
        If given, move the preconditioner there.
    """
    if kind == "multigrid":
        nx, ny = grid_size
        precond = GeometricMultigrid(
            nx_fine=nx, ny_fine=ny, n_levels=mg_levels,
            pre_smooth=mg_pre_smooth, post_smooth=mg_post_smooth, omega=mg_omega,
            a2=mg_a2, c=mg_c,
        )
    elif kind == "amg":
        if amg_sweeps is None:
            if mg_pre_smooth != mg_post_smooth:
                raise ValueError(
                    f"kind='amg' needs equal pre/post smoothing (the loss differentiates "
                    f"through P and the backward pass reuses the forward cycle, which is only "
                    f"valid while P is symmetric), got {mg_pre_smooth}/{mg_post_smooth}. "
                    f"Pass amg_sweeps explicitly to override.")
            amg_sweeps = mg_pre_smooth
        # Same operator the geometric V-cycle inverts: a**2 A + c M (Poisson: a2=1, c=0).
        A = problem.A * mg_a2 if mg_a2 != 1.0 else problem.A
        if mg_c != 0.0:
            A = A + problem.M * mg_c
        precond = AMGXPreconditioner(
            A, problem.boundary_mask, sweeps=amg_sweeps, algorithm=amg_algorithm,
            smoother=amg_smoother, relaxation=amg_relaxation,
            device=(device or "cuda:0"))
        return precond                      # already built on its device; AmgX is CUDA-only
    elif kind in ("blend", "power"):
        if method == "sine":
            nx, ny = grid_size
            precond = SineSpectralPreconditioner(
                problem.boundary_mask, nx=nx, ny=ny, kind=kind, strength=strength)
        elif method == "dense":
            precond = SpectralPreconditioner.from_problem(problem, kind=kind, strength=strength)
        else:
            raise ValueError(f"method must be 'dense' or 'sine', got {method!r}")
    else:
        raise ValueError(f"Unknown preconditioner kind {kind!r} "
                         "(expected 'multigrid', 'amg', 'blend', or 'power')")

    if device is not None:
        precond = precond.to(device)
    return precond
