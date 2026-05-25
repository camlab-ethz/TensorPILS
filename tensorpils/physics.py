"""FEM Poisson operator built on TensorMesh's assembled matrices.

:class:`PoissonProblem` replaces the standalone script's hand-rolled
``PoissonEquation``. It assembles the stiffness ``A`` and mass ``M`` matrices once
from a structured :class:`tensormesh.Mesh` (via TensorMesh's
``LaplaceElementAssembler`` / ``MassElementAssembler``) and exposes a batched,
autograd-differentiable interface used by the physics-informed losses:

* :meth:`load_vector`  — consistent load ``b = M f``
* :meth:`residual`     — boundary-masked Galerkin residual ``A u − b``
* :meth:`energy`       — Deep Ritz energy ``½ uᵀA u − uᵀ(M f)``

All public methods take/return node fields in ``[batch, n_nodes]`` (or ``[n_nodes]``)
convention; the ``[n_nodes, batch]`` transpose required by sparse mat–vec products is
confined to this module.
"""

import torch
import torch.nn as nn

from tensormesh import LaplaceElementAssembler, MassElementAssembler

__all__ = ["PoissonProblem", "apply_zero_boundary"]


def apply_zero_boundary(u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out boundary entries, keeping interior values. Supports ``[N]`` and ``[B, N]``."""
    out = torch.zeros_like(u)
    if u.dim() == 1:
        out[~mask] = u[~mask]
    else:
        out[:, ~mask] = u[:, ~mask]
    return out


class PoissonProblem(nn.Module):
    r"""Discrete FEM operator for :math:`-\Delta u = f` with zero Dirichlet BC.

    Parameters
    ----------
    mesh : tensormesh.Mesh
        The (structured) quad mesh. ``mesh.boundary_mask`` defines the Dirichlet nodes.
    quadrature_order : int, optional
        Gauss order per axis for assembling ``A`` and ``M``. Default ``4`` integrates
        the bilinear-quad stiffness and mass exactly on rectangular cells.

    Notes
    -----
    The energy uses the matrix identity
    :math:`E(u) = \tfrac12 u^\top A u - u^\top (M f)`, which is exactly the FE energy
    :math:`\int(\tfrac12|\nabla u_h|^2 - f_h u_h)` and whose gradient w.r.t. ``u`` is the
    residual :math:`A u - M f` — the property the Deep Ritz / preconditioned losses rely on.
    """

    def __init__(self, mesh, quadrature_order: int = 4):
        super().__init__()
        self.n_nodes = int(mesh.n_points)
        self.register_buffer("boundary_mask", mesh.boundary_mask.bool())
        # Assemble once; matrices inherit the mesh dtype/device and are cast at use.
        self.A = LaplaceElementAssembler.from_mesh(mesh, quadrature_order=quadrature_order)(mesh.points)
        self.M = MassElementAssembler.from_mesh(mesh, quadrature_order=quadrature_order)(mesh.points)

    # -- device/dtype management: move the (plain-attribute) sparse matrices too --
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.A = self.A.to(*args, **kwargs)
        self.M = self.M.to(*args, **kwargs)
        return self

    @staticmethod
    def _spmm(mat, x: torch.Tensor) -> torch.Tensor:
        """Differentiable sparse mat–vec / mat–mat. ``x``: ``[N]`` or ``[B, N]``."""
        if mat.dtype != x.dtype or mat.device != x.device:
            mat = mat.to(dtype=x.dtype, device=x.device)
        if x.dim() == 1:
            return mat @ x                 # [N]
        return (mat @ x.T).T               # [B, N]

    # ---------------------------------------------------------------- load vector
    def load_vector(self, f: torch.Tensor) -> torch.Tensor:
        r"""Consistent load :math:`b = M f`. ``f``: ``[N]`` or ``[B, N]``; returns same shape."""
        return self._spmm(self.M, f)

    # ------------------------------------------------------------------ residual
    def residual(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        r"""Boundary-masked Galerkin residual :math:`R = A\,u_{\mathrm{bc}} - M f`.

        ``u`` is first projected to zero on the Dirichlet boundary; the returned
        residual is also zeroed there. Fully differentiable in ``u``.
        """
        u_bc = apply_zero_boundary(u, self.boundary_mask)
        r = self._spmm(self.A, u_bc) - self.load_vector(f)
        return apply_zero_boundary(r, self.boundary_mask)

    # -------------------------------------------------------------------- energy
    def energy(self, u: torch.Tensor, f: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
        r"""Deep Ritz energy :math:`E(u) = \tfrac12 u^\top A u - u^\top (M f)`.

        ``u``, ``f``: ``[N]`` or ``[B, N]``. Returns a scalar (``reduce="mean"``/``"sum"``)
        or the per-sample energy ``[B]`` (``reduce="none"``). Apply any boundary
        projection to ``u`` *before* calling for hard-BC behaviour.
        """
        Au = self._spmm(self.A, u)
        b = self.load_vector(f)
        if u.dim() == 1:
            return 0.5 * (u * Au).sum() - (u * b).sum()
        e = 0.5 * (u * Au).sum(dim=-1) - (u * b).sum(dim=-1)   # [B]
        if reduce == "mean":
            return e.mean()
        if reduce == "sum":
            return e.sum()
        return e
