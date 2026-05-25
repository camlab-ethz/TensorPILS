"""Geometric multigrid V-cycle preconditioner for the structured-grid Poisson stiffness.

Provides :math:`P \approx A^{-1}` as one V-cycle, used by the preconditioned losses
(``pls`` and preconditioned Deep Ritz). The hierarchy coarsens the structured grid by
factors of two; the stiffness at every level is **re-discretized** with TensorMesh's
``LaplaceElementAssembler`` (the only change from the standalone script, which hand-rolled
the per-level stiffness). Smoother, prolongation/restriction, and the V-cycle recursion
are unchanged.

The V-cycle is a sequence of sparse mat–vecs and element-wise ops, all differentiable, so
autograd flows through it: for the symmetric ``M`` produced by symmetric Jacobi sweeps,
:math:`\partial(\tfrac12 r^\top M r)/\partial r = M r` is obtained exactly by backprop.
"""

import numpy as np
import torch
import torch.nn as nn

from tensormesh import LaplaceElementAssembler
from .meshing import structured_quad_mesh

__all__ = ["GeometricMultigrid"]


def _build_1d_linear_interp(n_fine: int, n_coarse: int) -> np.ndarray:
    """1D linear interpolation matrix ``P`` ``[n_fine, n_coarse]`` between ``linspace(0,1,·)`` grids."""
    xf = np.linspace(0.0, 1.0, n_fine)
    xc = np.linspace(0.0, 1.0, n_coarse)
    P = np.zeros((n_fine, n_coarse), dtype=np.float64)
    for i, x in enumerate(xf):
        if x <= xc[0]:
            P[i, 0] = 1.0
        elif x >= xc[-1]:
            P[i, -1] = 1.0
        else:
            j = int(np.searchsorted(xc, x, side="right") - 1)
            j = max(0, min(j, n_coarse - 2))
            t = (x - xc[j]) / (xc[j + 1] - xc[j])
            P[i, j] = 1.0 - t
            P[i, j + 1] = t
    return P


def _build_2d_prolongation(nx_f: int, ny_f: int, nx_c: int, ny_c: int) -> np.ndarray:
    """Bilinear prolongation for row-major ``(y, x)`` node order (``k = i_y*nx + j_x``)."""
    P_x = _build_1d_linear_interp(nx_f, nx_c)
    P_y = _build_1d_linear_interp(ny_f, ny_c)
    return np.kron(P_y, P_x)


class GeometricMultigrid(nn.Module):
    """Geometric multigrid V-cycle preconditioner :math:`M \\approx A^{-1}`.

    Each coarser level uses ``((nx+1)//2, (ny+1)//2)`` (floored at 3). Prolongation is
    bilinear interpolation respecting the row-major node order; restriction is ``R = Pᵀ``.
    The stiffness at every level is re-discretized via :func:`structured_quad_mesh` +
    ``LaplaceElementAssembler``, then its Dirichlet rows/cols are zeroed with a unit
    diagonal. Given a residual with zero boundary entries (as produced by
    :meth:`PoissonProblem.residual`), the V-cycle preserves a zero boundary.
    """

    def __init__(self, nx_fine: int, ny_fine: int, n_levels: int = 4,
                 pre_smooth: int = 2, post_smooth: int = 2,
                 omega: float = 2.0 / 3.0, ngp: int = 4):
        super().__init__()
        self.n_levels = n_levels
        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth
        self.omega = omega

        # ---- 1. dimensions at each level ----
        dims = [(nx_fine, ny_fine)]
        for _ in range(n_levels - 1):
            nx_p, ny_p = dims[-1]
            dims.append((max(3, (nx_p + 1) // 2), max(3, (ny_p + 1) // 2)))
        self.dims = dims

        # ---- 2. stiffness + diag at each level (with Dirichlet BC enforcement) ----
        for L, (nx, ny) in enumerate(dims):
            A = self._discretize(nx, ny, ngp)        # dense [N, N], double, BC-enforced
            A_t = torch.from_numpy(A).float()
            A_coo = A_t.to_sparse_coo().coalesce()
            self.register_buffer(f"A_indices_{L}", A_coo.indices())
            self.register_buffer(f"A_values_{L}", A_coo.values())
            self.register_buffer(f"diag_{L}", torch.from_numpy(np.diag(A).copy()).float())
            if L == n_levels - 1:
                self.register_buffer("A_inv_coarsest", torch.from_numpy(np.linalg.inv(A)).float())

        # ---- 3. prolongation P_L (coarse→fine) and R_L = P_L^T (fine→coarse) ----
        for L in range(n_levels - 1):
            nx_f, ny_f = dims[L]
            nx_c, ny_c = dims[L + 1]
            P_t = torch.from_numpy(_build_2d_prolongation(nx_f, ny_f, nx_c, ny_c)).float()
            P_coo = P_t.to_sparse_coo().coalesce()
            R_coo = P_t.T.contiguous().to_sparse_coo().coalesce()
            self.register_buffer(f"P_indices_{L}", P_coo.indices())
            self.register_buffer(f"P_values_{L}", P_coo.values())
            self.register_buffer(f"R_indices_{L}", R_coo.indices())
            self.register_buffer(f"R_values_{L}", R_coo.values())

    @staticmethod
    def _discretize(nx: int, ny: int, ngp: int) -> np.ndarray:
        """Assemble the Poisson stiffness on an ``nx×ny`` structured grid via TensorMesh,
        then zero Dirichlet rows/cols and set the boundary diagonal to 1. Returns dense double."""
        mesh = structured_quad_mesh(nx=nx, ny=ny)
        A = LaplaceElementAssembler.from_mesh(mesh, quadrature_order=ngp)(mesh.points)
        Ad = A.to_dense().double().cpu().numpy()
        mask = mesh.boundary_mask.cpu().numpy().astype(bool)
        Ad[mask, :] = 0.0
        Ad[:, mask] = 0.0
        Ad[mask, mask] = 1.0                          # numpy fancy diagonal assignment
        return Ad

    # -------------------- sparse tensor reconstruction --------------------
    def _size(self, L: int) -> int:
        nx, ny = self.dims[L]
        return nx * ny

    def _A_sparse(self, L: int):
        N = self._size(L)
        return torch.sparse_coo_tensor(getattr(self, f"A_indices_{L}"),
                                       getattr(self, f"A_values_{L}"), (N, N))

    def _P_sparse(self, L: int):
        return torch.sparse_coo_tensor(getattr(self, f"P_indices_{L}"),
                                       getattr(self, f"P_values_{L}"),
                                       (self._size(L), self._size(L + 1)))

    def _R_sparse(self, L: int):
        return torch.sparse_coo_tensor(getattr(self, f"R_indices_{L}"),
                                       getattr(self, f"R_values_{L}"),
                                       (self._size(L + 1), self._size(L)))

    # -------------------- V-cycle building blocks --------------------
    def _smooth(self, A_sp, diag, e, r, n_iter: int):
        """Weighted Jacobi: ``e ← e + ω D⁻¹ (r − A e)``. Batched ``[B, N]``."""
        for _ in range(n_iter):
            Ae = torch.sparse.mm(A_sp, e.T).T
            e = e + self.omega * (r - Ae) / diag.unsqueeze(0)
        return e

    def v_cycle(self, r: torch.Tensor) -> torch.Tensor:
        """One V-cycle: ``e ≈ A⁻¹ r``. ``r``: ``[B, N_fine]`` with zero-boundary entries."""
        return self._v_cycle_rec(0, r)

    def _v_cycle_rec(self, level: int, r: torch.Tensor) -> torch.Tensor:
        if level == self.n_levels - 1:
            return r @ self.A_inv_coarsest.T          # coarsest: direct solve per batch row

        A_sp = self._A_sparse(level)
        diag = getattr(self, f"diag_{level}")

        e = torch.zeros_like(r)
        e = self._smooth(A_sp, diag, e, r, self.pre_smooth)

        res = r - torch.sparse.mm(A_sp, e.T).T        # residual at this level
        res_c = torch.sparse.mm(self._R_sparse(level), res.T).T   # restrict
        e_c = self._v_cycle_rec(level + 1, res_c)     # recurse
        e = e + torch.sparse.mm(self._P_sparse(level), e_c.T).T   # prolong correction
        e = self._smooth(A_sp, diag, e, r, self.post_smooth)      # post-smooth
        return e
