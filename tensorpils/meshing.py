"""Structured quadrilateral grid → :class:`tensormesh.Mesh`, plus grid↔node maps.

TensorMesh has no tensor-product (structured) mesh generator — ``Mesh.gen_rectangle``
goes through Gmsh and yields an *unstructured* mesh with arbitrary node numbering.
A Fourier Neural Operator, however, needs a regular grid whose node index maps to a
grid cell by a plain reshape. So TensorPILS keeps its own structured-grid builder, but
it emits a genuine :class:`tensormesh.Mesh` that TensorMesh's assemblers consume directly.

Node ordering is row-major: node ``k = i * nx + j`` sits at grid row ``i`` (the second
physical coordinate, "y") and column ``j`` (the first physical coordinate, "x").
:func:`node_to_grid` / :func:`grid_to_node` are exact inverses that move between the
flat node vector ``[..., nx*ny]`` and the image-shaped grid ``[..., ny, nx]``.
"""

import numpy as np
import meshio
import torch

from tensormesh import Mesh

__all__ = ["structured_quad_mesh", "structured_quad9_mesh", "node_to_grid", "grid_to_node"]


def structured_quad_mesh(nx: int = 64, ny: int = 64,
                         xlims=(0.0, 1.0), ylims=(0.0, 1.0),
                         dtype=np.float64) -> Mesh:
    """Build a structured bilinear-quad mesh on a rectangle as a ``tensormesh.Mesh``.

    Parameters
    ----------
    nx, ny : int
        Number of nodes along x (columns) and y (rows). The grid has ``nx*ny`` nodes
        and ``(nx-1)*(ny-1)`` quad cells.
    xlims, ylims : tuple(float, float)
        Domain extents along the first / second coordinate.
    dtype : numpy dtype
        Point coordinate precision (``float64`` by default; the assembled stiffness/mass
        matrices inherit it and are downcast at use).

    Returns
    -------
    tensormesh.Mesh
        Mesh with a single ``"quad"`` cell block, node ``k = i*nx + j`` at
        ``(x_j, y_i)``, and a boolean ``boundary_mask`` over the outer frame.

    Notes
    -----
    Connectivity is emitted in TensorMesh's **lexicographic** quad node order
    ``(0,0),(1,0),(0,1),(1,1)`` → ``[BL, BR, TL, TR]`` so that, with ``reorder=False``,
    each physical corner maps to the matching reference corner with a positive Jacobian.
    """
    xs = np.linspace(xlims[0], xlims[1], nx)          # along columns (j → x)
    ys = np.linspace(ylims[0], ylims[1], ny)          # along rows (i → y)

    # points: node k = i*nx + j at (xs[j], ys[i]); row-major over (i, j).
    X, Y = np.meshgrid(xs, ys)                        # both [ny, nx]; X[i,j]=xs[j], Y[i,j]=ys[i]
    points = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(dtype)   # [nx*ny, 2]

    # node-id grid: nids[i, j] = i*nx + j
    nids = np.arange(nx * ny, dtype=np.int64).reshape(ny, nx)

    # quad cells in lexicographic local order [BL, BR, TL, TR]:
    #   BL = (i,   j  ) = nids[:-1, :-1]
    #   BR = (i,   j+1) = nids[:-1, 1:]
    #   TL = (i+1, j  ) = nids[1:,  :-1]
    #   TR = (i+1, j+1) = nids[1:,  1:]
    quads = np.stack([nids[:-1, :-1], nids[:-1, 1:],
                      nids[1:, :-1],  nids[1:, 1:]], axis=-1).reshape(-1, 4)

    # boundary frame mask over the outer rows/columns
    bmask = np.zeros((ny, nx), dtype=bool)
    bmask[0, :] = bmask[-1, :] = bmask[:, 0] = bmask[:, -1] = True

    mio = meshio.Mesh(
        points=points,
        cells=[("quad", quads)],
        point_data={"boundary_mask": bmask.ravel()},
    )
    return Mesh(mio, reorder=False)


def structured_quad9_mesh(nx: int = 33, ny: int = 33,
                          xlims=(0.0, 1.0), ylims=(0.0, 1.0),
                          dtype=np.float64) -> Mesh:
    """Structured **biquadratic** (``quad9``) mesh — the Taylor-Hood Q2/Q1 carrier.

    Parameters
    ----------
    nx, ny : int
        Number of *cell-corner* nodes along x and y. These corners carry the Q1
        (pressure) space; the mesh has ``(nx-1)*(ny-1)`` cells and
        ``(2nx-1)*(2ny-1)`` Q2 nodes.

    Returns
    -------
    tensormesh.Mesh
        Single ``"quad9"`` cell block on the **fine** node grid
        ``Nx = 2nx-1``, ``Ny = 2ny-1``, row-major (node ``k = I*Nx + J``), with a
        boolean ``boundary_mask`` over the outer frame.

    Notes
    -----
    Two facts make this mesh the bridge between the FNO and a mixed Taylor-Hood
    assembly, and both are relied on throughout ``physics.StokesProblem``:

    1. The Q2 velocity nodes form a *uniform* ``Ny x Nx`` grid — an image the FNO
       can consume and emit directly, exactly like the Q1 grid for Poisson.
    2. TensorMesh's Q1 sub-field on this mesh lives on the cell corners, i.e. the
       ``[::2, ::2]`` subgrid of the fine grid, and ``layout.node_ids("p")`` returns
       them **sorted** — which is precisely coarse row-major order. So pressure is a
       strided read of the fine grid, no scatter/gather table needed.

    Connectivity follows TensorMesh's ``quad9`` reference order — the four corners
    first (``BL, BR, TL, TR``), then the four edge midpoints in ``Quadrilateral.edge``
    order (bottom, left, right, top), then the cell centre — emitted with
    ``reorder=False`` so each physical node maps to the matching reference node.
    """
    Nx, Ny = 2 * nx - 1, 2 * ny - 1
    xs = np.linspace(xlims[0], xlims[1], Nx)
    ys = np.linspace(ylims[0], ylims[1], Ny)
    X, Y = np.meshgrid(xs, ys)                        # both [Ny, Nx]
    points = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(dtype)

    nid = np.arange(Nx * Ny, dtype=np.int64).reshape(Ny, Nx)
    # cell (i, j) spans fine rows {2i, 2i+1, 2i+2} and columns {2j, 2j+1, 2j+2}
    conn = np.stack([
        nid[0:-2:2, 0:-2:2],   # 0  BL corner
        nid[0:-2:2, 2::2],     # 1  BR corner
        nid[2::2, 0:-2:2],     # 2  TL corner
        nid[2::2, 2::2],       # 3  TR corner
        nid[0:-2:2, 1:-1:2],   # 4  bottom edge midpoint  (BL, BR)
        nid[1:-1:2, 0:-2:2],   # 5  left edge midpoint    (BL, TL)
        nid[1:-1:2, 2::2],     # 6  right edge midpoint   (BR, TR)
        nid[2::2, 1:-1:2],     # 7  top edge midpoint     (TL, TR)
        nid[1:-1:2, 1:-1:2],   # 8  cell centre
    ], axis=-1).reshape(-1, 9)

    bmask = np.zeros((Ny, Nx), dtype=bool)
    bmask[0, :] = bmask[-1, :] = bmask[:, 0] = bmask[:, -1] = True

    mio = meshio.Mesh(
        points=points,
        cells=[("quad9", conn)],
        point_data={"boundary_mask": bmask.ravel()},
    )
    return Mesh(mio, reorder=False)


def node_to_grid(f: torch.Tensor, nx: int, ny: int) -> torch.Tensor:
    """Flat node vector ``[..., nx*ny]`` → image grid ``[..., ny, nx]`` (node ``k=i*nx+j``)."""
    return f.reshape(*f.shape[:-1], ny, nx)


def grid_to_node(g: torch.Tensor, nx: int, ny: int) -> torch.Tensor:
    """Image grid ``[..., ny, nx]`` → flat node vector ``[..., nx*ny]`` (inverse of :func:`node_to_grid`)."""
    return g.reshape(*g.shape[:-2], ny * nx)
