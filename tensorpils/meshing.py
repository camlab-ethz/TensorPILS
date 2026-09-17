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

__all__ = ["structured_quad_mesh", "structured_quad9_mesh", "circle_mesh",
           "node_to_grid", "grid_to_node"]


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


def circle_mesh(chara_length: float = 0.015, cx: float = 0.5, cy: float = 0.5,
                r: float = 0.5, cache_path=None) -> Mesh:
    r"""Unstructured triangular mesh of a disc — the first domain the FNO cannot serve.

    Everything above this function builds a *structured* mesh because the FNO needs an image.
    This one does the opposite on purpose: it is TensorMesh's Gmsh generator, so the node
    numbering is whatever Gmsh produced and there is no ``node_to_grid``. Only the point-cloud
    architectures (:class:`~tensorpils.gaot.GAOTModel`) and the matrix-based preconditioners
    (``--precond_kind amg``) can consume it; the geometric V-cycle and the FFT cannot.

    Parameters
    ----------
    chara_length : float
        Gmsh target edge length. ``0.015`` gives ~4200 nodes on the unit disc, which is the
        node budget of the ``64^2`` structured grid (4096) — the point of that default is that
        a GAOT run here is comparable to the structured one in problem size.
    cx, cy, r : float
        Centre and radius. The default disc is inscribed in the unit square, so the *same*
        ``PoissonMultiFrequency`` source fields are in range (it asserts points lie in
        :math:`[0,1]^2`).
    cache_path : str, optional
        Write/read the generated mesh here, so a sweep does not re-run Gmsh per job.

    Notes
    -----
    Importing ``gmsh`` needs ``libGLU.so.1``: the PyPI wheel's ``libgmsh.so`` links the GUI
    code path even though meshing never calls it, and Ubuntu 22.04 does not ship GLU. On Euler
    that is the ``mesa-glu`` module, which ``~/cluster-kit/env/projects/TensorPILS.euler.sh``
    already loads — so jobs are fine and only a bare interactive shell trips over it.

    **The analytical solution does not survive the change of domain.** ``PoissonMultiFrequency``
    is a sum of ``sin(i pi x) sin(j pi y)``, which vanishes on the boundary of the *square*,
    not of the disc: measured on this mesh the closed form reaches 2.4e-2 on the circle against
    an interior peak of 2.6e-2, i.e. ~90 % of the peak. So labels here must come from the FEM
    solve (:class:`~tensorpils.data.FEMPoissonSolver`, which factorises the assembled ``A`` and
    is already mesh-agnostic), exactly as for Allen–Cahn. Passing ``solution="analytic"`` on a
    non-rectangular domain is silently wrong, and :mod:`tensorpils.cli` refuses it.
    """
    try:
        return Mesh.gen_circle(chara_length=chara_length, element_type="tri",
                               cx=cx, cy=cy, r=r, cache_path=cache_path)
    except OSError as e:                                        # pragma: no cover
        if "libGLU" in str(e):
            raise OSError(
                f"{e}\n\nGmsh's shared library needs libGLU.so.1 (it links the GUI path it "
                f"never calls). On Euler:  module load stack/.2024-04-silent gcc/8.5.0 "
                f"mesa-glu/9.0.2  -- and note `module load` inside a PIPELINE runs in a "
                f"subshell, so the change to LD_LIBRARY_PATH is lost. Submitted jobs already "
                f"get this from the project env file.") from e
        raise
