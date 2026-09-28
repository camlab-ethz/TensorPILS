"""Meshes as :class:`tensormesh.Mesh`: the structured grid, the obstacle domain, grid↔node maps.

**The structured grid** (Poisson, Allen–Cahn). TensorMesh has no tensor-product mesh
generator — ``Mesh.gen_rectangle`` goes through Gmsh and yields an *unstructured* mesh with
arbitrary node numbering. A Fourier Neural Operator, however, needs a regular grid whose node
index maps to a grid cell by a plain reshape. So TensorPILS keeps its own structured-grid
builder, which emits a genuine :class:`tensormesh.Mesh` that TensorMesh's assemblers consume
directly.

Node ordering is row-major: node ``k = i * nx + j`` sits at grid row ``i`` (the second
physical coordinate, "y") and column ``j`` (the first physical coordinate, "x").
:func:`node_to_grid` / :func:`grid_to_node` are exact inverses that move between the
flat node vector ``[..., nx*ny]`` and the image-shaped grid ``[..., ny, nx]``.

**The obstacle domain** (Stokes). :func:`obstacle_mesh` triangulates the unit square with a
circular hole — a geometry no structured grid resolves — as a ``triangle6`` mesh carrying the
Taylor-Hood P2/P1 pair.
"""

import os
import tempfile

import numpy as np
import meshio
import torch

from tensormesh import Mesh

__all__ = ["structured_quad_mesh", "obstacle_mesh", "topological_boundary_mask",
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


def node_to_grid(f: torch.Tensor, nx: int, ny: int) -> torch.Tensor:
    """Flat node vector ``[..., nx*ny]`` → image grid ``[..., ny, nx]`` (node ``k=i*nx+j``)."""
    return f.reshape(*f.shape[:-1], ny, nx)


def grid_to_node(g: torch.Tensor, nx: int, ny: int) -> torch.Tensor:
    """Image grid ``[..., ny, nx]`` → flat node vector ``[..., nx*ny]`` (inverse of :func:`node_to_grid`)."""
    return g.reshape(*g.shape[:-2], ny * nx)



def topological_boundary_mask(mesh) -> torch.Tensor:
    r"""Boundary nodes from the mesh's **boundary facet cells**, not from coordinates.

    A Gmsh mesh carries its boundary as ``line`` / ``line3`` cells, and a node is on the boundary
    exactly when some boundary facet uses it. That is a topological fact and it is exact for any
    geometry.

    This exists because the coordinate test is not. A boundary mask built from exact coordinate
    equalities (``points[:, 0] == left``, ``radius == r``) misses every node Gmsh places an ULP
    off a curved locus; those nodes are then *free*, the Dirichlet condition is not imposed
    there, and nothing raises — the FEM reference simply solves a different problem. On the
    obstacle domain the whole hole would be missed.
    """
    keys = list(mesh.cells.keys())
    facet_key = next((k for k in ("line3", "line") if k in keys), None)
    if facet_key is None:                                    # pragma: no cover
        raise RuntimeError(
            f"mesh has no boundary facet cells (has {keys}); cannot identify the Dirichlet "
            f"nodes topologically. Generate it with a boundary physical group.")
    mask = torch.zeros(mesh.n_points, dtype=torch.bool)
    mask[torch.unique(mesh.cells[facet_key].reshape(-1))] = True
    return mask


def _obstacle_cache_name(chara_length: float, order: int, xlims, ylims,
                         cx: float, cy: float, r: float) -> str:
    """File name encoding every parameter that determines the mesh.

    A cache hit is therefore only possible for the mesh that was asked for: two calls that differ
    in any geometric parameter cannot read each other's file.
    """
    return (f"obstacle_h{chara_length:g}_order{order}_x{xlims[0]:g}-{xlims[1]:g}"
            f"_y{ylims[0]:g}-{ylims[1]:g}_hole{cx:g}-{cy:g}-{r:g}.msh")


def obstacle_mesh(chara_length: float = 0.035, order: int = 2,
                  xlims=(0.0, 1.0), ylims=(0.0, 1.0),
                  cx: float = 0.40, cy: float = 0.50, r: float = 0.14,
                  cache_dir=None) -> Mesh:
    r"""Rectangle with a circular hole — the Stokes domain, which no structured grid resolves.

    Emitted as ``triangle6`` (``order=2``) so it carries a **P2/P1 Taylor-Hood** pair, with zero
    velocity on *both* boundary components (the outer box and the hole).

    Parameters
    ----------
    chara_length : float
        Gmsh target edge length. The default ``0.035`` gives 3998 P2 / 1035 P1 nodes on the
        default geometry, the mesh of the paper's Stokes experiment.
    order : int
        2 for ``triangle6`` (Taylor-Hood velocity); 1 gives plain ``triangle``.
    xlims, ylims : tuple(float, float)
        The outer box.
    cx, cy, r : float
        Centre and radius of the hole. Off-centre by default: a centred hole makes the
        solution inherit the domain's symmetry, which is a weaker test.
    cache_dir : str, optional
        Directory for generated meshes. The file name encodes every parameter above, so a
        cached mesh is reused only for the exact geometry it was built for.

    Notes
    -----
    The boundary mask is taken from the mesh's **boundary line cells**
    (:func:`topological_boundary_mask`), not from coordinate comparisons, which cannot mark a
    curved hole: every node on the circle would be missed and "flow past a body" would silently
    become "flow through a body".

    Importing ``gmsh`` needs the system library ``libGLU.so.1`` (the PyPI wheel links Gmsh's GUI
    code path even though meshing never calls it); on Debian/Ubuntu it is ``libglu1-mesa``.
    """
    if cache_dir is not None:
        path = os.path.join(cache_dir, _obstacle_cache_name(chara_length, order, xlims, ylims,
                                                            cx, cy, r))
        if not os.path.exists(path):
            os.makedirs(cache_dir, exist_ok=True)
            _write_obstacle_msh(path, chara_length, order, xlims, ylims, cx, cy, r)
        return _load_obstacle_msh(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "obstacle.msh")
        _write_obstacle_msh(path, chara_length, order, xlims, ylims, cx, cy, r)
        return _load_obstacle_msh(path)


def _write_obstacle_msh(path: str, chara_length: float, order: int, xlims, ylims,
                        cx: float, cy: float, r: float) -> None:
    """Mesh the box-minus-disc with Gmsh and write it to ``path``."""
    try:
        import gmsh
    except OSError as e:                                        # pragma: no cover
        if "libGLU" in str(e):
            raise OSError(
                f"{e}\n\nGmsh's shared library needs libGLU.so.1 (it links the GUI code path "
                f"it never calls). Install your system's GLU package, e.g. `apt install "
                f"libglu1-mesa`, or load the corresponding module on a cluster.") from e
        raise

    x0, x1 = xlims
    y0, y1 = ylims
    gmsh.initialize()
    gmsh.model.add("obstacle")
    outer = gmsh.model.occ.addRectangle(x0, y0, 0, x1 - x0, y1 - y0)
    inner = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
    # The boolean CONSUMES both inputs and returns the new entities; everything after this
    # must refer to `cut`, never to `outer`/`inner`.
    cut, _ = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
    gmsh.model.occ.synchronize()

    surfaces = [tag for dim, tag in cut if dim == 2]
    boundary = gmsh.model.getBoundary([(2, s) for s in surfaces], oriented=False)
    grp = gmsh.model.addPhysicalGroup(1, [tag for dim, tag in boundary if dim == 1])
    gmsh.model.setPhysicalName(1, grp, "boundary")
    gmsh.model.addPhysicalGroup(2, surfaces)

    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), chara_length)
    gmsh.model.mesh.generate(2)
    gmsh.write(path)
    gmsh.finalize()


def _load_obstacle_msh(path: str) -> Mesh:
    """Read a mesh written by :func:`_write_obstacle_msh`, with the topological boundary mask."""
    # reorder=True is REQUIRED, not cosmetic: Gmsh numbers the triangle6 edge nodes differently
    # from TensorMesh, and with the raw numbering the mixed P2/P1 assembly is silently wrong --
    # the constant pressure mode then leaves a residual of 2.1e-1 on the free momentum rows (it
    # must be 0, since B^T 1 = 0 is what makes the pressure gauge a gauge). With reorder=True it
    # is 6.6e-15. Nothing raises either way.
    mesh = Mesh.from_file(path, reorder=True)
    mesh.register_point_data("is_boundary", topological_boundary_mask(mesh))
    return mesh
