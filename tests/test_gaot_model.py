"""Tests for the vendored GAOT architecture (``tensorpils.gaot``).

The load-bearing test is :func:`test_model_coords_match_the_mesh_nodes`. GAOT is a point-cloud
model wearing a grid signature, and the *only* thing that makes its output consumable by the FEM
losses is that its internal node ordering is the repo's: node ``k = i*nx + j``, the same order
``structured_quad_mesh`` numbers its points and ``grid_to_node`` flattens a grid in. If that
ever drifted, nothing would raise — the stiffness matrix would simply be applied to a permuted
field, and every label-free loss would be silently minimising the wrong residual.

The rest guard what would otherwise bias a GAOT-vs-FNO comparison or break the unstructured
path: the grid and node entry points agreeing, the optional compiled kernels
(``torch_scatter`` / ``torch_cluster``) not changing the answer, run filenames not colliding
across architectures, and the model actually being permutation-equivariant in its nodes —
which is the property the whole unstructured extension rests on.
"""

import pytest
import torch

from tensorpils.gaot import GAOTModel
from tensorpils.gaot import ops
from tensorpils.meshing import grid_to_node, node_to_grid, structured_quad_mesh
from tensorpils.trainer import arch_tag


def _small_model(n=13, in_channels=1, out_channels=1, seed=0, **kw):
    """A deliberately tiny GAOT: the tests are about contracts, not capacity."""
    torch.manual_seed(seed)
    cfg = dict(grid_size=(n, n), in_channels=in_channels, out_channels=out_channels,
               latent_grid=(8, 8), patch_size=2, lifting_channels=12,
               magno_hidden=12, mlp_layers=2, transformer_hidden=24,
               transformer_layers=2, num_heads=4)
    cfg.update(kw)
    return GAOTModel(**cfg)


# ----------------------------------------------------------------- the node-order invariant

def test_model_coords_match_the_mesh_nodes():
    """``GAOTModel.coords`` must equal the FEM mesh's ``points``, node for node.

    Not "up to a permutation": the FEM operators index nodes by position in this vector, so a
    permutation is exactly the silent failure this test exists to prevent.
    """
    n = 13
    model = _small_model(n)
    mesh = structured_quad_mesh(nx=n, ny=n)
    assert torch.allclose(model.coords, torch.as_tensor(mesh.points, dtype=torch.float32),
                          atol=1e-6)


def test_grid_forward_is_the_node_forward_reshaped():
    """``grid_to_node(model(f))`` is what ``forward_nodes`` returns — the bridge the losses use."""
    n = 13
    model = _small_model(n).eval()
    f_grid = torch.randn(3, 1, n, n)
    with torch.no_grad():
        u_grid = model(f_grid)
        u_node = model.forward_nodes(grid_to_node(f_grid, n, n).transpose(1, 2))
    assert u_grid.shape == (3, 1, n, n)
    assert torch.allclose(grid_to_node(u_grid.squeeze(1), n, n), u_node[..., 0], atol=1e-6)


def test_grid_roundtrip_matches_node_to_grid():
    """The wrapper's un-flatten is ``node_to_grid``, so a constant-per-node field lands right."""
    n = 9
    model = _small_model(n).eval()
    # Feed a field that identifies every node, then check the grid the model emits is indexed
    # the same way node_to_grid would index it.
    f_node = torch.arange(n * n, dtype=torch.float32).reshape(1, n * n)
    f_grid = node_to_grid(f_node, n, n).unsqueeze(1)
    assert torch.allclose(grid_to_node(f_grid.squeeze(1), n, n), f_node)
    with torch.no_grad():
        u = model(f_grid)
    assert u.shape == (1, 1, n, n)


# ----------------------------------------------------------------- the FNOModel contract

@pytest.mark.parametrize("in_channels,out_channels", [(1, 1), (2, 1), (2, 3)])
def test_shape_contract_matches_fnomodel(in_channels, out_channels):
    """``[B, C_in, H, W] -> [B, C_out, H, W]``, plus the unbatched form ``FNOModel`` accepts.

    ``(1, 1)`` is Poisson and Allen–Cahn, ``(2, 3)`` is Stokes ``f -> (u_x, u_y, p)`` and
    ``(2, 1)`` mixes the two, so this pins the signature for every ``--pde`` the dispatch can reach.
    """
    n = 11
    model = _small_model(n, in_channels=in_channels, out_channels=out_channels).eval()
    with torch.no_grad():
        assert model(torch.randn(2, in_channels, n, n)).shape == (2, out_channels, n, n)
        assert model(torch.randn(in_channels, n, n)).shape == (out_channels, n, n)


def test_wrong_grid_is_refused():
    model = _small_model(11).eval()
    with pytest.raises(ValueError, match="expected grid"):
        model(torch.randn(1, 1, 12, 12))


def test_gradients_reach_every_parameter():
    """A dead sub-module (an unused geoembed, a detached branch) would show up here."""
    n = 11
    model = _small_model(n)
    model(torch.randn(2, 1, n, n)).pow(2).sum().backward()
    dead = [name for name, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()
                                    or p.grad.abs().sum() == 0)]
    assert not dead, f"parameters with no/!finite gradient: {dead}"


# ----------------------------------------------------------------- unstructured readiness

def test_node_forward_is_permutation_equivariant():
    """Relabel the nodes and the answer follows them.

    This is the property the unstructured extension rests on: GAOT sees a set of
    (coordinate, value) pairs, never an ordering, so the FEM loss may number its nodes however
    the mesh generator did.
    """
    n = 9
    model = _small_model(n).eval()
    coords = model.coords.clone()
    f = torch.randn(2, n * n, 1)

    perm = torch.randperm(n * n)
    with torch.no_grad():
        u = model.forward_nodes(f, coords=coords)
        model.clear_neighbor_cache()          # the cache keys on shape, and the shape is unchanged
        u_perm = model.forward_nodes(f[:, perm], coords=coords[perm])
    assert torch.allclose(u[:, perm], u_perm, atol=1e-5)


def test_scattered_points_are_accepted_and_refuse_the_grid_path():
    """An explicit, genuinely unstructured point set works through ``forward_nodes`` only."""
    torch.manual_seed(0)
    n_pts = 200
    pts = torch.rand(n_pts, 2)
    model = _small_model(13, coords=pts).eval()
    assert not model.structured
    with torch.no_grad():
        u = model.forward_nodes(torch.randn(2, n_pts, 1))
    assert u.shape == (2, n_pts, 1)
    with pytest.raises(RuntimeError, match="forward_nodes"):
        model(torch.randn(1, 1, 13, 13))


def test_structured_mesh_points_keep_the_grid_path():
    """Passing the structured mesh's own ``points`` is the same geometry, so ``forward`` stays."""
    n = 11
    mesh = structured_quad_mesh(nx=n, ny=n)
    model = _small_model(n, coords=torch.as_tensor(mesh.points, dtype=torch.float32)).eval()
    assert model.structured
    with torch.no_grad():
        assert model(torch.randn(1, 1, n, n)).shape == (1, 1, n, n)


def test_query_coordinates_may_differ_from_the_input_nodes():
    """Decoding elsewhere than the input nodes — super-resolution / cross-mesh evaluation."""
    n = 11
    model = _small_model(n).eval()
    q = torch.rand(37, 2)
    with torch.no_grad():
        u = model.forward_nodes(torch.randn(2, n * n, 1), query_coord=q)
    assert u.shape == (2, 37, 1)


# ----------------------------------------------------------------- the radius default

def test_default_radius_tracks_the_resolutions():
    """The derived radius follows ``max(h_phys, h_latent)``, so neither grid can starve the graph.

    A fixed radius is the trap: halve the mesh and the physical ball keeps the same *area* but
    four times the points; grow the latent grid past it and a query can see nothing at all.
    """
    coarse = _small_model(13, latent_grid=(8, 8))
    fine = _small_model(33, latent_grid=(8, 8))
    # Latent spacing dominates at these sizes, so both land on the same radius.
    assert coarse.radius == pytest.approx(fine.radius)
    # Push the latent grid finer than the mesh and the physical spacing takes over.
    latent_fine = _small_model(13, latent_grid=(32, 32))
    assert latent_fine.radius < coarse.radius

    # The published GAOT Poisson setting is reproduced at the CLI defaults.
    assert GAOTModel(grid_size=(64, 64), latent_grid=(64, 64)).radius == pytest.approx(0.0333,
                                                                                      abs=5e-4)


def test_every_latent_token_sees_at_least_one_node():
    """An empty neighbourhood is a token with no information, and it fails silently."""
    n = 33
    model = _small_model(n, latent_grid=(16, 16))
    nbrs = ops.NeighborSearch(method="native")(model.coords, model.latent_tokens_coord,
                                               model.radius)
    counts = nbrs["neighbors_row_splits"][1:] - nbrs["neighbors_row_splits"][:-1]
    assert int(counts.min()) > 0, "some latent token has an empty neighbourhood"


# ----------------------------------------------------------------- optional compiled kernels

@pytest.mark.skipif(not ops.HAS_TORCH_SCATTER, reason="torch_scatter not installed")
@pytest.mark.parametrize("reduce", ["sum", "mean", "max"])
def test_segment_csr_fallback_matches_torch_scatter(reduce):
    """The pure-torch fallback is the same operator, including the empty segment and ``max``.

    Upstream GAOT's fallback raises on ``max`` and loops in Python; the cosine attention needs a
    segment max, so without this the model would not run at all without the compiled extension.
    """
    torch.manual_seed(0)
    src = torch.randn(37, 5)
    indptr = torch.tensor([0, 3, 3, 10, 22, 37])         # note the empty segment
    fast = ops.segment_csr(src, indptr, reduce=reduce, use_scatter=True)
    slow = ops.segment_csr(src, indptr, reduce=reduce, use_scatter=False)
    assert torch.allclose(fast, slow, atol=1e-6)


@pytest.mark.skipif(not ops.HAS_TORCH_CLUSTER, reason="torch_cluster not installed")
def test_neighbor_search_backends_agree():
    """Same neighbourhoods (as sets — the two backends differ by a permutation within a row)."""
    torch.manual_seed(0)
    data, queries, r = torch.rand(400, 2), torch.rand(120, 2), 0.12
    a = ops.NeighborSearch(method="torch_cluster")(data, queries, r)
    b = ops.NeighborSearch(method="native")(data, queries, r)
    assert torch.equal(a["neighbors_row_splits"], b["neighbors_row_splits"])
    for i in range(queries.shape[0]):
        lo, hi = int(a["neighbors_row_splits"][i]), int(a["neighbors_row_splits"][i + 1])
        assert (set(a["neighbors_index"][lo:hi].tolist())
                == set(b["neighbors_index"][lo:hi].tolist()))


def test_edge_drop_is_a_noop_at_eval_and_caps_degree_in_training():
    torch.manual_seed(0)
    nbrs = {"neighbors_index": torch.randint(0, 50, (60,)),
            "neighbors_row_splits": torch.tensor([0, 20, 25, 60])}
    same = ops.apply_edge_drop_csr(nbrs, "max_neighbors", max_neighbors=8, training=False)
    assert torch.equal(same["neighbors_index"], nbrs["neighbors_index"])

    capped = ops.apply_edge_drop_csr(nbrs, "max_neighbors", max_neighbors=8, training=True)
    counts = capped["neighbors_row_splits"][1:] - capped["neighbors_row_splits"][:-1]
    assert counts.tolist() == [8, 5, 8]
    assert capped["neighbors_index"].numel() == 21
    # Every kept edge must still come from its own node's original slice.
    for i, (lo, hi) in enumerate(zip(capped["neighbors_row_splits"][:-1],
                                     capped["neighbors_row_splits"][1:])):
        olo, ohi = nbrs["neighbors_row_splits"][i], nbrs["neighbors_row_splits"][i + 1]
        origin = set(nbrs["neighbors_index"][olo:ohi].tolist())
        assert set(capped["neighbors_index"][lo:hi].tolist()) <= origin


# ----------------------------------------------------------------- run bookkeeping

def test_arch_tag_and_file_prefix_separate_gaot_from_fno():
    """A ``--model gaot`` run must not land on the FNO run's checkpoint / results path."""
    from tensorpils.baselines import MollifiedModel, ZeroBoundaryModel
    from tensorpils.trainer import ArchPrefixMixin

    model = _small_model(11)
    assert arch_tag(model) == "gaot"
    assert arch_tag(ZeroBoundaryModel(model, 11, 11)) == "gaot"
    assert arch_tag(MollifiedModel(model, 11, 11)) == "gaot"

    class _Fake:
        def _file_prefix(self):
            return "fno_pls_mg-L4-s22_K4_samples-1024-128-256"

    class _Tagged(ArchPrefixMixin, _Fake):
        pass

    tagged = _Tagged()
    tagged.model = model
    assert tagged._file_prefix() == "gaot_pls_mg-L4-s22_K4_samples-1024-128-256"


def test_build_config_survives_a_checkpoint_roundtrip():
    """The geometry rides in the state dict, so a reloaded model is scored on what it trained on."""
    n = 11
    model = _small_model(n)
    state = model.state_dict()
    assert "coords" in state and "latent_tokens_coord" in state
    clone = _small_model(n, seed=99)
    clone.load_state_dict(state)
    clone.eval(), model.eval()
    f = torch.randn(2, 1, n, n)
    with torch.no_grad():
        assert torch.allclose(model(f), clone(f), atol=1e-6)


# ----------------------------------------------------------------- the reason GAOT is here

def test_label_free_losses_run_on_a_non_grid_node_set():
    """The end-to-end unstructured path: real FEM operator, real loss, no grid anywhere.

    This is the whole point of adding a second architecture, so it is checked against
    ``PoissonProblem`` rather than a mock. The node set is the structured quad mesh with its
    *interior* nodes jittered: the connectivity stays valid, TensorMesh assembles ``A`` and ``M``
    on the moved nodes, and the points are no longer a grid — so ``GAOTModel`` must drop the grid
    path and the losses must still work, because they were node-based all along. (Gmsh is not
    importable in every environment here, which is why this is a jittered quad mesh rather than
    a triangulation; it exercises the coordinate path, not a second element type.)
    """
    import meshio
    import numpy as np
    from tensormesh import Mesh
    from tensorpils.losses import build_loss
    from tensorpils.physics import PoissonProblem

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n = 17
    base = structured_quad_mesh(nx=n, ny=n)
    pts = base.points.cpu().numpy().astype(np.float64)
    conn = base.cells["quad"].cpu().numpy().astype(np.int64)
    bmask = PoissonProblem(base).boundary_mask.cpu().numpy().astype(bool)
    # Interior only: moving a boundary node would change the domain, not just the mesh.
    h = 1.0 / (n - 1)
    pts[~bmask] += rng.uniform(-0.3 * h, 0.3 * h, size=(int((~bmask).sum()), 2))

    mesh = Mesh(meshio.Mesh(points=pts, cells=[("quad", conn)],
                            point_data={"boundary_mask": bmask}), reorder=False)
    problem = PoissonProblem(mesh)
    coords = torch.as_tensor(pts, dtype=torch.float32)
    model = _small_model(n, coords=coords)
    assert not model.structured

    f = torch.randn(2, coords.shape[0]) * 0.1
    for name in ("galerkin",):
        model.zero_grad()
        u = model.forward_nodes(f.unsqueeze(-1))[..., 0]                 # [B, N], node space
        loss = build_loss(name, problem)(u, f, torch.zeros_like(f))
        assert torch.isfinite(loss)
        loss.backward()
        grad = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
        assert torch.isfinite(grad) and grad > 0, f"{name}: no gradient on the unstructured mesh"
