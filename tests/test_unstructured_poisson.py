"""Tests for the unstructured (disc) Poisson path — ``--mesh circle``.

The load-bearing test is :func:`test_analytic_solution_is_invalid_on_the_disc`. The whole
dataset hinges on one fact that nothing in the code would otherwise announce:
``PoissonMultiFrequency`` is a sum of ``sin(i pi x) sin(j pi y)``, which is zero on the boundary
of the unit *square* and emphatically not on the boundary of the inscribed disc. Labelling with
the closed form there raises nothing — it just trains the model against a field that does not
solve the problem. The CLI refuses ``--dataset_solution analytic`` for that reason, and this
pins the magnitude behind the refusal.

The rest guard the pieces that had to be rewritten because they assumed an image: the node-form
collate/forward/eval, the boundary projection off the frame, the file prefix not colliding with
a structured run, and — the other half of going unstructured — that the geometric V-cycle really
cannot serve this mesh while the algebraic one can.
"""

import numpy as np
import pytest
import torch

from tensorpils.gaot import GAOTModel, ops
from tensorpils.losses import NodeDataLoss, build_loss
from tensorpils.physics import PoissonProblem

try:
    from tensorpils.meshing import circle_mesh
    _MESH = circle_mesh(chara_length=0.09)
except Exception as exc:  # pragma: no cover - gmsh needs libGLU; see circle_mesh's docstring
    _MESH, _WHY = None, str(exc)[:120]

pytestmark = pytest.mark.skipif(_MESH is None,
                                reason=f"gmsh unavailable: {locals().get('_WHY', '')}")


@pytest.fixture(scope="module")
def mesh():
    return _MESH


@pytest.fixture(scope="module")
def problem(mesh):
    return PoissonProblem(mesh)


def _model(mesh, **kw):
    torch.manual_seed(0)
    cfg = dict(grid_size=(32, 32), coords=mesh.points.float(), latent_grid=(8, 8),
               patch_size=2, lifting_channels=12, magno_hidden=12, mlp_layers=2,
               transformer_hidden=24, transformer_layers=2, num_heads=4)
    cfg.update(kw)
    return GAOTModel(**cfg)


# ----------------------------------------------------------------- the label trap

def test_analytic_solution_is_invalid_on_the_disc(mesh, problem):
    """The closed form is large on the disc's boundary, so it cannot be a label here.

    ``sin(i pi x) sin(j pi y)`` vanishes on the square's edges; the disc's boundary runs through
    the interior of the square, where the field is at full amplitude. Measured, the boundary
    value is a substantial fraction of the interior peak — nowhere near the ``0`` the problem
    imposes.
    """
    from tensormesh.dataset import PoissonMultiFrequency

    torch.manual_seed(0)
    a = torch.rand(4, 4, 4) * 2 - 1
    u = PoissonMultiFrequency(a=a, r=-0.5).solution(mesh.points).float()
    bnd = problem.boundary_mask.bool()
    boundary_peak = u[:, bnd].abs().max()
    interior_peak = u[:, ~bnd].abs().max()
    assert boundary_peak > 0.25 * interior_peak, (
        "the closed form happens to be small on this boundary — re-check the claim in "
        "meshing.circle_mesh before relying on it")


def test_fem_labels_satisfy_the_boundary_condition_and_zero_the_residual(mesh, problem):
    """The FEM solve is the label, and it is the *discrete* solution the residual losses target."""
    from tensormesh.dataset import PoissonMultiFrequency
    from tensorpils.data import FEMPoissonSolver

    torch.manual_seed(0)
    a = torch.rand(3, 4, 4) * 2 - 1
    f = PoissonMultiFrequency(a=a, r=-0.5).source_term(mesh.points, domain="rectangle").float()
    u = FEMPoissonSolver(problem)(f)

    bnd = problem.boundary_mask.bool()
    assert torch.all(u[:, bnd] == 0), "FEM labels must be exactly zero on the boundary"
    # The same residual the galerkin loss minimises, evaluated at the label.
    assert problem.residual(u, f).norm(dim=1).max() < 1e-5


# ----------------------------------------------------------------- dataset

def test_dataset_items_are_node_vectors_with_no_grid():
    from tensorpils.data import create_unstructured_datasets

    tr, va, te = create_unstructured_datasets(n_train=4, n_val=2, n_test=2, K=4,
                                              chara_length=0.09, seed=0)
    assert (len(tr), len(va), len(te)) == (4, 2, 2)
    assert tr.grid_size is None, "an unstructured dataset must not advertise a grid"
    f, u = tr[0]
    assert f.shape == u.shape == (tr.n_nodes,)
    # One mesh, one problem, one pool across the splits.
    assert tr.mesh is va.mesh is te.mesh and tr.problem is va.problem
    # The splits must be disjoint samples, not the same pool re-indexed.
    assert not torch.allclose(tr[0][0], te[0][0])


# ----------------------------------------------------------------- losses off the grid

def test_node_data_loss_equals_grid_data_loss_on_a_uniform_grid():
    """The unstructured ``data`` arm must be the same number as the structured one.

    On a uniform grid the two differ only in shape, so if this drifted the two tables would stop
    being comparable while still both looking plausible.
    """
    from tensorpils.losses import DataLoss
    from tensorpils.meshing import structured_quad_mesh, node_to_grid

    n = 9
    sq = PoissonProblem(structured_quad_mesh(nx=n, ny=n))
    torch.manual_seed(0)
    pred, true = torch.randn(3, n * n), torch.randn(3, n * n)
    grid = DataLoss(bc_mode="penalty")(node_to_grid(pred, n, n), node_to_grid(true, n, n))
    node = NodeDataLoss(sq, bc_mode="penalty")(pred, true)
    assert torch.allclose(grid, node, atol=1e-6)

    grid_h = DataLoss(bc_mode="hard")(node_to_grid(pred, n, n), node_to_grid(true, n, n))
    node_h = NodeDataLoss(sq, bc_mode="hard")(pred, true)
    assert torch.allclose(grid_h, node_h, atol=1e-6)


@pytest.mark.parametrize("loss_type", ["galerkin", "deepritz", "data", "data_l2", "data_h1"])
def test_losses_train_on_the_disc(mesh, problem, loss_type):
    """Every mesh-agnostic loss computes and backpropagates through ``forward_nodes``."""
    model = _model(mesh)
    N = mesh.points.shape[0]
    torch.manual_seed(0)
    f = torch.randn(2, N) * 0.1
    u_true = torch.randn(2, N) * 0.01

    crit = build_loss(loss_type, problem, lambda_bc=1.0, bc_mode="hard", node_form=True)
    u = model.forward_nodes(f.unsqueeze(-1))[..., 0]
    loss = crit(u, f, u_true) if loss_type != "data" else crit(u, u_true)
    assert torch.isfinite(loss)
    loss.backward()
    g = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(g) and g > 0


# ----------------------------------------------------------------- the preconditioner half

def test_geometric_multigrid_cannot_serve_the_disc(problem):
    """GMG builds without complaint and dies at the first apply — worth pinning, since the CLI
    guard is the only thing that turns that into a legible message."""
    from tensorpils.preconditioners import build_preconditioner

    P = build_preconditioner("multigrid", problem, grid_size=(64, 64))
    with pytest.raises(RuntimeError):
        P(torch.randn(2, problem.A.shape[0]))


# ----------------------------------------------------------------- GAOT on this geometry

def test_latent_grid_follows_the_point_cloud(mesh):
    """The latent grid must cover the geometry, not a hard-coded unit square."""
    pts = mesh.points.float()
    shifted = pts + torch.tensor([10.0, -5.0])
    m = _model(mesh, coords=shifted)
    lat = m.latent_tokens_coord
    assert lat[:, 0].min() == pytest.approx(float(shifted[:, 0].min()), abs=1e-4)
    assert lat[:, 1].max() == pytest.approx(float(shifted[:, 1].max()), abs=1e-4)


def test_every_mesh_node_can_be_decoded(mesh):
    """Latent tokens outside the disc are expected (and harmless — an empty neighbourhood is a
    zero encoding). A *query* node with no latent token in range is not: its decoded value would
    be zero before the projection, for reasons unrelated to the network."""
    m = _model(mesh, latent_grid=(16, 16))
    dec = ops.NeighborSearch(method="native")(m.latent_tokens_coord, m.coords, m.radius)
    counts = dec["neighbors_row_splits"][1:] - dec["neighbors_row_splits"][:-1]
    assert int(counts.min()) > 0


# ----------------------------------------------------------------- trainer wiring

def test_trainer_runs_an_epoch_and_tags_its_files():
    from tensorpils.data import create_unstructured_datasets
    from tensorpils.trainer import GAOTUnstructuredPoissonTrainer

    tr, va, te = create_unstructured_datasets(n_train=4, n_val=2, n_test=2, K=4,
                                              chara_length=0.09, seed=0)
    model = _model(tr.mesh)
    trainer = GAOTUnstructuredPoissonTrainer(
        model=model, train_dataset=tr, val_dataset=va, test_dataset=te,
        loss_type="galerkin", batch_size=2, epochs=1, device="cpu",
        output_dir="/tmp/tensorpils_unstructured_test")
    trainer.save_checkpoints = False
    loss = trainer.train_epoch()
    assert np.isfinite(loss)
    mse, l2, rl2 = trainer.validate()
    assert all(np.isfinite(v) for v in (mse, l2, rl2))

    prefix = trainer._file_prefix()
    assert prefix.startswith("gaot_"), "architecture must be in the prefix"
    assert f"circle-n{tr.n_nodes}" in prefix, (
        "an unstructured run must not be able to land on a structured run's checkpoint")


def test_eval_projects_the_boundary_to_zero():
    """``eval_project_bc`` is inherited from PoissonTrainer and must act on the mesh's boundary
    mask, not on an outer frame that does not exist here."""
    from tensorpils.data import create_unstructured_datasets
    from tensorpils.trainer import GAOTUnstructuredPoissonTrainer

    tr, va, te = create_unstructured_datasets(n_train=2, n_val=2, n_test=2, K=4,
                                              chara_length=0.09, seed=0)
    trainer = GAOTUnstructuredPoissonTrainer(
        model=_model(tr.mesh), train_dataset=tr, val_dataset=va, test_dataset=te,
        loss_type="galerkin", batch_size=2, epochs=1, device="cpu",
        output_dir="/tmp/tensorpils_unstructured_test")
    assert trainer.eval_project_bc is True
    u = torch.randn(2, tr.n_nodes)
    projected = trainer._apply_eval_bc(u)
    bnd = trainer.problem.boundary_mask.bool()
    assert torch.all(projected[:, bnd] == 0)
    assert torch.allclose(projected[:, ~bnd], u[:, ~bnd])


# ----------------------------------------------------------------- how to read the two tables

def test_square_target_is_sixteen_dimensional_and_the_disc_target_is_not(mesh, problem):
    """The structured task is a diagonal linear map on ``K²`` coefficients; the disc task is not.

    This is not a detail — it is how the two tables have to be read. On a uniform grid the
    discrete sines are exact eigenvectors of both the ``Q1`` stiffness and mass matrices, so a
    source in the 16-mode span gives a solution in the *same* span, and the whole dataset
    collapses to 16 scalars. A disc number is therefore not a degraded square number: the
    function class changed.
    """
    import numpy as np
    from tensormesh.dataset import PoissonMultiFrequency
    from tensorpils.data import FEMPoissonSolver
    from tensorpils.meshing import structured_quad_mesh

    K = 4
    torch.manual_seed(0)
    a = torch.rand(16, K, K) * 2 - 1
    eq = PoissonMultiFrequency(a=a, r=-0.5)

    def outside_span(m, prob):
        pts = m.points
        f = eq.source_term(pts, domain="rectangle").float()
        u = FEMPoissonSolver(prob)(f)
        x, y = pts[:, 0].float(), pts[:, 1].float()
        modes = torch.stack([torch.sin(np.pi * (i + 1) * x) * torch.sin(np.pi * (j + 1) * y)
                             for i in range(K) for j in range(K)], dim=1)
        Mm = prob._spmm(prob.M, modes.T.contiguous()).T
        G = (modes.T @ Mm).double()
        c = torch.linalg.solve(G, (Mm.T @ u.T).double()).float()
        r = u - (modes @ c).T
        Mr, Mu = prob._spmm(prob.M, r), prob._spmm(prob.M, u)
        return ((r * Mr).sum(1).clamp(min=0).sqrt()
                / (u * Mu).sum(1).clamp(min=0).sqrt()).mean().item()

    sq = structured_quad_mesh(nx=32, ny=32)
    assert outside_span(sq, PoissonProblem(sq)) < 1e-5, (
        "the square's FEM solution must lie in the 16-mode span exactly")
    assert outside_span(mesh, problem) > 0.02, (
        "the disc's FEM solution must leave that span — otherwise the unstructured case tests "
        "nothing the structured one did not")


def test_boundary_mask_is_topological_not_coordinate_based(mesh):
    """Every node on a boundary facet must be marked — a coordinate test cannot do this.

    ``Mesh.gen_circle`` builds ``is_boundary`` from ``radius == r``, which misses the nodes Gmsh
    places an ULP off the circle: 37 of 210 on the disc this repo trains on, sitting 5.6e-17 to
    1.1e-16 off the radius. Unmarked means *free*, so the Dirichlet condition is not imposed
    there and nothing raises — the reference simply solves a different problem (the labels move
    by 2.8 % in FEM relative L², with ``|u|`` reaching 17 % of the interior peak on those nodes).
    ``circle_mesh`` therefore replaces that mask; this test is what keeps it replaced.
    """
    import numpy as np
    from tensorpils.meshing import topological_boundary_mask

    assert torch.equal(mesh.boundary_mask.bool(), topological_boundary_mask(mesh))

    # ... and the coordinate test the generator uses really is the thing that fails.
    pts = mesh.points.numpy()
    r = np.hypot(pts[:, 0] - 0.5, pts[:, 1] - 0.5)
    marked = mesh.boundary_mask.numpy().astype(bool)
    exact = r == 0.5
    assert exact.sum() < marked.sum(), (
        "exact float equality happens to catch every boundary node on this mesh, so this test "
        "is not exercising anything; pick a resolution where it does not")
