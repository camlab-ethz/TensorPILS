"""Tests for unstructured Taylor-Hood Stokes — ``--pde stokes --mesh obstacle``.

The load-bearing test is :func:`test_constant_pressure_mode_is_in_the_kernel`. TensorMesh's
mixed P2/P1 assembly is **silently wrong** on a mesh loaded with ``reorder=False``: the system
still solves, the solution still looks plausible, and a self-consistent residual check still
returns ~1e-17 — but the constant pressure mode is no longer in the kernel of ``Bᵀ`` on the free
momentum rows, which is exactly what makes the pressure gauge a gauge. Measured 2.1e-1 against
6.6e-15 for the correct numbering. The only thing that catches it is probing ``K·(0, 1)``
directly, which is what this test does.

The rest guard what had to be rebuilt off the grid: the pressure space read from the layout
rather than strided, ``M_p`` assembled on a sub-mesh, the boundary covering *both* components of
the boundary (a curved obstacle cannot be found by coordinate comparison), and the block
preconditioner's velocity half swapped for an algebraic V-cycle.
"""

import numpy as np
import pytest
import torch

from tensorpils.data import stokes_body_force

try:
    from tensorpils.meshing import obstacle_mesh
    _MESH = obstacle_mesh(chara_length=0.09, order=2)
except Exception as exc:  # pragma: no cover - gmsh needs libGLU; see circle_mesh's docstring
    _MESH, _WHY = None, str(exc)[:120]

pytestmark = pytest.mark.skipif(_MESH is None,
                                reason=f"gmsh unavailable: {locals().get('_WHY', '')}")

CX, CY, R = 0.40, 0.50, 0.14


@pytest.fixture(scope="module")
def mesh():
    return _MESH


@pytest.fixture(scope="module")
def problem(mesh):
    from tensorpils.physics import UnstructuredStokesProblem
    return UnstructuredStokesProblem(mesh, mu=1.0)


# ----------------------------------------------------------------- the silent-assembly guard

def test_constant_pressure_mode_is_in_the_kernel(problem):
    r"""``K·(0, 1)`` must vanish on the free momentum rows, i.e. ``Bᵀ1 = 0``.

    ``(Bᵀ1)_i = -∫ ∇·v_i`` and ``v_i`` vanishes on ``∂Ω`` for an interior node, so the divergence
    theorem makes this exactly zero — *if* the assembly is right. It is the property that makes
    the pressure defined only up to a constant, so if it fails, ``project_pressure_gauge`` moves
    the solution off the manifold and every residual-based loss is minimising the wrong thing.
    """
    c = problem.pack(torch.zeros(1, problem.n_u, 2, dtype=torch.float64),
                     torch.ones(1, problem.n_p, dtype=torch.float64))
    Kc = problem._spmm(problem.K.to(torch.float64), c)[0]
    free_momentum = (~problem.dirichlet_mask).clone()
    free_momentum[problem.off_p:] = False
    assert Kc[free_momentum].norm() < 1e-10, (
        "the constant pressure mode is not in the kernel: the mixed assembly is wrong. Check "
        "that the mesh was loaded with reorder=True.")


def test_pressure_gauge_does_not_move_the_residual(problem):
    """Shifting the pressure by a constant must leave the residual alone — the same fact, end to end."""
    torch.manual_seed(0)
    f = stokes_body_force(torch.rand(2, 2, 6, 6) * 2 - 1, problem_points(problem)).float()
    u, p = problem.fem_reference(f, chunk=2)
    r0 = problem.residual(u, p, f).norm(dim=1)
    r1 = problem.residual(u, p + 0.37, f).norm(dim=1)
    assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-8)


def problem_points(problem):
    """The P2 node coordinates the problem was built on."""
    return _MESH.points


# ----------------------------------------------------------------- the operator

def test_reference_zeroes_the_training_residual(problem):
    """The label is the discrete solution the label-free losses target, to solver accuracy."""
    torch.manual_seed(0)
    f = stokes_body_force(torch.rand(4, 2, 6, 6) * 2 - 1, _MESH.points).float()
    u, p = problem.fem_reference(f, chunk=4)
    load = torch.stack([problem._spmm(problem.M, f[..., c]) for c in range(2)], dim=-1)
    b = problem.pack(load, torch.zeros(4, problem.n_p))
    rel = (problem.residual(u, p, f).norm(dim=1) / b.norm(dim=1)).max()
    assert rel < 1e-4, f"reference does not solve the system (relative residual {rel:.2e})"
    assert torch.all(u[:, problem.boundary_mask] == 0)


def test_pressure_mass_integrates_the_domain(problem):
    """``1ᵀ M_p 1`` is the area of the domain *with the hole removed*.

    A cheap, sharp check on the P1 sub-mesh: if its connectivity or its node ordering were wrong
    the mass matrix would still be symmetric positive definite and would still look fine — but
    it would integrate to something else entirely.

    The tolerance is 1 % because this fixture is deliberately coarse (``chara_length=0.09``, so
    the hole's boundary is about ten elements) and the mesh's approximation of the disc carries
    a 0.4 % area error of its own. At the production resolution the same check is 0.06 %
    (0.939022 against 0.938425). A wrong ordering is not a 1 % effect.
    """
    area = 1.0 - np.pi * R ** 2
    assert float(problem.m_p_lumped.sum()) == pytest.approx(area, rel=1e-2)


def test_pressure_nodes_are_the_triangle_corners(mesh, problem):
    corners = torch.unique(mesh.cells["triangle6"][:, :3].reshape(-1))
    assert torch.equal(torch.sort(corners).values, torch.sort(problem.p_node_ids).values)
    assert problem.n_p < problem.n_u, "P1 must be a strict subspace of the P2 node set"


def test_from_nodes_splits_the_three_channels(problem):
    """The model emits 3 channels on the P2 nodes; pressure is a gather, not a reshape."""
    n_u, n_p = problem.n_u, problem.n_p
    out = torch.arange(2 * n_u * 3, dtype=torch.float32).reshape(2, n_u, 3)
    u, p = problem.from_nodes(out)
    assert u.shape == (2, n_u, 2) and p.shape == (2, n_p)
    assert torch.equal(u, out[..., :2])
    assert torch.equal(p, out[:, problem.p_node_ids, 2])


def test_grid_readers_are_absent(problem):
    assert problem.grid_size is None and problem.pgrid_size is None
    with pytest.raises(AttributeError):
        problem.from_grid(torch.zeros(1, 3, 4, 4))


# ----------------------------------------------------------------- the geometry

def test_boundary_covers_both_components_and_the_hole_is_empty(mesh):
    """Outer frame *and* obstacle, with nothing inside the hole.

    A coordinate test — which is what TensorMesh's own generators use — cannot mark a curved
    obstacle at all, and the problem would silently become flow *through* the body.
    """
    pts = mesh.points.numpy()
    b = mesh.boundary_mask.numpy().astype(bool)
    d = np.hypot(pts[:, 0] - CX, pts[:, 1] - CY)
    on_outer = ((np.abs(pts[:, 0]) < 1e-9) | (np.abs(pts[:, 0] - 1) < 1e-9)
                | (np.abs(pts[:, 1]) < 1e-9) | (np.abs(pts[:, 1] - 1) < 1e-9))
    on_hole = np.abs(d - R) < 1e-6

    assert (on_outer & ~b).sum() == 0, "outer-frame nodes left unconstrained"
    assert (on_hole & ~b).sum() == 0, "obstacle nodes left unconstrained"
    assert (b & on_hole).sum() > 0, "no boundary node on the obstacle at all"
    assert (b & ~(on_outer | on_hole)).sum() == 0, "a boundary node on neither component"
    assert (d < R - 1e-6).sum() == 0, "mesh nodes inside the obstacle"


# ----------------------------------------------------------------- dataset / trainer

def test_dataset_items_are_node_vectors():
    from tensorpils.data import create_unstructured_stokes_datasets

    tr, va, te = create_unstructured_stokes_datasets(
        n_train=4, n_val=2, n_test=2, K=6, chara_length=0.09, seed=0)
    assert tr.grid_size is None and tr.pgrid_size is None
    f, u, p = tr[0]
    assert f.shape == (tr.n_u, 2) and u.shape == (tr.n_u, 2) and p.shape == (tr.n_p,)
    assert tr.mesh is te.mesh and tr.problem is te.problem
    # The per-sample normalisation is to unit velocity FE-L2.
    assert float(tr.problem.velocity_l2(u.unsqueeze(0))[0]) == pytest.approx(1.0, rel=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AmgX is CUDA-only")
def test_block_preconditioner_takes_an_algebraic_velocity_block(problem):
    """``P = diag(Â⁻¹/μ, ω μ/diag(M_p))`` with an AMG velocity half, and it must be symmetric.

    ``P`` is used as a *norm*, so symmetry is not optional.
    """
    from tensorpils.preconditioners import AMGXPreconditioner, StokesBlockPreconditioner

    dev = "cuda"
    prob = problem.to(dev)
    amg = AMGXPreconditioner(prob.A, prob.boundary_mask, sweeps=2, device=dev)
    P = StokesBlockPreconditioner(prob, schur_omega=0.5, velocity_precond=amg).to(dev)

    torch.manual_seed(0)
    x = torch.randn(4, prob.n_dofs, device=dev)
    y = torch.randn(4, prob.n_dofs, device=dev)
    xPy = (x * P(y)).sum(dim=1)
    yPx = (y * P(x)).sum(dim=1)
    assert torch.allclose(xPy, yPx, rtol=1e-3, atol=1e-5), "P is not symmetric"
    assert ((x * P(x)).sum(dim=1) > 0).all(), "P is not positive definite on these vectors"
    prob.to("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AmgX is CUDA-only")
def test_trainer_runs_an_epoch_and_tags_its_files():
    from tensorpils.data import create_unstructured_stokes_datasets
    from tensorpils.gaot import GAOTModel
    from tensorpils.trainer import GAOTUnstructuredStokesTrainer

    tr, va, te = create_unstructured_stokes_datasets(
        n_train=4, n_val=2, n_test=2, K=6, chara_length=0.09, seed=0)
    torch.manual_seed(0)
    model = GAOTModel(grid_size=(32, 32), coords=tr.mesh.points.float(), in_channels=2,
                      out_channels=3, latent_grid=(8, 8), patch_size=2, lifting_channels=12,
                      magno_hidden=12, mlp_layers=2, transformer_hidden=24,
                      transformer_layers=2, num_heads=4)
    trainer = GAOTUnstructuredStokesTrainer(
        model=model, train_dataset=tr, val_dataset=va, test_dataset=te,
        loss_type="pls", batch_size=2, epochs=1, device="cuda",
        output_dir="/tmp/tensorpils_unstructured_stokes_test")
    trainer.save_checkpoints = False
    assert np.isfinite(trainer.train_epoch())
    assert np.isfinite(trainer.validate())

    prefix = trainer._file_prefix()
    assert prefix.startswith("gaot_stokes_")
    assert "amg-" in prefix, "the algebraic V-cycle must be visible in the run name"
    assert f"obstacle-n{tr.n_u}" in prefix, (
        "an unstructured run must not be able to land on a structured run's checkpoint")


def test_monolithic_preconditioner_is_refused():
    """It has no unstructured form, and the message has to say why rather than crash later."""
    from tensorpils.trainer import UnstructuredStokesTrainer

    with pytest.raises(SystemExit, match="geometric"):
        UnstructuredStokesTrainer(model=None, train_dataset=None, val_dataset=None,
                                  test_dataset=None, precond_kind="monolithic")
