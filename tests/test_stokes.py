"""Tests for Taylor-Hood P2/P1 Stokes on the obstacle mesh (``--pde stokes``).

The load-bearing test is :func:`test_constant_pressure_mode_is_in_the_kernel`. TensorMesh's
mixed P2/P1 assembly is **silently wrong** on a mesh loaded with ``reorder=False``: the system
still solves, the solution still looks plausible, and a self-consistent residual check still
returns ~1e-17 — but the constant pressure mode is no longer in the kernel of ``Bᵀ`` on the free
momentum rows, which is exactly what makes the pressure gauge a gauge. Measured 2.1e-1 against
6.6e-15 for the correct numbering. The only thing that catches it is probing ``K·(0, 1)``
directly, which is what this test does.

The rest guard the operator (layout, residual, reference solve), the pressure space read from
the assembly layout, ``M_p`` assembled on a sub-mesh, the boundary covering *both* components of
the boundary (a curved obstacle cannot be found by coordinate comparison), the block
preconditioner and losses, and the trainer's prediction path.
"""

import numpy as np
import pytest
import torch

from tensorpils.data import stokes_body_force

try:
    from tensorpils.meshing import obstacle_mesh
    _MESH = obstacle_mesh(chara_length=0.09, order=2)
except Exception as exc:  # pragma: no cover - gmsh needs libGLU; see obstacle_mesh's docstring
    _MESH, _WHY = None, str(exc)[:120]

pytestmark = pytest.mark.skipif(_MESH is None,
                                reason=f"gmsh unavailable: {locals().get('_WHY', '')}")

CX, CY, R = 0.40, 0.50, 0.14


def _amgx_available():
    try:
        import torch_amgx
        return torch_amgx.is_available()
    except Exception:
        return False


requires_amgx = pytest.mark.skipif(
    not _amgx_available(),
    reason="torch-amgx (CUDA-only) not importable; see the installation notes in README.md")


@pytest.fixture(scope="module")
def mesh():
    return _MESH


@pytest.fixture(scope="module")
def problem(mesh):
    from tensorpils.physics import StokesProblem
    return StokesProblem(mesh, mu=1.0)


def _random_fields(problem, B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(B, problem.n_u, 2, generator=g)
    p = torch.randn(B, problem.n_p, generator=g)
    f = torch.randn(B, problem.n_u, 2, generator=g)
    return u, p, f


class _JacobiVelocity(torch.nn.Module):
    """A CPU stand-in for the algebraic V-cycle: the inverse diagonal of the scalar P2 stiffness.

    SPD and ``[B, n_u] -> [B, n_u]``, which is all the block preconditioner asks of its velocity
    half, so the loss and preconditioner tests run without AmgX (which is CUDA-only).
    """

    def __init__(self, problem):
        super().__init__()
        d = problem.A.to_dense().diagonal().float()
        self.register_buffer("inv_d", 1.0 / d)

    def forward(self, r):
        return r * self.inv_d


def _block_precond(problem, schur_omega=0.5):
    from tensorpils.preconditioners import StokesBlockPreconditioner
    return StokesBlockPreconditioner(problem, velocity_precond=_JacobiVelocity(problem),
                                     schur_omega=schur_omega)


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
    f = stokes_body_force(torch.rand(2, 2, 6, 6) * 2 - 1, _MESH.points).float()
    u, p = problem.fem_reference(f, chunk=2)
    r0 = problem.residual(u, p, f).norm(dim=1)
    r1 = problem.residual(u, p + 0.37, f).norm(dim=1)
    assert torch.allclose(r0, r1, rtol=1e-3, atol=1e-8)


# ----------------------------------------------------------------- the operator

def test_pack_unpack_roundtrip(problem):
    u, p, _ = _random_fields(problem)
    c = problem.pack(u, p)
    assert c.shape == (u.shape[0], problem.n_dofs)
    u2, p2 = problem.unpack(c)
    assert torch.allclose(u, u2) and torch.allclose(p, p2)


def test_saddle_point_operator_structure(problem):
    """K is symmetric, its velocity block decouples across components and equals mu*A_P2, and
    the pressure-pressure block is zero (a genuine saddle point)."""
    Kd = problem.K.to_dense().double()
    assert torch.allclose(Kd, Kd.T, atol=1e-10)
    A = Kd[:problem.off_p, :problem.off_p].reshape(problem.n_u, 2, problem.n_u, 2)
    assert A[:, 0, :, 1].abs().max() < 1e-12          # components decouple
    A_p2 = problem.A.to_dense().double()
    assert torch.allclose(A[:, 0, :, 0], problem.mu * A_p2, atol=1e-9)
    assert Kd[problem.off_p:, problem.off_p:].abs().max() < 1e-14


def test_residual_masks_dirichlet_dofs(problem):
    u, p, f = _random_fields(problem)
    r = problem.residual(u, p, f)
    assert r.shape == (u.shape[0], problem.n_dofs)
    assert r[:, problem.dirichlet_mask].abs().max() < 1e-12


def test_pressure_gauge_projection(problem):
    _, p, _ = _random_fields(problem)
    pg = problem.project_pressure_gauge(p)
    w = problem.m_p_lumped
    assert (pg * w).sum(dim=-1).abs().max() < 1e-5
    # idempotent, and only removes a constant
    assert torch.allclose(pg, problem.project_pressure_gauge(pg), atol=1e-6)
    assert torch.allclose(p - pg, (p - pg)[:, :1].expand_as(p), atol=1e-5)


def test_residual_forward_backward(problem):
    u, p, f = _random_fields(problem)
    u.requires_grad_(True)
    p.requires_grad_(True)
    r = problem.residual(u, p, f)
    (0.5 * (r ** 2).sum()).backward()
    for g in (u.grad, p.grad):
        assert g is not None and torch.isfinite(g).all() and g.abs().max() > 0


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


def test_fem_reference_satisfies_bc_and_gauge(problem):
    torch.manual_seed(1)
    f = torch.randn(3, problem.n_u, 2)
    u, p = problem.fem_reference(f)
    assert u[:, problem.boundary_mask].abs().max() < 1e-10          # zero Dirichlet velocity
    assert (p * problem.m_p_lumped).sum(dim=-1).abs().max() < 1e-4  # zero-mean pressure
    assert torch.isfinite(u).all() and torch.isfinite(p).all()


def test_fem_reference_is_linear(problem):
    """Stokes is linear — the property the per-sample rescaling in ``StokesDataset`` relies on."""
    torch.manual_seed(2)
    f = torch.randn(1, problem.n_u, 2, dtype=torch.float32)
    u1, p1 = problem.fem_reference(f)
    u2, p2 = problem.fem_reference(2.5 * f)
    assert torch.allclose(u2, 2.5 * u1, atol=1e-5, rtol=1e-4)
    assert torch.allclose(p2, 2.5 * p1, atol=1e-5, rtol=1e-4)


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


# ----------------------------------------------------------------- the geometry

def test_boundary_covers_both_components_and_the_hole_is_empty(mesh):
    """Outer frame *and* obstacle, with nothing inside the hole.

    A coordinate test cannot mark a curved obstacle at all, and the problem would silently
    become flow *through* the body.
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


def test_mesh_cache_is_keyed_on_the_geometry(tmp_path):
    """Two geometries cannot share a cache entry: the file name encodes every parameter."""
    m1 = obstacle_mesh(chara_length=0.09, cache_dir=str(tmp_path))
    m2 = obstacle_mesh(chara_length=0.12, cache_dir=str(tmp_path))
    m3 = obstacle_mesh(chara_length=0.09, cache_dir=str(tmp_path))       # a cache hit
    assert len(list(tmp_path.glob("*.msh"))) == 2
    assert m1.n_points != m2.n_points
    assert m1.n_points == m3.n_points and torch.equal(m1.points, m3.points)


_SLOW_WRITER = """
import os, sys, tempfile, time
import tensorpils.meshing as m
real = m._write_obstacle_msh
def slow(path, *args):
    # Write the file in two halves with a pause, as a network filesystem may: until the second
    # half lands, `path` exists but holds a truncated mesh.
    with tempfile.TemporaryDirectory() as d:
        real(os.path.join(d, "full.msh"), *args)
        data = open(os.path.join(d, "full.msh"), "rb").read()
    with open(path, "wb") as fh:
        fh.write(data[:len(data) // 2]); fh.flush(); time.sleep(3.0); fh.write(data[len(data) // 2:])
m._write_obstacle_msh = slow
print(m.obstacle_mesh(chara_length=0.09, cache_dir=sys.argv[1]).n_points)
"""

_READER = """
import os, sys, time
import tensorpils.meshing as m
path = os.path.join(sys.argv[1], m._obstacle_cache_name(0.09, 2, (0.0, 1.0), (0.0, 1.0),
                                                        0.40, 0.50, 0.14))
t0 = time.time()
while not os.path.exists(path):          # start reading the moment the cache entry appears
    assert time.time() - t0 < 120, "the writer never produced the cache entry"
    time.sleep(0.01)
print(m.obstacle_mesh(chara_length=0.09, cache_dir=sys.argv[1]).n_points)
"""


def test_mesh_cache_is_never_read_half_written(tmp_path):
    """Runs that start together on a cold cache (a SLURM array) must each read a complete file.

    A second run that finds the cache entry while the first is still writing it used to read a
    truncated mesh and die in the reader; the cache now writes through a temporary file and an
    atomic rename, so the entry appears only once it is complete.
    """
    import os
    import subprocess
    import sys

    import tensorpils
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(tensorpils.__file__)))
    procs = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for code in (_SLOW_WRITER, _READER)]
    outs = [p.communicate(timeout=300) for p in procs]
    assert all(p.returncode == 0 for p in procs), [err[-800:] for _, err in outs]
    assert len({out.strip().splitlines()[-1] for out, _ in outs}) == 1
    assert len(list(tmp_path.iterdir())) == 1                      # no temporary file left behind


# ----------------------------------------------------------------- preconditioner / losses

def test_block_preconditioner_is_spd(problem):
    r"""``P`` is used as a **norm weight** in ``½ rᵀPr``, so it must be symmetric positive
    definite — otherwise the loss is unbounded below."""
    pre = _block_precond(problem)
    P = pre(torch.eye(problem.n_dofs))                     # rows: P e_i
    assert torch.isfinite(P).all()
    asym = (P - P.T).abs().max() / P.abs().max()
    assert asym < 1e-5, f"preconditioner is not symmetric (rel asym {asym:.2e})"
    ev = torch.linalg.eigvalsh(0.5 * (P + P.T))
    assert ev.min() > 0, f"preconditioner is not positive definite (lambda_min={ev.min():.3e})"


@pytest.mark.parametrize("loss_type", ["galerkin", "pls"])
def test_losses_forward_backward(problem, loss_type):
    from tensorpils.losses import build_stokes_loss
    pre = _block_precond(problem) if loss_type == "pls" else None
    crit = build_stokes_loss(loss_type, problem, precond=pre)
    u, p, f = _random_fields(problem)
    u.requires_grad_(True)
    p.requires_grad_(True)
    loss = crit(u, p, f)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert torch.isfinite(u.grad).all() and u.grad.abs().max() > 0
    assert torch.isfinite(p.grad).all() and p.grad.abs().max() > 0


def test_losses_vanish_at_the_reference(problem):
    """Both label-free losses are exactly zero at the discrete solution."""
    from tensorpils.losses import StokesGalerkinLoss, StokesPLSLoss
    torch.manual_seed(3)
    f = torch.randn(2, problem.n_u, 2)
    u, p = problem.fem_reference(f)
    ref_scale = StokesGalerkinLoss(problem)(torch.zeros_like(u), torch.zeros_like(p), f)
    for crit in (StokesGalerkinLoss(problem), StokesPLSLoss(problem, _block_precond(problem))):
        assert crit(u, p, f).item() < 1e-8 * max(ref_scale.item(), 1.0)


def test_pls_requires_a_preconditioner(problem):
    from tensorpils.losses import build_stokes_loss
    with pytest.raises(ValueError, match="requires a preconditioner"):
        build_stokes_loss("pls", problem, precond=None)


@requires_amgx
def test_block_preconditioner_takes_an_algebraic_velocity_block(problem):
    """``P = diag(Â⁻¹/μ, ω μ/diag(M_p))`` with an AMG velocity half, and it must be symmetric.

    ``P`` is used as a *norm*, so symmetry is not optional.
    """
    from tensorpils.preconditioners import AMGXPreconditioner, StokesBlockPreconditioner

    dev = "cuda"
    prob = problem.to(dev)
    amg = AMGXPreconditioner(prob.A, prob.boundary_mask, sweeps=2, device=dev)
    P = StokesBlockPreconditioner(prob, velocity_precond=amg, schur_omega=0.5).to(dev)

    torch.manual_seed(0)
    x = torch.randn(4, prob.n_dofs, device=dev)
    y = torch.randn(4, prob.n_dofs, device=dev)
    xPy = (x * P(y)).sum(dim=1)
    yPx = (y * P(x)).sum(dim=1)
    assert torch.allclose(xPy, yPx, rtol=1e-3, atol=1e-5), "P is not symmetric"
    assert ((x * P(x)).sum(dim=1) > 0).all(), "P is not positive definite on these vectors"
    prob.to("cpu")


# ----------------------------------------------------------------- dataset / trainer

def test_body_force_shape_and_decay(mesh):
    coeffs = torch.ones(2, 2, 3, 3)
    f = stokes_body_force(coeffs, mesh.points)
    assert f.shape == (2, mesh.points.shape[0], 2)
    assert torch.isfinite(f).all()


@pytest.fixture(scope="module")
def datasets():
    from tensorpils.data import create_stokes_datasets
    return create_stokes_datasets(n_train=4, n_val=2, n_test=4, K=6, chara_length=0.09, seed=0)


def test_dataset_items_are_node_vectors(datasets):
    tr, va, te = datasets
    f, u, p = tr[0]
    assert f.shape == (tr.n_u, 2) and u.shape == (tr.n_u, 2) and p.shape == (tr.n_p,)
    assert tr.mesh is te.mesh and tr.problem is te.problem
    # The per-sample normalisation is to unit velocity FE-L2.
    assert float(tr.problem.velocity_l2(u.unsqueeze(0))[0]) == pytest.approx(1.0, rel=1e-4)


class _DummyNodeNet(torch.nn.Module):
    """Stand-in for GAOT: ``forward_nodes`` maps ``[B, n_u, 2] -> [B, n_u, 3]``."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 3)

    def forward_nodes(self, x):
        return self.lin(x)


def test_trainer_and_viz_report_the_same_error(datasets, tmp_path):
    """The reported error must not depend on *which* code path computed the prediction.

    Regression test: ``viz`` once rebuilt the prediction pipeline itself and silently dropped the
    input/pressure scalings, reporting 4102 % where the trainer measured 7.7 %. Both now go
    through ``StokesTrainer._predict``.
    """
    from tensorpils.trainer import StokesTrainer

    tr, va, te = datasets
    trainer = StokesTrainer(
        model=_DummyNodeNet(), train_dataset=tr, val_dataset=va, test_dataset=te,
        loss_type="data", device="cpu", epochs=1, batch_size=2, output_dir=str(tmp_path))

    _, ru, rp = trainer._eval_loader(trainer.test_loader)
    med_u, _, _ = trainer.compute_error_distribution()
    # dataset-level vs per-sample-median aggregation differ slightly, but not by orders
    # of magnitude — which is what a dropped scaling looks like.
    assert med_u / 100.0 == pytest.approx(ru, rel=0.5)


def test_trainer_predict_applies_scalings_and_projections(datasets, tmp_path):
    from tensorpils.trainer import StokesTrainer

    tr, va, te = datasets
    trainer = StokesTrainer(
        model=_DummyNodeNet(), train_dataset=tr, val_dataset=va, test_dataset=te,
        loss_type="galerkin", device="cpu", epochs=1, batch_size=2, output_dir=str(tmp_path))
    assert trainer.f_scale > 1.0 and trainer.p_scale > 1.0     # the scales are non-trivial here

    f = torch.stack([tr[i][0] for i in range(2)])
    u, p = trainer._predict(f)
    assert u[:, trainer.problem.boundary_mask].abs().max() < 1e-6          # zero velocity BC
    assert (p * trainer.problem.m_p_lumped).sum(dim=-1).abs().max() < 1e-4  # zero-mean pressure
    assert np.isfinite(trainer.train_epoch())


@requires_amgx
def test_trainer_runs_an_epoch_and_tags_its_files(datasets, tmp_path):
    from tensorpils.gaot import GAOTModel
    from tensorpils.trainer import GAOTStokesTrainer

    tr, va, te = datasets
    torch.manual_seed(0)
    model = GAOTModel(grid_size=(32, 32), coords=tr.mesh.points.float(), in_channels=2,
                      out_channels=3, latent_grid=(8, 8), patch_size=2, lifting_channels=12,
                      magno_hidden=12, mlp_layers=2, transformer_hidden=24,
                      transformer_layers=2, num_heads=4)
    trainer = GAOTStokesTrainer(
        model=model, train_dataset=tr, val_dataset=va, test_dataset=te,
        loss_type="pls", batch_size=2, epochs=1, device="cuda", schur_omega=16,
        output_dir=str(tmp_path))
    trainer.save_checkpoints = False
    assert np.isfinite(trainer.train_epoch())
    assert np.isfinite(trainer.validate())

    prefix = trainer._file_prefix()
    assert prefix.startswith("gaot_stokes_pls_amg-s2-w16_")
    assert f"obstacle-n{tr.n_u}" in prefix


# ----------------------------------------------------------------- the baseline that follows

@pytest.mark.skipif(not torch.cuda.is_available(), reason="matches the other trainer test")
def test_pi_deeponet_runs_on_the_mesh(datasets, tmp_path):
    """The autodiff residual works off the grid, which is why PI-DeepONet is the baseline that
    can follow and PINO is not: PINO's residual is a finite-difference stencil and does not
    exist here, while this one differentiates through a coordinate trunk.
    """
    from tensorpils.baselines import DeepONetModel, PIDeepONetStokesTrainer

    tr, va, te = datasets
    torch.manual_seed(0)
    model = DeepONetModel(grid_size=(65, 65), in_channels=2, out_channels=3, p=16, width=32,
                          depth=3, trunk_fourier=16, coords=tr.mesh.points.float())
    assert model.grid_size is None
    with pytest.raises(RuntimeError, match="forward_at"):
        model(torch.randn(1, 2, 65, 65))

    trainer = PIDeepONetStokesTrainer(
        model=model, train_dataset=tr, val_dataset=va, test_dataset=te,
        batch_size=2, epochs=1, device="cuda", n_colloc=64, lambda_bc_pi=0.1,
        output_dir=str(tmp_path))
    trainer.save_checkpoints = False
    assert np.isfinite(trainer.train_epoch())
    assert np.isfinite(trainer.validate())
    assert "obstacle-n" in trainer._file_prefix()


def test_unstructured_deeponet_branch_reads_node_sensors():
    """``[B, N, C]`` mesh sensors must not be mistaken for an unbatched ``[C, H, W]`` image.

    The two are both 3-D, so the sensor set the model was built on has to decide; getting it
    wrong flattens the batch into one branch input and the shapes happen to line up often enough
    to be missed.
    """
    from tensorpils.baselines import DeepONetModel

    torch.manual_seed(0)
    pts = torch.rand(40, 2)
    m = DeepONetModel(grid_size=(9, 9), in_channels=2, out_channels=3, p=8, width=16, depth=2,
                      trunk_fourier=0, coords=pts)
    out = m.forward_at(torch.randn(5, 40, 2), pts)
    assert out.shape == (5, 40, 3), "the branch collapsed the batch"
    # Different samples must give different outputs -- the batch really is carried through.
    f = torch.randn(2, 40, 2)
    o = m.forward_at(f, pts)
    assert not torch.allclose(o[0], o[1])
