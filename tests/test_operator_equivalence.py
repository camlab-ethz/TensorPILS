"""Correctness checks for the TensorMesh-backed Poisson operator and the analytical data.

These verify, without importing the standalone reference script or the FNO model
(so no ``neuralop`` needed), that:

* the assembled stiffness ``A`` is the correct bilinear-quad Laplacian (symmetric,
  constant null space, interior-positive-definite, O(h^2)-consistent);
* the mass matrix ``M`` integrates correctly (``sum(M) == area``);
* the analytical fields match the standalone's spectrum (``r = -0.5`` convention);
* node ordering / boundary mask are as expected;
* every physics-informed loss runs forward + backward (incl. the multigrid ones).
"""

import math

import numpy as np
import torch
import pytest

from tensorpils.meshing import structured_quad_mesh, node_to_grid, grid_to_node
from tensorpils.physics import PoissonProblem
from tensorpils.losses import build_loss
from tensorpils.preconditioners import GeometricMultigrid
from tensormesh.dataset import PoissonMultiFrequency


# ----------------------------- standalone reference (numpy) -----------------------------
def _ref_source(points, a, r=0.5):
    """Verbatim port of the standalone script's poisson_source (a: [N,K,K])."""
    K = a.shape[-1]
    j, i = np.meshgrid(np.arange(1, K + 1), np.arange(1, K + 1))
    coeff = a * (i * i + j * j) ** r
    coeff = coeff[:, None, ...]
    x = points[:, 0][None, :, None, None]
    y = points[:, 1][None, :, None, None]
    return np.pi / K / K * (coeff * np.sin(np.pi * i * x) * np.sin(np.pi * j * y)).sum((-2, -1))


def _ref_solution(points, a, r=0.5):
    """Verbatim port of the standalone script's poisson_solution."""
    K = a.shape[-1]
    j, i = np.meshgrid(np.arange(1, K + 1), np.arange(1, K + 1))
    coeff = a * (i * i + j * j) ** (r - 1)
    coeff = coeff[:, None, ...]
    x = points[:, 0][None, :, None, None]
    y = points[:, 1][None, :, None, None]
    return 1.0 / np.pi / K / K * (coeff * np.sin(np.pi * i * x) * np.sin(np.pi * j * y)).sum((-2, -1))


# ------------------------------------- fixtures ----------------------------------------
@pytest.fixture(scope="module")
def problem17():
    mesh = structured_quad_mesh(17, 17)
    return mesh, PoissonProblem(mesh)


# ------------------------------------- tests -------------------------------------------
def test_node_ordering_and_boundary_mask():
    nx, ny = 5, 4
    mesh = structured_quad_mesh(nx, ny)
    assert mesh.n_points == nx * ny
    A = PoissonProblem(mesh).A
    covered = torch.unique(torch.cat([A.row, A.col])).sort().values
    assert covered.tolist() == list(range(nx * ny)), "A must touch every node index"
    # boundary frame
    mask = mesh.boundary_mask.reshape(ny, nx)
    assert mask[0].all() and mask[-1].all() and mask[:, 0].all() and mask[:, -1].all()
    assert not mask[1:-1, 1:-1].any()


def test_grid_node_roundtrip():
    nx, ny = 7, 5
    f = torch.randn(3, nx * ny)
    assert torch.equal(grid_to_node(node_to_grid(f, nx, ny), nx, ny), f)


def test_stiffness_is_laplacian(problem17):
    mesh, prob = problem17
    A = prob.A.to_dense().double()
    assert torch.allclose(A, A.T, atol=1e-10), "A must be symmetric"
    assert A.sum(1).abs().max() < 1e-9, "constant vector must be in the null space"
    inner = ~mesh.boundary_mask
    ev = torch.linalg.eigvalsh(A[inner][:, inner])
    assert ev.min() > 0, "interior stiffness must be positive-definite (correct winding)"


def test_mass_matrix_area(problem17):
    _, prob = problem17
    M = prob.M.to_dense().double()
    assert torch.allclose(M, M.T, atol=1e-10)
    assert math.isclose(M.sum().item(), 1.0, abs_tol=1e-9), "sum(M) == unit-square area"


def test_galerkin_consistency_O_h2():
    """Galerkin residual of the analytical solution must decay like O(h^2)."""
    torch.manual_seed(0)
    a = torch.zeros(4, 4, 4).uniform_(-1, 1)
    eq = PoissonMultiFrequency(a=a, r=-0.5)
    rels = []
    for n in (9, 17, 33):
        mesh = structured_quad_mesh(n, n)
        prob = PoissonProblem(mesh)
        pts = mesh.points
        f = eq.source_term(pts, domain="rectangle").double()
        u = eq.solution(pts).double()
        r = prob.residual(u, f)
        b = prob.load_vector(f)
        rels.append((r.norm(dim=1) / b.norm(dim=1)).mean().item())
    # each halving of h should cut the residual by ~4x; require >3x to be safe
    assert rels[0] / rels[1] > 3.0 and rels[1] / rels[2] > 3.0, rels


def test_r_convention_matches_standalone():
    """PoissonMultiFrequency(r=-0.5) reproduces the standalone's source/solution (r=0.5)."""
    mesh = structured_quad_mesh(13, 13)
    pts = mesh.points
    torch.manual_seed(2)
    a = torch.zeros(2, 4, 4).uniform_(-1, 1)
    eq = PoissonMultiFrequency(a=a, r=-0.5)
    f_tm = eq.source_term(pts, domain="rectangle").double().numpy()
    u_tm = eq.solution(pts).double().numpy()
    pts_np = pts.double().numpy()
    a_np = a.double().numpy()
    assert np.allclose(f_tm, _ref_source(pts_np, a_np, r=0.5), atol=1e-10)
    assert np.allclose(u_tm, _ref_solution(pts_np, a_np, r=0.5), atol=1e-10)


@pytest.mark.parametrize("loss_type", ["galerkin", "pls"])
def test_losses_forward_backward(loss_type):
    """Every FEM-based loss runs forward + backward and yields finite gradients."""
    nx = ny = 17
    mesh = structured_quad_mesh(nx, ny)
    prob = PoissonProblem(mesh)
    mg = GeometricMultigrid(nx, ny, n_levels=3) if loss_type == "pls" else None
    crit = build_loss(loss_type, prob, precond=mg)
    torch.manual_seed(3)
    u = torch.randn(4, nx * ny, requires_grad=True)
    f = torch.randn(4, nx * ny)
    loss = crit(u, f)
    assert torch.isfinite(loss)
    loss.backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()
