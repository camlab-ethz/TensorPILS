"""Correctness checks for the TensorMesh-backed wave operator and the analytical data.

These verify, without importing the FNO model (so no ``neuralop`` needed), that:

* ``WaveMultiFrequency`` produces the right shapes and honours the zero Dirichlet BC;
* the central-difference wave residual is boundary-masked and differentiable;
* the residual of the analytical trajectory is O(h^2 + dt^2)-consistent (decays under
  joint space/time refinement);
* the wave Galerkin loss runs forward + backward with finite gradients.
"""

import torch
import pytest

from tensorpils.meshing import structured_quad_mesh
from tensorpils.physics import WaveProblem, apply_zero_boundary
from tensorpils.losses import build_wave_loss
from tensormesh.dataset import WaveMultiFrequency


# ------------------------------------- fixtures ----------------------------------------
@pytest.fixture(scope="module")
def wave17():
    mesh = structured_quad_mesh(17, 17)
    return mesh, WaveProblem(mesh)


# ------------------------------------- tests -------------------------------------------
def test_wave_dataset_shapes_and_boundary():
    """initial_condition / solution return [N] (or [batch, N]) and vanish on the boundary."""
    mesh = structured_quad_mesh(13, 13)
    pts = mesh.points
    mask = mesh.boundary_mask.bool()

    eq = WaveMultiFrequency(K=4, c=1.0, r=0.5)          # a is [K, K] -> single sample
    u0 = eq.initial_condition(pts)
    ut = eq.solution(pts, t=0.1)
    assert u0.shape == (mesh.n_points,)
    assert ut.shape == (mesh.n_points,)
    assert u0[mask].abs().max() < 1e-6, "IC must be zero on the Dirichlet boundary"
    assert ut[mask].abs().max() < 1e-6, "solution must be zero on the Dirichlet boundary"

    # batched a -> [batch, N]
    a = torch.zeros(3, 4, 4).uniform_(-1, 1)
    eqb = WaveMultiFrequency(a=a, c=1.0, r=0.5)
    traj = eqb.solution(pts, t=0.05)
    assert traj.shape == (3, mesh.n_points)


def test_wave_residual_boundary_masked(wave17):
    """The wave residual is zero on the Dirichlet boundary regardless of the inputs."""
    mesh, prob = wave17
    n = prob.n_nodes
    mask = prob.boundary_mask
    torch.manual_seed(0)
    up, uc, un = (torch.randn(4, n) for _ in range(3))
    r = prob.residual(up, uc, un, c=1.0, dt=0.01)
    assert r[:, mask].abs().max() < 1e-12


def test_wave_residual_forward_backward(wave17):
    """Residual is finite and differentiable w.r.t. all three time levels."""
    _, prob = wave17
    n = prob.n_nodes
    torch.manual_seed(1)
    up = torch.randn(4, n, requires_grad=True)
    uc = torch.randn(4, n, requires_grad=True)
    un = torch.randn(4, n, requires_grad=True)
    loss = (prob.residual(up, uc, un, c=1.0, dt=0.01) ** 2).mean()
    assert torch.isfinite(loss)
    loss.backward()
    for g in (up.grad, uc.grad, un.grad):
        assert g is not None and torch.isfinite(g).all()


def test_wave_residual_consistency_O_h2_dt2():
    """Residual of the analytical trajectory decays under joint space+time refinement."""
    torch.manual_seed(2)
    a = torch.zeros(3, 3).uniform_(-1, 1)               # K=3, single sample
    c, r, t = 1.0, 0.5, 0.1
    eq = WaveMultiFrequency(a=a, c=c, r=r)

    rels = []
    for n, dt in ((17, 0.01), (33, 0.005), (65, 0.0025)):
        mesh = structured_quad_mesh(n, n)
        prob = WaveProblem(mesh)
        pts = mesh.points
        up = eq.solution(pts, t - dt).double()
        uc = eq.solution(pts, t).double()
        un = eq.solution(pts, t + dt).double()
        res = prob.residual(up, uc, un, c=c, dt=dt)
        # normalize by a representative O(1) term (the stiffness contribution)
        stiff = apply_zero_boundary((c * c) * prob._spmm(prob.A, apply_zero_boundary(uc, prob.boundary_mask)),
                                    prob.boundary_mask)
        rels.append((res.norm() / stiff.norm()).item())

    # halving h and dt together should cut the residual by ~4x; require >3x to be safe
    assert rels[0] / rels[1] > 3.0 and rels[1] / rels[2] > 3.0, rels


def test_wave_galerkin_loss_forward_backward(wave17):
    """WaveGalerkinLoss runs over a rollout sequence and yields finite gradients."""
    _, prob = wave17
    n = prob.n_nodes
    crit = build_wave_loss(prob, c=1.0, dt=0.01, discount=0.9)
    torch.manual_seed(3)
    seq = torch.randn(4, 5, n, requires_grad=True)      # [B, L=5, N] -> 3 triples
    loss = crit(seq)
    assert torch.isfinite(loss)
    loss.backward()
    assert seq.grad is not None and torch.isfinite(seq.grad).all()
