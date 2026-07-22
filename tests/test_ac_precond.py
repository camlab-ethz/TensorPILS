"""Tests for the preconditioned Allen–Cahn least-squares loss: the multigrid V-cycle on the
frozen Newton Jacobian J0 = a²A + cM (c = 1/dt + 3ε²), and its wiring into ACGalerkinLoss.

See notes/ac_autoregressive/ §"Preconditioning the least-squares loss".
"""

import pytest
import torch

from tensorpils.meshing import structured_quad_mesh
from tensorpils.physics import ACProblem
from tensorpils.preconditioners import GeometricMultigrid, build_preconditioner
from tensorpils.losses import build_ac_loss

G = 17


def _interior(mesh):
    return (~mesh.boundary_mask.bool()).nonzero(as_tuple=False).squeeze(1)


def test_poisson_default_operator_is_bare_stiffness():
    """Defaults (a2=1, c=0) must reproduce the bare Poisson stiffness — no behaviour change."""
    mesh = structured_quad_mesh(G, G)
    A = ACProblem(mesh).A.to_dense().double()
    mg = GeometricMultigrid(G, G, n_levels=3)              # a2=1, c=0
    assert mg.c == 0.0 and mg.a2 == 1.0
    idx = _interior(mesh)
    op = mg._A_sparse(0).to_dense().double()
    assert torch.allclose(op[idx][:, idx], A[idx][:, idx], rtol=1e-4, atol=1e-4)


def test_screened_operator_matches_assembled_a2A_plus_cM():
    """The finest-level MG operator interior block equals a²A + cM from the same assemblers."""
    mesh = structured_quad_mesh(G, G)
    prob = ACProblem(mesh)
    a2, c = 1.0, 1.0 / 0.01 + 3.0 * 2.0 ** 2              # = 112 (dt=0.01, eps=2)
    mg = GeometricMultigrid(G, G, n_levels=3, a2=a2, c=c)
    A = prob.A.to_dense().double()
    M = prob.M.to_dense().double()
    idx = _interior(mesh)
    ref = (a2 * A + c * M)[idx][:, idx]
    got = mg._A_sparse(0).to_dense().double()[idx][:, idx]
    assert torch.allclose(got, ref, rtol=1e-3, atol=1e-3)


def test_vcycle_is_approximate_inverse_of_screened_operator():
    """One-plus V-cycles as a stationary iteration converge for a²A + cM (genuine P ≈ J0^-1)."""
    mg = GeometricMultigrid(33, 33, n_levels=4, a2=1.0, c=112.0)
    J = mg._A_sparse(0).to_dense()
    mask = structured_quad_mesh(33, 33).boundary_mask.bool()
    torch.manual_seed(0)
    x = torch.randn(2, 33 * 33)
    x[:, mask] = 0.0
    r = (J @ x.T).T
    e = torch.zeros_like(x)
    for _ in range(5):
        e = e + mg(r - (J @ e.T).T)
    rel = ((e - x).norm(dim=1) / x.norm(dim=1)).mean().item()
    assert rel < 1e-2                                      # 5 V-cycles well below 1%


def test_preconditioned_ac_loss_runs_differs_and_is_differentiable():
    mesh = structured_quad_mesh(G, G)
    prob = ACProblem(mesh)
    a, eps, dt = 1.0, 2.0, 0.01
    P = build_preconditioner("multigrid", prob, (G, G), mg_levels=3,
                             mg_a2=a * a, mg_c=1.0 / dt + 3.0 * eps * eps)
    common = dict(a=a, eps=eps, dt=dt, integrator="convex_concave", form="galerkin")
    bare = build_ac_loss(prob, **common)
    prec = build_ac_loss(prob, precond=P, **common)

    torch.manual_seed(0)
    seq = torch.randn(3, 2, G * G)                         # [B, L=2, N]
    lb, lp = bare(seq), prec(seq)
    assert torch.isfinite(lb) and torch.isfinite(lp)
    assert not torch.allclose(lb, lp)                      # P reshapes the residual -> different value

    seq2 = torch.randn(3, 2, G * G, requires_grad=True)
    prec(seq2).backward()                                  # autograd flows through the V-cycle
    assert seq2.grad is not None and torch.isfinite(seq2.grad).all()


def test_precond_rejected_for_min_movement():
    mesh = structured_quad_mesh(G, G)
    prob = ACProblem(mesh)
    P = build_preconditioner("multigrid", prob, (G, G), mg_levels=3, mg_c=112.0)
    with pytest.raises(ValueError):
        build_ac_loss(prob, a=1.0, eps=2.0, dt=0.01, form="min_movement", precond=P)
