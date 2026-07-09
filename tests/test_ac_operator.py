"""Correctness checks for the TensorMesh-backed Allen–Cahn operator and reference solver.

These verify, without importing the FNO model (so no ``neuralop`` needed), that:

* the backward-Euler Allen–Cahn residual is boundary-masked and differentiable;
* the FEM implicit-Euler + Newton reference trajectory is **self-consistent** — it zeroes the
  same residual the physics loss uses (there is no analytical AC solution to compare against);
* the reference stays on the zero Dirichlet boundary, is finite/bounded, and dissipates energy;
* the wave-style AC Galerkin loss runs forward + backward with finite gradients.
"""

import torch
import pytest

from tensorpils.meshing import structured_quad_mesh
from tensorpils.physics import ACProblem, apply_zero_boundary
from tensorpils.losses import build_ac_loss
from tensormesh.dataset import WaveMultiFrequency


# ------------------------------------- fixtures ----------------------------------------
@pytest.fixture(scope="module")
def ac17():
    mesh = structured_quad_mesh(17, 17)
    return mesh, ACProblem(mesh)


def _ic(mesh, K=3, r=0.5, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    a = torch.zeros(K, K).uniform_(-1, 1)
    return WaveMultiFrequency(a=a, r=r).initial_condition(mesh.points).to(dtype)


# ------------------------------------- tests -------------------------------------------
def test_ac_residual_boundary_masked(ac17):
    """The AC residual vanishes on the Dirichlet boundary regardless of the inputs."""
    _, prob = ac17
    n = prob.n_nodes
    mask = prob.boundary_mask
    torch.manual_seed(0)
    uc, un = torch.randn(4, n), torch.randn(4, n)
    r = prob.residual(uc, un, a=1.0, eps=2.0, dt=1e-3)
    assert r[:, mask].abs().max() < 1e-12


def test_ac_residual_forward_backward(ac17):
    """Residual is finite and differentiable w.r.t. both time levels (nonlinear reaction)."""
    _, prob = ac17
    n = prob.n_nodes
    torch.manual_seed(1)
    uc = torch.randn(4, n, requires_grad=True)
    un = torch.randn(4, n, requires_grad=True)
    loss = (prob.residual(uc, un, a=1.0, eps=2.0, dt=1e-3) ** 2).mean()
    assert torch.isfinite(loss)
    loss.backward()
    for g in (uc.grad, un.grad):
        assert g is not None and torch.isfinite(g).all()


def test_ac_reference_self_consistency():
    """The backward-Euler reference zeroes the (backward-Euler) AC residual on every pair.

    ``ACProblem.residual`` is the fully-implicit residual, so this checks the reference built
    with the matching ``integrator="backward_euler"`` (not the convex_concave default)."""
    mesh = structured_quad_mesh(17, 17)
    prob = ACProblem(mesh)
    u0 = _ic(mesh, K=3, dtype=torch.float64).unsqueeze(0)          # [1, N]
    a, eps, dt, n_steps = 1.0, 2.0, 1e-3, 5
    traj = prob.fem_reference(u0, a=a, eps=eps, dt=dt, n_steps=n_steps,
                              newton_tol=1e-10, newton_max=50,
                              integrator="backward_euler")          # [1, T+1, N]
    worst = 0.0
    for k in range(n_steps):
        r = prob.residual(traj[:, k], traj[:, k + 1], a=a, eps=eps, dt=dt)
        worst = max(worst, r.norm().item())
    assert worst < 1e-5, f"reference should zero the discrete residual, got {worst:.2e}"


def _cc_residual(prob, u_prev, u_next, a, eps, dt):
    """Eyre convex-splitting AC residual: M(uⁿ⁺¹-uⁿ)/dt + a²A uⁿ⁺¹ + ε²M((uⁿ⁺¹)³ - uⁿ).

    Cubic implicit, linear reaction explicit at uⁿ; boundary-masked like ``ACProblem.residual``."""
    from tensorpils.physics import apply_zero_boundary
    mask = prob.boundary_mask
    up = apply_zero_boundary(u_prev, mask)
    un = apply_zero_boundary(u_next, mask)
    r = prob._spmm(prob.M, (un - up) / dt) \
        + (a * a) * prob._spmm(prob.A, un) \
        + (eps * eps) * prob._spmm(prob.M, un ** 3 - up)
    return apply_zero_boundary(r, mask)


def test_ac_reference_self_consistency_convex_concave():
    """The convex_concave reference (the default) zeroes the convex-splitting residual."""
    mesh = structured_quad_mesh(17, 17)
    prob = ACProblem(mesh)
    u0 = _ic(mesh, K=3, dtype=torch.float64).unsqueeze(0)          # [1, N]
    a, eps, dt, n_steps = 1.0, 2.0, 1e-3, 5
    traj = prob.fem_reference(u0, a=a, eps=eps, dt=dt, n_steps=n_steps,
                              newton_tol=1e-10, newton_max=50,
                              integrator="convex_concave")         # [1, T+1, N]
    worst = 0.0
    for k in range(n_steps):
        r = _cc_residual(prob, traj[:, k], traj[:, k + 1], a=a, eps=eps, dt=dt)
        worst = max(worst, r.norm().item())
    assert worst < 1e-5, f"CC reference should zero the CC residual, got {worst:.2e}"


def test_ac_reference_sanity():
    """Reference trajectory: zero on boundary, finite/bounded, and net energy non-increasing."""
    mesh = structured_quad_mesh(21, 21)
    prob = ACProblem(mesh)
    mask = prob.boundary_mask
    u0 = _ic(mesh, K=4, dtype=torch.float64).unsqueeze(0)
    a, eps, dt, n_steps = 1.0, 2.0, 1e-3, 10
    traj = prob.fem_reference(u0, a=a, eps=eps, dt=dt, n_steps=n_steps,
                              newton_tol=1e-10, newton_max=50)[0]    # [T+1, N]
    assert torch.isfinite(traj).all()
    assert traj[:, mask].abs().max() < 1e-12, "boundary must stay zero"
    assert traj.abs().max() < 10.0, "solution must stay bounded"
    energies = torch.stack([prob.energy(traj[k], a=a, eps=eps) for k in range(n_steps + 1)])
    assert energies[-1] <= energies[0] + 1e-6, "Allen–Cahn is a gradient flow (energy dissipates)"


def test_ac_galerkin_loss_forward_backward(ac17):
    """ACGalerkinLoss runs over a rollout sequence and yields finite gradients."""
    _, prob = ac17
    n = prob.n_nodes
    crit = build_ac_loss(prob, a=1.0, eps=2.0, dt=1e-3, discount=0.9)
    torch.manual_seed(3)
    seq = torch.randn(4, 4, n, requires_grad=True)      # [B, L=4, N] -> 3 pairs
    loss = crit(seq)
    assert torch.isfinite(loss)
    loss.backward()
    assert seq.grad is not None and torch.isfinite(seq.grad).all()
