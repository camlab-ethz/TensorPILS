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
    """The convex_concave reference (the default) zeroes the convex-splitting residual.

    Uses the production ``ACProblem.residual(integrator="convex_concave")`` — the exact residual
    the least-squares physics loss is built on — and cross-checks it against an independent
    hand-derived formula."""
    mesh = structured_quad_mesh(17, 17)
    prob = ACProblem(mesh)
    u0 = _ic(mesh, K=3, dtype=torch.float64).unsqueeze(0)          # [1, N]
    a, eps, dt, n_steps = 1.0, 2.0, 1e-3, 5
    traj = prob.fem_reference(u0, a=a, eps=eps, dt=dt, n_steps=n_steps,
                              newton_tol=1e-10, newton_max=50,
                              integrator="convex_concave")         # [1, T+1, N]
    worst = 0.0
    for k in range(n_steps):
        r = prob.residual(traj[:, k], traj[:, k + 1], a=a, eps=eps, dt=dt,
                          integrator="convex_concave")
        # production residual must match the independent hand-derived one
        r_ref = _cc_residual(prob, traj[:, k], traj[:, k + 1], a=a, eps=eps, dt=dt)
        assert torch.allclose(r, r_ref, atol=1e-10), "production CC residual != hand-derived formula"
        worst = max(worst, r.norm().item())
    assert worst < 1e-5, f"CC reference should zero the CC residual, got {worst:.2e}"


def test_ac_mm_objective_gradient():
    """∇J of the minimizing-movement objective equals the convex-concave step residual with the
    lumped mass diag(M·1) on the cubic — so J's minimiser is the convex-concave step."""
    from tensorpils.physics import apply_zero_boundary
    mesh = structured_quad_mesh(11, 11)
    prob = ACProblem(mesh)
    mask = prob.boundary_mask
    a, eps, dt = 1.0, 2.0, 0.01
    torch.manual_seed(0)
    N = prob.n_nodes
    uc = apply_zero_boundary(torch.randn(N, dtype=torch.float64), mask)
    un = apply_zero_boundary(torch.randn(N, dtype=torch.float64), mask).requires_grad_(True)

    J = prob.mm_objective(uc, un, a=a, eps=eps, dt=dt)
    g, = torch.autograd.grad(J, un)
    g = apply_zero_boundary(g, mask)

    m1 = prob._spmm(prob.M, torch.ones(N, dtype=torch.float64))       # lumped mass M·1
    analytic = apply_zero_boundary(
        prob._spmm(prob.M, (un.detach() - uc) / dt) + (a * a) * prob._spmm(prob.A, un.detach())
        + (eps * eps) * (m1 * un.detach() ** 3) - (eps * eps) * prob._spmm(prob.M, uc), mask)
    assert torch.allclose(g, analytic, atol=1e-9), "grad(J) != lumped-mass convex-concave residual"


def test_ac_min_movement_loss_runs():
    """ACMinMovementLoss returns a finite scalar and is differentiable over a rollout."""
    from tensorpils.losses import build_ac_loss
    mesh = structured_quad_mesh(11, 11)
    prob = ACProblem(mesh)
    crit = build_ac_loss(prob, a=1.0, eps=2.0, dt=0.005, discount=0.9, form="min_movement")
    seq = torch.randn(3, 4, prob.n_nodes, dtype=torch.float64, requires_grad=True)
    loss = crit(seq)
    loss.backward()
    assert torch.isfinite(loss) and seq.grad is not None and torch.isfinite(seq.grad).all()


def test_ac_min_movement_detaches_proximal_centre():
    """The previous frame u^k is the fixed proximal centre of u^{k+1}=argmin_u J(u^k;u), so the
    loss must not backprop into it. Otherwise every rollout frame receives a spurious ∂J/∂u^k
    gradient that does NOT vanish at the correct dynamics, distorting the trained trajectory."""
    from tensorpils.losses import build_ac_loss
    prob = ACProblem(structured_quad_mesh(11, 11))
    crit = build_ac_loss(prob, a=1.0, eps=2.0, dt=0.0025, discount=0.9, form="min_movement")
    seq = torch.randn(2, 4, prob.n_nodes, dtype=torch.float64, requires_grad=True)  # [B, L=4, N]
    crit(seq).backward()
    g = seq.grad.reshape(2, 4, -1).norm(dim=(0, 2))          # per-frame gradient norm, frames 0..3
    assert g[0].item() == 0.0, "seed frame (pure proximal centre) must receive no gradient"
    assert (g[1:] > 0).all(), "model-predicted frames must still be trained"


def test_fem_reference_v2_matches_v1():
    """Sparse fem_reference_v2 reproduces dense fem_reference AND always solves in float64.

    Differential test guarding the future swap (v2 must match v1 to solver tolerance), plus a
    check that the reference precision is decoupled from the caller's dtype: a float32 call still
    solves in float64 and only casts the result down."""
    mesh = structured_quad_mesh(17, 17)
    prob = ACProblem(mesh)
    u0 = torch.stack([_ic(mesh, K=3, dtype=torch.float64),
                      _ic(mesh, K=4, dtype=torch.float64)], dim=0)     # B=2, float64
    for integ in ("convex_concave", "backward_euler"):
        kw = dict(a=1.0, eps=8.0, dt=0.0025, n_steps=5, newton_tol=1e-10,
                  newton_max=50, integrator=integ)
        t1 = prob.fem_reference(u0, chunk=2, **kw)                     # dense, float64
        t2 = prob.fem_reference_v2(u0, chunk=2, **kw)                  # sparse, float64
        assert t2.dtype == torch.float64
        assert torch.allclose(t1, t2, atol=1e-6), \
            f"fem_reference_v2 != fem_reference ({integ}): {(t1 - t2).abs().max():.2e}"

        # float32 input: returned in float32 but solved in float64, so (cast up) it matches the
        # float64 call to ~float32 rounding -- NOT the ~1e-6 floor a genuine float32 solve hits.
        t32 = prob.fem_reference_v2(u0.float(), chunk=2, **kw)
        assert t32.dtype == torch.float32
        assert torch.allclose(t32.double(), t2, atol=1e-5), \
            f"float32-input v2 lost precision ({integ}): {(t32.double() - t2).abs().max():.2e}"


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
