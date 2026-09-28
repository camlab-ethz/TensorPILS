"""Correctness checks for the TensorMesh-backed Allen–Cahn operator and reference solver.

These verify, without importing the FNO model (so no ``neuralop`` needed), that:

* the convex–concave Allen–Cahn step residual is boundary-masked and differentiable;
* the FEM convex–concave + Newton reference trajectory is **self-consistent** — it zeroes the
  same residual the physics loss uses (there is no analytical AC solution to compare against);
* the reference stays on the zero Dirichlet boundary, is finite/bounded, and dissipates energy;
* the least-squares loss detaches the previous frame and the rollout is pushforward.
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


def _cc_residual(prob, u_prev, u_next, a, eps, dt):
    """Eyre convex-splitting AC residual: M(uⁿ⁺¹-uⁿ)/dt + a²A uⁿ⁺¹ + ε²M((uⁿ⁺¹)³ - uⁿ).

    Cubic implicit, linear reaction explicit at uⁿ; boundary-masked like ``ACProblem.residual``."""
    mask = prob.boundary_mask
    up = apply_zero_boundary(u_prev, mask)
    un = apply_zero_boundary(u_next, mask)
    r = prob._spmm(prob.M, (un - up) / dt) \
        + (a * a) * prob._spmm(prob.A, un) \
        + (eps * eps) * prob._spmm(prob.M, un ** 3 - up)
    return apply_zero_boundary(r, mask)


def _energy(prob, u, a, eps):
    r"""Ginzburg–Landau energy :math:`\tfrac12 a^2 u^\top A u + \epsilon^2 \tfrac14 u^\top M (u^2-1)^2`."""
    Au = prob._spmm(prob.A, u)
    well = 0.25 * (u * u - 1.0) ** 2
    return 0.5 * (a * a) * (u * Au).sum() + (eps * eps) * prob._spmm(prob.M, well).sum()


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
    """The reference zeroes the convex-splitting residual on every pair.

    Uses the production ``ACProblem.residual`` — the exact residual the least-squares physics
    loss is built on — and cross-checks it against an independent hand-derived formula."""
    mesh = structured_quad_mesh(17, 17)
    prob = ACProblem(mesh)
    u0 = _ic(mesh, K=3, dtype=torch.float64).unsqueeze(0)          # [1, N]
    a, eps, dt, n_steps = 1.0, 2.0, 1e-3, 5
    traj = prob.fem_reference(u0, a=a, eps=eps, dt=dt, n_steps=n_steps,
                              newton_tol=1e-10, newton_max=50)     # [1, T+1, N]
    worst = 0.0
    for k in range(n_steps):
        r = prob.residual(traj[:, k], traj[:, k + 1], a=a, eps=eps, dt=dt)
        # production residual must match the independent hand-derived one
        r_ref = _cc_residual(prob, traj[:, k], traj[:, k + 1], a=a, eps=eps, dt=dt)
        assert torch.allclose(r, r_ref, atol=1e-10), "production CC residual != hand-derived formula"
        worst = max(worst, r.norm().item())
    assert worst < 1e-5, f"CC reference should zero the CC residual, got {worst:.2e}"


def test_fem_reference_always_solves_in_float64():
    """The reference precision is decoupled from the caller's dtype: a float32 call still solves
    in float64 and only casts the result down, so (cast up) it matches the float64 call to
    ~float32 rounding -- NOT the ~1e-6 floor a genuine float32 solve hits."""
    mesh = structured_quad_mesh(17, 17)
    prob = ACProblem(mesh)
    u0 = torch.stack([_ic(mesh, K=3, dtype=torch.float64),
                      _ic(mesh, K=4, dtype=torch.float64)], dim=0)     # B=2, float64
    kw = dict(a=1.0, eps=8.0, dt=0.0025, n_steps=5, newton_tol=1e-10, newton_max=50)
    t64 = prob.fem_reference(u0, chunk=2, **kw)
    assert t64.dtype == torch.float64
    t32 = prob.fem_reference(u0.float(), chunk=2, **kw)
    assert t32.dtype == torch.float32
    assert torch.allclose(t32.double(), t64, atol=1e-5), \
        f"float32-input reference lost precision: {(t32.double() - t64).abs().max():.2e}"


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
    energies = torch.stack([_energy(prob, traj[k], a=a, eps=eps) for k in range(n_steps + 1)])
    assert energies[-1] <= energies[0] + 1e-6, "Allen–Cahn is a gradient flow (energy dissipates)"


def test_ac_galerkin_loss_forward_backward(ac17):
    """ACGalerkinLoss runs over a rollout sequence and yields finite gradients."""
    _, prob = ac17
    n = prob.n_nodes
    crit = build_ac_loss(prob, a=1.0, eps=2.0, dt=1e-3)
    torch.manual_seed(3)
    seq = torch.randn(4, 4, n, requires_grad=True)      # [B, L=4, N] -> 3 pairs
    loss = crit(seq)
    assert torch.isfinite(loss)
    loss.backward()
    assert seq.grad is not None and torch.isfinite(seq.grad).all()


# ------------------------------- pushforward training ----------------------------------

def test_ac_loss_detaches_the_previous_frame():
    """In R(u^k, u^{k+1}) the previous frame is detached: the seed frame, which only ever appears
    as a previous frame, receives no gradient, while every predicted frame is trained."""
    prob = ACProblem(structured_quad_mesh(11, 11))
    crit = build_ac_loss(prob, a=1.0, eps=4.0, dt=0.0025)
    seq = torch.randn(2, 4, prob.n_nodes, dtype=torch.float64, requires_grad=True)  # [B, L=4, N]
    crit(seq).backward()
    g = seq.grad.reshape(2, 4, -1).norm(dim=(0, 2))          # per-frame gradient norm
    assert g[0].item() == 0.0, "the seed frame must receive no gradient"
    assert (g[1:] > 0).all(), "model-predicted frames must still be trained"


def test_rollout_is_pushforward():
    """Each rollout frame is a one-step function of a detached input: the gradient of a loss on
    the last frame equals that of one model application to the (detached) previous frame."""
    from tensorpils.trainer import RolloutTrainer
    torch.manual_seed(0)

    class Stub:                                               # minimal carrier for RolloutTrainer._rollout
        grid_size = (8, 8)
        n_seed_frames = 1
        rollout_steps = 3

        def _project_zero_bc(self, x):
            return x
    stub = Stub()
    stub.model = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False).double()
    traj = torch.randn(2, 4, 8, 8, dtype=torch.float64)      # [B, T+1, H, W]

    preds, _ = RolloutTrainer._rollout(stub, traj)
    (preds[:, -1] ** 2).mean().backward()                    # loss on the LAST frame only
    g_rollout = stub.model.weight.grad.clone()

    stub.model.zero_grad()
    prev = preds[:, -2].detach().unsqueeze(1)                # [B, 1, H, W]
    (stub.model(prev).squeeze(1) ** 2).mean().backward()
    assert torch.allclose(g_rollout, stub.model.weight.grad, atol=1e-12)
