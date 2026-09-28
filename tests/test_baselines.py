"""Tests for the PINO / PI-DeepONet baselines (``tensorpils.baselines``).

The load-bearing test is :func:`test_fd_and_fem_residuals_have_the_same_discretization_error`.
The whole branch exists to compare a strong-form finite-difference / autodiff residual against
our weak-form FEM residual, and that comparison is only meaningful if the two are *the same
PDE at the same accuracy*. If they were not, a difference in training results could be blamed
on the discretisation rather than on the loss, which is precisely the confound the experiment
is meant to remove.

The rest guard the things that would quietly bias the baseline: the autodiff Laplacian being
per-sample rather than batch-summed, the DeepONet's output scale at initialisation, the hard
BC actually being hard, and run filenames not colliding across architectures.
"""

import math

import pytest
import torch

from tensorpils.baselines import (DeepONetModel, MollifiedModel, ZeroBoundaryModel,
                                  boundary_mask_grid, PINOACLoss,
                                  PINOPoissonLoss, PIDeepONetACLoss,
                                  PIDeepONetStokesLoss, autodiff_stokes,
                                  autodiff_laplacian, mollifier_grid, rel_lp)
from tensorpils.baselines.trainers import _arch_tag
from tensorpils.meshing import node_to_grid, structured_quad_mesh
from tensorpils.physics import ACProblem, PoissonProblem

try:
    from tensormesh.dataset.equation.poisson import PoissonMultiFrequency
except ImportError:  # pragma: no cover
    PoissonMultiFrequency = None

pytestmark = pytest.mark.skipif(PoissonMultiFrequency is None,
                                reason="tensormesh is required")


def _analytic_pair(n, n_samples=2, K=4, seed=0):
    """``(u_grid, f_grid, problem, u_node, f_node)`` for an exact Poisson pair on an n x n grid."""
    torch.manual_seed(seed)
    mesh = structured_quad_mesh(nx=n, ny=n)
    prob = PoissonProblem(mesh)
    a = torch.rand(n_samples, K, K) * 2 - 1
    eq = PoissonMultiFrequency(a=a, r=-0.5)
    f = eq.source_term(mesh.points, domain="rectangle").float()          # [B, N]
    u = eq.solution(mesh.points).float()                                 # [B, N]
    return node_to_grid(u, n, n), node_to_grid(f, n, n), prob, u, f


# ------------------------------------------------------- the load-bearing equivalence

@pytest.mark.parametrize("n", [33, 65])
def test_fd_and_fem_residuals_have_the_same_discretization_error(n):
    """``-lap_FD(u*) vs f`` and ``A u* vs M f`` must agree — same PDE, same accuracy.

    If these differed, the PINO-vs-ours comparison would be measuring the discretisation
    rather than the loss.
    """
    ug, fg, prob, u, f = _analytic_pair(n)

    lap = PINOPoissonLoss((n, n)).fd.laplacian(ug)
    lhs, rhs = -lap[..., 1:-1, 1:-1], fg[..., 1:-1, 1:-1]
    rel_fd = ((lhs - rhs).norm() / rhs.norm()).item()

    mask = prob.boundary_mask
    Au, Mf = prob._spmm(prob.A, u), prob._spmm(prob.M, f)
    rel_fem = ((Au - Mf)[:, ~mask].norm() / Mf[:, ~mask].norm()).item()

    assert rel_fd == pytest.approx(rel_fem, rel=0.15), (
        f"FD and FEM discretization errors differ ({rel_fd:.3e} vs {rel_fem:.3e}); "
        "the two residuals are not the same PDE at the same accuracy"
    )


def test_fd_laplacian_is_second_order():
    """Halving h must cut the FD residual error by ~4 — otherwise the stencil is wrong."""
    errs = []
    for n in (33, 65):
        ug, fg, *_ = _analytic_pair(n)
        lap = PINOPoissonLoss((n, n)).fd.laplacian(ug)
        lhs, rhs = -lap[..., 1:-1, 1:-1], fg[..., 1:-1, 1:-1]
        errs.append(((lhs - rhs).norm() / rhs.norm()).item())
    rate = math.log2(errs[0] / errs[1])
    assert 1.7 < rate < 2.3, f"FD Laplacian convergence rate {rate:.2f}, expected ~2"


def test_pino_loss_is_minimized_at_the_analytical_solution():
    """The PINO loss at ``u*`` is its own discretization floor, far below any other field."""
    n = 33
    ug, fg, *_ = _analytic_pair(n)
    loss = PINOPoissonLoss((n, n), reduction="rel")
    at_star = loss(ug, fg).item()
    at_zero = loss(torch.zeros_like(ug), fg).item()
    torch.manual_seed(0)
    at_noise = loss(ug + 0.1 * ug.abs().max() * torch.randn_like(ug), fg).item()
    assert at_star < 0.05
    assert at_star < 0.2 * at_zero
    assert at_star < 0.05 * at_noise


# ------------------------------------------------------------------ autodiff Laplacian

class _Analytic(torch.nn.Module):
    """``u = sin(pi x) sin(2 pi y)``; Laplacian ``-(pi^2 + 4 pi^2) u`` in closed form."""

    def forward_at(self, f_grid, coords):
        return torch.sin(math.pi * coords[..., 0]) * torch.sin(2 * math.pi * coords[..., 1])


def test_autodiff_laplacian_is_exact():
    """Unlike FD, autodiff differentiates the model exactly — machine precision, no O(h^2)."""
    coords = torch.rand(3, 64, 2, dtype=torch.float64).requires_grad_(True)
    u, lap = autodiff_laplacian(_Analytic(), None, coords)
    exact = -(math.pi ** 2 + 4 * math.pi ** 2) * u
    assert ((lap - exact).norm() / exact.norm()).item() < 1e-12


def test_autodiff_laplacian_is_per_sample_not_batch_summed():
    """The reason coords are ``[B, Q, 2]``.

    With shared ``[Q, 2]`` coordinates a single ``grad`` would return the batch *sum* of the
    per-sample gradients, silently turning the residual into a residual of the mean. Each
    row here must instead equal the batch-of-one result for the same input.
    """
    n = 17
    torch.manual_seed(0)
    model = DeepONetModel((n, n), p=16, width=32, depth=3, trunk_fourier=8).double()
    f = torch.randn(3, 1, n, n, dtype=torch.float64)
    c = model.grid_coords.double().unsqueeze(0).expand(3, -1, -1).clone().requires_grad_(True)
    _, lap_batch = autodiff_laplacian(model, f, c)

    for b in range(3):
        c1 = model.grid_coords.double().unsqueeze(0).clone().requires_grad_(True)
        _, lap_one = autodiff_laplacian(model, f[b:b + 1], c1)
        assert torch.allclose(lap_batch[b], lap_one[0], rtol=1e-8, atol=1e-8)

    # And the samples really are different, so the check above is not vacuous.
    assert not torch.allclose(lap_batch[0], lap_batch[1], rtol=1e-3, atol=1e-3)


def test_autodiff_laplacian_rejects_shared_coords():
    with pytest.raises(ValueError, match="per-sample"):
        autodiff_laplacian(_Analytic(), None, torch.rand(8, 2).requires_grad_(True))


def test_autodiff_laplacian_keeps_the_parameter_graph():
    """``create_graph`` must stay on, or the residual would have no gradient and training
    would silently do nothing."""
    n = 17
    model = DeepONetModel((n, n), p=8, width=16, depth=2, trunk_fourier=4)
    f = torch.randn(2, 1, n, n)
    c = model.grid_coords.unsqueeze(0).expand(2, -1, -1).clone().requires_grad_(True)
    _, lap = autodiff_laplacian(model, f, c)
    (lap ** 2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_autodiff_and_fd_laplacians_agree_on_a_resolved_field():
    """The two differentiations must agree where FD is accurate — a smooth, resolved field."""
    n = 65
    coords_model = _Analytic()
    xs = torch.linspace(0, 1, n)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1).unsqueeze(0).clone()
    coords.requires_grad_(True)
    u, lap_ad = autodiff_laplacian(coords_model, None, coords)
    lap_fd = PINOPoissonLoss((n, n)).fd.laplacian(node_to_grid(u.detach(), n, n))
    i = slice(2, -2)
    rel = ((lap_ad.detach().reshape(1, n, n)[:, i, i] - lap_fd[:, i, i]).norm()
           / lap_fd[:, i, i].norm()).item()
    assert rel < 0.02, f"autodiff and FD Laplacians differ by {rel:.3%} on a resolved field"


# ---------------------------------------------------------------------------- DeepONet

def test_deeponet_is_a_drop_in_for_the_fno():
    """Same call signature and output shape, so every existing loss and the unmodified
    trainer accept it — that is what makes the "our loss, other architecture" cell free."""
    n = 17
    model = DeepONetModel((n, n), p=16, width=32, depth=3)
    assert model(torch.randn(4, 1, n, n)).shape == (4, 1, n, n)
    assert model(torch.randn(1, n, n)).shape == (1, n, n)          # unbatched, like FNOModel


def test_deeponet_grid_forward_matches_a_coordinate_query():
    """``forward`` must be exactly ``forward_at`` on the grid nodes, in the repo's row-major
    node order — otherwise the residual and the reported error would see transposed fields."""
    n = 17
    model = DeepONetModel((n, n), p=16, width=32, depth=3)
    f = torch.randn(3, 1, n, n)
    grid = model(f).squeeze(1)                                     # [B, n, n]
    at = model.forward_at(f, model.grid_coords).reshape(3, n, n)
    assert torch.allclose(grid, at, rtol=1e-6, atol=1e-6)


def test_deeponet_output_scale_is_independent_of_p():
    """The 1/sqrt(p) normalisation.

    Without it the initial output grows like sqrt(p) — measured 5.5x the target magnitude at
    p=32, 12.4x at p=128 and 15.7x at p=256, a 2.85x spread — so the choice of p sets how far the
    operator starts from its target. That costs the baseline epochs for a reason that has
    nothing to do with its loss. The invariant is p-*independence*, not any absolute value.
    """
    n = 17
    rms = []
    for p in (32, 128, 256):
        torch.manual_seed(0)
        model = DeepONetModel((n, n), p=p, width=64, depth=3, trunk_fourier=16)
        rms.append(model(torch.randn(8, 1, n, n)).pow(2).mean().sqrt().item())
    spread = max(rms) / min(rms)
    assert spread < 1.6, (
        f"initial output scale varies {spread:.2f}x over p in (32, 128, 256) ({rms}); "
        "the 1/sqrt(p) normalisation is not doing its job"
    )


def test_mollifier_imposes_the_boundary_condition_exactly():
    n = 17
    m = mollifier_grid(n, n)
    assert m[0].abs().max() == 0 and m[-1].abs().max() == 0
    assert m[:, 0].abs().max() == 0 and m[:, -1].abs().max() == 0
    assert m[n // 2, n // 2].item() == pytest.approx(1.0, abs=1e-6)

    wrapped = MollifiedModel(torch.nn.Identity(), n, n)
    out = wrapped(torch.ones(2, 1, n, n))
    assert out[..., 0, :].abs().max() == 0 and out[..., :, -1].abs().max() == 0

    model = DeepONetModel((n, n), p=16, width=32, depth=3, mollify=True)
    out = model(torch.randn(2, 1, n, n))
    assert out[..., 0, :].abs().max() == 0 and out[..., :, -1].abs().max() == 0


def test_zero_boundary_matches_apply_zero_boundary_on_our_arms():
    """The PINO default BC must be the *same operation* losses.py applies to our arms.

    Not merely "also zero on the boundary": the interior has to be passed through untouched, or
    PINO would differ from the galerkin arm in the BC as well as in the residual, and the
    comparison would no longer be attributable to the residual alone.
    """
    from tensorpils.physics import apply_zero_boundary
    from tensorpils.meshing import grid_to_node, node_to_grid

    n = 17
    m = boundary_mask_grid(n, n)
    assert m[0].abs().max() == 0 and m[-1].abs().max() == 0
    assert m[:, 0].abs().max() == 0 and m[:, -1].abs().max() == 0
    assert m[1:-1, 1:-1].min().item() == 1.0                    # interior untouched

    torch.manual_seed(0)
    u = torch.randn(2, 1, n, n)
    wrapped = ZeroBoundaryModel(torch.nn.Identity(), n, n)
    got = wrapped(u)

    # The FEM path: grid -> nodes -> apply_zero_boundary -> grid.
    grid_mask = torch.zeros(n, n, dtype=torch.bool)
    grid_mask[0, :] = grid_mask[-1, :] = grid_mask[:, 0] = grid_mask[:, -1] = True
    mask = grid_mask.reshape(-1)
    want = node_to_grid(apply_zero_boundary(grid_to_node(u.squeeze(1), n, n), mask), n, n)
    assert torch.equal(got.squeeze(1), want)


def test_zero_boundary_is_exact_unlike_the_mollifier():
    """sin(pi * 1.0) is 8.7e-8 in float32; masking is bit-exact zero, with no snap needed."""
    n = 33
    out = ZeroBoundaryModel(torch.nn.Identity(), n, n)(torch.ones(1, 1, n, n))
    for sl in (out[..., 0, :], out[..., -1, :], out[..., :, 0], out[..., :, -1]):
        assert sl.abs().max().item() == 0.0


def test_deeponet_zero_boundary_masks_the_grid_but_not_the_query():
    """``--pideeponet_bc zero``: the grid output has an exactly-zero boundary ring (the same
    operation as ``ZeroBoundaryModel`` / ``apply_zero_boundary``), the interior is passed
    through untouched, and a coordinate query at a boundary point returns the *raw* network
    value -- that is what the boundary penalty acts on."""
    n = 17
    torch.manual_seed(0)
    model = DeepONetModel((n, n), p=16, width=32, depth=3, zero_boundary=True)
    f = torch.randn(2, 1, n, n)
    out = model(f)
    for sl in (out[..., 0, :], out[..., -1, :], out[..., :, 0], out[..., :, -1]):
        assert sl.abs().max().item() == 0.0
    raw = model.forward_at(f, model.grid_coords).reshape(2, 1, n, n)
    assert torch.equal(out[..., 1:-1, 1:-1], raw[..., 1:-1, 1:-1])
    bmask = torch.zeros(n, n, dtype=torch.bool)
    bmask[0, :] = bmask[-1, :] = bmask[:, 0] = bmask[:, -1] = True
    u_b = model.forward_at(f, model.grid_coords[bmask.reshape(-1)])
    assert u_b.abs().max().item() > 0.0
    with pytest.raises(ValueError):
        DeepONetModel((n, n), p=8, width=16, depth=2, mollify=True, zero_boundary=True)


def test_pideeponet_boundary_penalty_vanishes_under_the_mollifier_and_is_scale_free():
    """The ``rel`` penalty is exactly 0 for a mollified model (hard BC), positive and
    differentiable for a zero-BC model, and invariant to the output scale -- the property
    that makes ``lambda_bc = 1`` meaningful at any solution magnitude."""
    from tensorpils.baselines.pi_deeponet import PIDeepONetPoissonLoss

    n = 9
    torch.manual_seed(0)
    f = torch.randn(2, 1, n, n)
    bmask = torch.zeros(n, n, dtype=torch.bool)
    bmask[0, :] = bmask[-1, :] = bmask[:, 0] = bmask[:, -1] = True
    bmask = bmask.reshape(-1)

    def interior_coords(model):
        c = model.grid_coords[~bmask]
        return c.unsqueeze(0).expand(2, -1, -1).clone().requires_grad_(True)

    loss = PIDeepONetPoissonLoss(reduction="rel", lambda_bc=1.0)

    moll = DeepONetModel((n, n), p=8, width=16, depth=2, mollify=True)
    u_int, _ = autodiff_laplacian(moll, f, interior_coords(moll))
    # sqrt(mean + 1e-24) floors the boundary RMS at 1e-12, so 'zero' means ~1e-10 here.
    assert loss.boundary_term(moll, f, moll.grid_coords[bmask], u_int).item() < 1e-9

    zero = DeepONetModel((n, n), p=8, width=16, depth=2, zero_boundary=True)
    u_int, _ = autodiff_laplacian(zero, f, interior_coords(zero))
    term = loss.boundary_term(zero, f, zero.grid_coords[bmask], u_int)
    assert term.item() > 0.0
    term.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in zero.parameters())

    with torch.no_grad():                       # u -> 100 u
        zero.branch[-1].weight.mul_(100.0)
        zero.branch[-1].bias.mul_(100.0)
        zero.bias.mul_(100.0)
    u_int2, _ = autodiff_laplacian(zero, f, interior_coords(zero))
    term2 = loss.boundary_term(zero, f, zero.grid_coords[bmask], u_int2)
    assert term2.item() == pytest.approx(term.item(), rel=1e-4)

    # The full loss actually includes the penalty when bc_coords is given, and not otherwise.
    f_at = f.reshape(2, -1)[:, ~bmask]
    with_bc = loss(zero, f, interior_coords(zero), f_at, bc_coords=zero.grid_coords[bmask])
    without = loss(zero, f, interior_coords(zero), f_at)
    assert with_bc.item() == pytest.approx(without.item() + term2.item(), rel=1e-5)


def test_rel_lp_is_the_reference_reduction():
    """PINO's ``LpLoss.rel(size_average=True)``: per-sample relative norm, mean over batch."""
    pred = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    # sample 0: ||pred-target|| / ||target|| = 5 / 0 -> clamped; sample 1: 5/5 = 1
    assert rel_lp(pred[1:], target[1:]).item() == pytest.approx(1.0)
    assert rel_lp(target, target).item() == pytest.approx(0.0)


# -------------------------------------------------------------------------- Allen-Cahn

def test_pino_ac_residual_is_small_at_the_fem_reference():
    """The strong-form AC step residual must nearly vanish on the FEM reference trajectory.

    The reference zeros the *weak* residual exactly, so this checks that the strong form
    carries the same signs, the same integrator and the same ``A <-> -Laplacian``
    correspondence. A sign slip would show up here as an O(1) residual.
    """
    n, dt, a, eps = 33, 1e-3, 1.0, 2.0
    torch.manual_seed(0)
    mesh = structured_quad_mesh(nx=n, ny=n)
    prob = ACProblem(mesh)
    # A *smooth* initial condition, the same multi-frequency family ACDataset uses. A random
    # nodal IC would be grid-scale noise, where a finite-difference Laplacian and the FEM one
    # legitimately disagree at O(1) — that would test the resolution, not the conventions.
    from tensormesh.dataset.equation.wave import WaveMultiFrequency
    ic = WaveMultiFrequency(a=torch.rand(2, 4, 4) * 2 - 1, r=0.5)
    u0 = ic.initial_condition(mesh.points).float()                  # [B, N]
    traj = prob.fem_reference(u0, a=a, eps=eps, dt=dt, n_steps=2)   # [B, 3, N]

    seq_grid = node_to_grid(traj.float(), n, n)                     # [B, 3, H, W]
    loss = PINOACLoss((n, n), a=a, eps=eps, dt=dt)
    residual = loss(seq_grid).item()
    # Scale: the leading term is (u^{n+1}-u^n)/dt, whose mean square sets what "small" means.
    du = ((seq_grid[:, 1:] - seq_grid[:, :-1]) / dt) ** 2
    assert residual < 1e-2 * du.mean().item(), (
        f"strong AC residual {residual:.3e} is not small against the step scale "
        f"{du.mean().item():.3e} — check integrator/sign conventions"
    )


def test_pino_and_pi_deeponet_ac_losses_agree_on_the_same_field():
    """FD and autodiff versions of the *same* Allen–Cahn residual, on one smooth field.

    Ties the two baselines together: they differ in how the Laplacian is taken, not in what
    equation is being enforced.
    """
    # dt=1 deliberately: at small dt the (u^{n+1}-u^n)/dt term nearly cancels a^2*lap(u) for
    # this field, and the near-cancellation amplifies FD truncation into a spurious 6%
    # disagreement. A large dt leaves the Laplacian dominant, so the test measures what it
    # says it measures.
    n, dt, a, eps = 65, 1.0, 1.0, 2.0
    xs = torch.linspace(0, 1, n)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1).unsqueeze(0).clone()
    coords.requires_grad_(True)
    u, lap = autodiff_laplacian(_Analytic(), None, coords)          # [1, N]

    seq_node = torch.stack([0.5 * u, u], dim=1)                     # [1, 2, N]
    # Same node set on both sides: PINOACLoss averages over the grid interior, so the autodiff
    # loss gets the interior nodes, as the trainer's collocation points are. Comparing a
    # full-grid mean against an interior mean differs by 63^2/65^2 = 6% at this resolution for
    # scale reasons alone.
    interior = ~structured_quad_mesh(nx=n, ny=n).boundary_mask.bool()
    ad = PIDeepONetACLoss(a=a, eps=eps, dt=dt)(
        seq_node[..., interior], lap[..., interior].unsqueeze(1)).item()
    fd = PINOACLoss((n, n), a=a, eps=eps, dt=dt)(
        node_to_grid(seq_node.detach(), n, n)).item()
    assert ad == pytest.approx(fd, rel=0.02), (
        f"autodiff ({ad:.4e}) and FD ({fd:.4e}) Allen–Cahn residuals disagree by more than "
        "FD truncation explains"
    )


# ------------------------------------------------------------------------- run plumbing

def test_arch_tag_sees_through_the_mollifier_wrapper():
    n = 17
    fno_like = torch.nn.Identity()
    assert _arch_tag(MollifiedModel(fno_like, n, n)) == "fno"
    assert _arch_tag(ZeroBoundaryModel(fno_like, n, n)) == "fno"
    assert _arch_tag(ZeroBoundaryModel(DeepONetModel((n, n), p=8, width=16, depth=2),
                                       n, n)) == "deeponet"
    assert _arch_tag(DeepONetModel((n, n), p=8, width=16, depth=2)) == "deeponet"
    assert _arch_tag(MollifiedModel(DeepONetModel((n, n), p=8, width=16, depth=2),
                                    n, n)) == "deeponet"


def test_deeponet_run_prefix_does_not_collide_with_the_fno_one():
    """A run of another architecture must not overwrite the identically-configured FNO run's
    checkpoint and results JSON."""
    from tensorpils.trainer import ArchPrefixMixin as _ArchPrefixMixin

    class _Fake:
        def _file_prefix(self):
            return "fno_data_K4_samples-1024-128-256"

    class _Tagged(_ArchPrefixMixin, _Fake):
        def __init__(self, model):
            self.model = model

    n = 17
    deeponet = _Tagged(DeepONetModel((n, n), p=8, width=16, depth=2))._file_prefix()
    fno = _Tagged(torch.nn.Identity())._file_prefix()
    assert fno == "fno_data_K4_samples-1024-128-256"        # existing names unchanged
    assert deeponet == "deeponet_data_K4_samples-1024-128-256"
    assert deeponet != fno


# ------------------------------------------------------------------------- Stokes

_PI = math.pi


def _stokes_manufactured(pts):
    """Divergence-free, zero-boundary velocity (curl of sin^2 sin^2), a smooth pressure, and
    the body force f = -lap u + grad p that they solve (mu = 1). ``pts`` ``[..., 2]``."""
    x, y = pts[..., 0], pts[..., 1]
    u = torch.stack([_PI * torch.sin(_PI * x) ** 2 * torch.sin(2 * _PI * y),
                     -_PI * torch.sin(2 * _PI * x) * torch.sin(_PI * y) ** 2], dim=-1)
    lap = torch.stack([2 * _PI ** 3 * torch.cos(2 * _PI * x) * torch.sin(2 * _PI * y)
                       - 4 * _PI ** 3 * torch.sin(_PI * x) ** 2 * torch.sin(2 * _PI * y),
                       4 * _PI ** 3 * torch.sin(2 * _PI * x) * torch.sin(_PI * y) ** 2
                       - 2 * _PI ** 3 * torch.sin(2 * _PI * x) * torch.cos(2 * _PI * y)], dim=-1)
    p = torch.sin(_PI * x) * torch.cos(_PI * y)
    gp = torch.stack([_PI * torch.cos(_PI * x) * torch.cos(_PI * y),
                      -_PI * torch.sin(_PI * x) * torch.sin(_PI * y)], dim=-1)
    return u, p, lap, gp, -lap + gp


def test_stokes_rel_reduction_is_the_ratio_of_residual_to_force():
    """The 'rel' Stokes loss is ||(r_mom, w r_div)|| / ||f|| with r_mom the residual: it must
    vanish at the exact solution and be exactly 1 for the all-zero prediction (r_mom = -f). A
    version that fed the residual through rel_lp(residual, f) subtracted f twice, giving 1.0 at
    the solution and 2.0 at zero, and trained the Stokes baseline toward the solution for 2f."""
    torch.manual_seed(0)
    coords = torch.rand(2, 64, 2, dtype=torch.float64)
    _, _, _, _, f = _stokes_manufactured(coords)
    loss = PIDeepONetStokesLoss(mu=1.0, reduction="rel", div_weight=1.0)
    # exact solution -> 0; zero prediction -> 1; doubled solution (r_mom = f) -> 1
    for scale, expected in ((1.0, 0.0), (0.0, 1.0), (2.0, 1.0)):
        c = coords.clone().requires_grad_(True)
        assert loss(_AnalyticStokesModel(scale), None, c, f).item() == pytest.approx(
            expected, abs=1e-8)


class _AnalyticStokesModel(torch.nn.Module):
    """A 'model' whose forward_at returns ``scale`` times the manufactured (u_x, u_y, p) -- to
    test autodiff."""
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward_at(self, f_grid, coords):
        u, p, _, _, _ = _stokes_manufactured(coords)
        return self.scale * torch.cat([u, p.unsqueeze(-1)], dim=-1)          # [B, Q, 3]


def test_autodiff_stokes_matches_the_analytic_derivatives():
    torch.manual_seed(0)
    coords = torch.rand(2, 50, 2, dtype=torch.float64).requires_grad_(True)
    d = autodiff_stokes(_AnalyticStokesModel(), None, coords, p_scale=1.0)
    u, p, lap, gp, _ = _stokes_manufactured(coords.detach())
    assert torch.allclose(d["u"], u) and torch.allclose(d["p"], p)
    assert torch.allclose(d["lap_u"], lap, atol=1e-8, rtol=1e-8)
    assert torch.allclose(d["grad_p"], gp, atol=1e-8, rtol=1e-8)
    assert d["div_u"].abs().max() < 1e-8                         # curl field: exactly solenoidal
    # p_scale is applied to the pressure and hence to its gradient
    d2 = autodiff_stokes(_AnalyticStokesModel(), None, coords, p_scale=50.0)
    assert torch.allclose(d2["grad_p"], 50.0 * gp, atol=1e-6, rtol=1e-8)


def test_multi_output_deeponet_bc_acts_on_the_velocity_channels_only():
    n = 9
    torch.manual_seed(0)
    f = torch.randn(2, 2, n, n)
    for kw in (dict(zero_boundary=True), dict(mollify=True)):
        model = DeepONetModel((n, n), in_channels=2, out_channels=3, bc_channels=(0, 1),
                              p=8, width=16, depth=2, **kw)
        out = model(f)
        assert out.shape == (2, 3, n, n)
        for c in (0, 1):
            for sl in (out[:, c, 0, :], out[:, c, -1, :], out[:, c, :, 0], out[:, c, :, -1]):
                assert sl.abs().max().item() == 0.0
        assert out[:, 2, 0, :].abs().max().item() > 0.0        # pressure boundary untouched
        q = model.forward_at(f, torch.rand(5, 2))
        assert q.shape == (2, 5, 3)
    # single-output contract unchanged
    m1 = DeepONetModel((n, n), p=8, width=16, depth=2)
    assert m1(torch.randn(2, 1, n, n)).shape == (2, 1, n, n)
    assert m1.forward_at(torch.randn(2, 1, n, n), torch.rand(5, 2)).shape == (2, 5)
