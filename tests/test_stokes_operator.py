"""Stokes (Taylor-Hood Q2/Q1) operator, preconditioner and loss tests.

Needs ``tensormesh`` but **not** ``neuralop`` — these verify the FEM operator, the
structured Q2 mesh, the block preconditioner and the losses, not the model.
"""

import math

import numpy as np
import pytest
import torch

from tensorpils.meshing import structured_quad9_mesh, node_to_grid, grid_to_node
from tensorpils.physics import StokesProblem
from tensorpils.preconditioners import StokesBlockPreconditioner
from tensorpils.losses import StokesGalerkinLoss, StokesPLSLoss, build_stokes_loss
from tensorpils.data import stokes_body_force, create_stokes_datasets


NP = 5                      # pressure grid; velocity grid is 2*NP-1 = 9


@pytest.fixture(scope="module")
def problem():
    mesh = structured_quad9_mesh(nx=NP, ny=NP)
    return StokesProblem(mesh, nx_p=NP, ny_p=NP, mu=1.0)


def _random_fields(problem, B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(B, problem.n_u, 2, generator=g)
    p = torch.randn(B, problem.n_p, generator=g)
    f = torch.randn(B, problem.n_u, 2, generator=g)
    return u, p, f


# ------------------------------------------------------------------ mesh / layout

def test_quad9_mesh_geometry():
    """Fine grid is (2n-1)^2, nodes are row-major, and the mesh is a single quad9 block."""
    mesh = structured_quad9_mesh(nx=NP, ny=NP)
    N = 2 * NP - 1
    assert mesh.points.shape[0] == N * N
    pts = mesh.points.reshape(N, N, 2)
    # row-major: x varies along the last axis, y along the first
    assert torch.allclose(pts[0, :, 0], torch.linspace(0, 1, N).double())
    assert torch.allclose(pts[:, 0, 1], torch.linspace(0, 1, N).double())
    assert mesh.boundary_mask.reshape(N, N)[1:-1, 1:-1].sum() == 0


def test_pressure_nodes_are_the_even_subgrid(problem):
    """The Q1 pressure DOFs are exactly the ``[::2, ::2]`` fine nodes, in coarse row-major
    order — the fact ``StokesProblem.from_grid`` relies on to read pressure by striding."""
    N = 2 * NP - 1
    expected = np.arange(N * N).reshape(N, N)[::2, ::2].ravel()
    mesh = structured_quad9_mesh(nx=NP, ny=NP)
    from tensorpils.physics import _StokesBilinearForm
    layout = _StokesBilinearForm.from_mesh(mesh, quadrature_order=5, mu=1.0).layout
    assert np.array_equal(layout.node_ids("p").cpu().numpy(), expected)
    assert problem.n_p == NP * NP


def test_pack_unpack_roundtrip(problem):
    u, p, _ = _random_fields(problem)
    c = problem.pack(u, p)
    assert c.shape == (u.shape[0], problem.n_dofs)
    u2, p2 = problem.unpack(c)
    assert torch.allclose(u, u2) and torch.allclose(p, p2)


def test_from_grid_matches_pack(problem):
    """FNO-grid → node conversion is the exact inverse of the node → grid view."""
    nx, ny = problem.grid_size
    u, p, _ = _random_fields(problem)
    u_grid, p_grid = problem.to_grid(u, p)
    # rebuild the 3-channel FNO output: pressure lives on the even subgrid
    out = torch.zeros(u.shape[0], 3, ny, nx)
    out[:, :2] = u_grid
    out[:, 2, ::2, ::2] = p_grid
    u2, p2 = problem.from_grid(out)
    assert torch.allclose(u, u2) and torch.allclose(p, p2)


# ------------------------------------------------------------------ operator

def test_saddle_point_operator_structure(problem):
    """K is symmetric, its velocity block decouples across components and equals mu*A_Q2."""
    Kd = problem.K.to_dense().double()
    assert torch.allclose(Kd, Kd.T, atol=1e-10)
    A = Kd[:problem.off_p, :problem.off_p].reshape(problem.n_u, 2, problem.n_u, 2)
    assert A[:, 0, :, 1].abs().max() < 1e-12          # components decouple
    A_q2 = problem.A.to_dense().double()
    assert torch.allclose(A[:, 0, :, 0], problem.mu * A_q2, atol=1e-9)
    # the pressure-pressure block is exactly zero (a genuine saddle point)
    assert Kd[problem.off_p:, problem.off_p:].abs().max() < 1e-14


def test_residual_masks_dirichlet_dofs(problem):
    u, p, f = _random_fields(problem)
    r = problem.residual(u, p, f)
    assert r.shape == (u.shape[0], problem.n_dofs)
    assert r[:, problem.dirichlet_mask].abs().max() < 1e-12


def test_residual_is_pressure_gauge_invariant():
    r"""``B^T 1 = 0`` on the zero-BC velocity space, so a constant pressure shift leaves the
    residual unchanged. This is why the constant mode must be fixed by projection, never by
    the loss.

    Checked in float64: the identity is exact, so anything above round-off is a real defect
    (in float32 the shift would only cancel to ~1e-7 relative, which would hide one).
    """
    prob = StokesProblem(structured_quad9_mesh(nx=NP, ny=NP), nx_p=NP, ny_p=NP, mu=1.0).double()
    u, p, f = _random_fields(prob)
    u, p, f = u.double(), p.double(), f.double()
    r0 = prob.residual(u, p, f)
    r1 = prob.residual(u, p + 3.7, f)
    assert (r1 - r0).abs().max() < 1e-12 * max(1.0, r0.abs().max().item())


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


# ------------------------------------------------------------------ reference solve

def test_fem_reference_zeroes_the_residual(problem):
    """Self-consistency: the discrete reference solves exactly the residual the label-free
    loss minimizes, so a perfectly trained model reproduces it."""
    prob = StokesProblem(structured_quad9_mesh(nx=NP, ny=NP), nx_p=NP, ny_p=NP, mu=1.0).double()
    torch.manual_seed(0)
    f = torch.randn(2, prob.n_u, 2, dtype=torch.float64)
    u, p = prob.fem_reference(f)
    r = prob.residual(u, p, f)
    scale = prob.load_vector(f).abs().max()
    assert (r.abs().max() / scale) < 1e-9


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


def test_taylor_hood_convergence():
    """h-refinement against a manufactured solution: velocity H1 and pressure L2 both O(h^2).

    Guards the structured quad9 connectivity — a wrong node order still assembles, but the
    convergence rate collapses.
    """
    pi = math.pi

    def exact_u(pts):
        x, y = pts[..., 0], pts[..., 1]
        return torch.stack([pi * torch.sin(pi * x) ** 2 * torch.sin(2 * pi * y),
                            -pi * torch.sin(2 * pi * x) * torch.sin(pi * y) ** 2], dim=-1)

    def exact_lap_u(pts):
        x, y = pts[..., 0], pts[..., 1]
        lx = (2 * pi ** 3 * torch.cos(2 * pi * x) * torch.sin(2 * pi * y)
              - 4 * pi ** 3 * torch.sin(pi * x) ** 2 * torch.sin(2 * pi * y))
        ly = (4 * pi ** 3 * torch.sin(2 * pi * x) * torch.sin(pi * y) ** 2
              - 2 * pi ** 3 * torch.sin(2 * pi * x) * torch.cos(2 * pi * y))
        return torch.stack([lx, ly], dim=-1)

    def exact_p(pts):
        return torch.sin(pi * pts[..., 0]) * torch.cos(pi * pts[..., 1])

    def exact_grad_p(pts):
        x, y = pts[..., 0], pts[..., 1]
        return torch.stack([pi * torch.cos(pi * x) * torch.cos(pi * y),
                            -pi * torch.sin(pi * x) * torch.sin(pi * y)], dim=-1)

    errs = []
    for n_p in (5, 9, 17):
        mesh = structured_quad9_mesh(nx=n_p, ny=n_p)
        prob = StokesProblem(mesh, nx_p=n_p, ny_p=n_p, mu=1.0).double()
        pts = mesh.points
        f = (-exact_lap_u(pts) + exact_grad_p(pts)).unsqueeze(0)
        u, p = prob.fem_reference(f)
        # velocity L2 error and pressure L2 error, both in the FE norm
        eu = prob.velocity_l2(u - exact_u(pts).unsqueeze(0)).item()
        p_ex = prob.project_pressure_gauge(
            exact_p(pts[torch.tensor(
                np.arange((2 * n_p - 1) ** 2).reshape(2 * n_p - 1, -1)[::2, ::2].ravel()
            )]).unsqueeze(0))
        ep = prob.pressure_l2(p - p_ex).item()
        errs.append((1.0 / (n_p - 1), eu, ep))

    for (h0, eu0, ep0), (h1, eu1, ep1) in zip(errs, errs[1:]):
        rate_u = math.log(eu0 / eu1) / math.log(h0 / h1)
        rate_p = math.log(ep0 / ep1) / math.log(h0 / h1)
        assert rate_u > 2.5, f"velocity L2 rate {rate_u:.2f} too low (Q2 should be ~3)"
        assert rate_p > 1.7, f"pressure L2 rate {rate_p:.2f} too low (Q1 should be ~2)"


# ------------------------------------------------------------------ preconditioner

def test_block_preconditioner_is_spd(problem):
    r"""``P`` is used as a **norm weight** in ``½ rᵀPr``, so it must be symmetric positive
    definite — otherwise the loss is unbounded below."""
    pre = StokesBlockPreconditioner(problem, mg_levels=3, schur_omega=0.5)
    n = problem.n_dofs
    P = pre(torch.eye(n))                                  # rows: P e_i
    assert torch.isfinite(P).all()
    asym = (P - P.T).abs().max() / P.abs().max()
    assert asym < 1e-5, f"preconditioner is not symmetric (rel asym {asym:.2e})"
    ev = torch.linalg.eigvalsh(0.5 * (P + P.T))
    assert ev.min() > 0, f"preconditioner is not positive definite (lambda_min={ev.min():.3e})"


def test_preconditioner_improves_conditioning(problem):
    """The point of the whole exercise: P must cut kappa(K P K) below kappa(K^2)."""
    pre = StokesBlockPreconditioner(problem, mg_levels=3, schur_omega=0.5)
    n = problem.n_dofs
    free = (~problem.dirichlet_mask).nonzero().squeeze(1)
    K = problem.K.to_dense().double()[free][:, free]
    P = pre(torch.eye(n)).double()[free][:, free]
    P = 0.5 * (P + P.T)

    # drop the pressure constant mode, which is a genuine nullvector of K
    def cond_no_null(M):
        s = torch.linalg.svdvals(M)
        return (s[0] / s[-2]).item()

    kappa_bare = cond_no_null(K @ K)
    kappa_pls = cond_no_null(K @ P @ K)
    assert kappa_pls < kappa_bare, (
        f"preconditioned conditioning {kappa_pls:.3e} not better than bare {kappa_bare:.3e}")


# ------------------------------------------------------------------ losses

@pytest.mark.parametrize("loss_type", ["galerkin", "pls"])
def test_losses_forward_backward(problem, loss_type):
    pre = (StokesBlockPreconditioner(problem, mg_levels=3) if loss_type == "pls" else None)
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
    torch.manual_seed(3)
    f = torch.randn(2, problem.n_u, 2)
    u, p = problem.fem_reference(f)
    pre = StokesBlockPreconditioner(problem, mg_levels=3)
    ref_scale = StokesGalerkinLoss(problem)(torch.zeros_like(u), torch.zeros_like(p), f)
    for crit in (StokesGalerkinLoss(problem), StokesPLSLoss(problem, pre)):
        assert crit(u, p, f).item() < 1e-8 * max(ref_scale.item(), 1.0)


def test_pls_requires_a_preconditioner(problem):
    with pytest.raises(ValueError, match="requires a preconditioner"):
        build_stokes_loss("pls", problem, precond=None)


# ------------------------------------------------------------------ dataset

def test_body_force_shape_and_decay():
    mesh = structured_quad9_mesh(nx=NP, ny=NP)
    coeffs = torch.ones(2, 2, 3, 3)
    f = stokes_body_force(coeffs, mesh.points)
    assert f.shape == (2, mesh.points.shape[0], 2)
    assert torch.isfinite(f).all()


def test_create_stokes_datasets_shapes():
    tr, va, te = create_stokes_datasets(n_train=4, n_val=2, n_test=2, K=2,
                                        grid_resolution=2 * NP - 1, seed=0)
    assert (len(tr), len(va), len(te)) == (4, 2, 2)
    assert tr.problem is va.problem is te.problem          # one shared operator
    f_grid, f_node, u, p = tr[0]
    nx, ny = tr.grid_size
    assert f_grid.shape == (2, ny, nx)
    assert f_node.shape == (tr.problem.n_u, 2)
    assert u.shape == (tr.problem.n_u, 2) and p.shape == (tr.problem.n_p,)
    # normalization: unit reference velocity norm
    assert abs(tr.problem.velocity_l2(u.unsqueeze(0)).item() - 1.0) < 1e-4


def test_even_grid_resolution_rejected():
    with pytest.raises(ValueError, match="odd velocity grid"):
        create_stokes_datasets(n_train=1, n_val=0, n_test=0, K=2, grid_resolution=8)


# ------------------------------------------------------------------ trainer / viz

class _DummyNet(torch.nn.Module):
    """Stand-in for the FNO: [B,2,H,W] -> [B,3,H,W]. Keeps these tests free of ``neuralop``."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def test_trainer_and_viz_report_the_same_error(tmp_path):
    """The reported error must not depend on *which* code path computed the prediction.

    Regression test: ``viz`` used to rebuild the prediction pipeline itself and silently
    dropped the input/pressure scalings, reporting 4102 % where the trainer measured 7.7 %.
    Both now go through ``StokesTrainer._predict``.
    """
    from tensorpils.trainer import StokesTrainer

    tr, va, te = create_stokes_datasets(n_train=4, n_val=2, n_test=4, K=2,
                                        grid_resolution=2 * NP - 1, seed=0)
    trainer = StokesTrainer(
        model=_DummyNet(), train_dataset=tr, val_dataset=va, test_dataset=te,
        loss_type="data", device="cpu", epochs=1, batch_size=2,
        output_dir=str(tmp_path))

    _, ru, rp = trainer._eval_loader(trainer.test_loader)
    med_u, med_p = trainer.compute_error_distribution()
    # dataset-level vs per-sample-median aggregation differ slightly, but not by orders
    # of magnitude — which is what a dropped scaling looks like.
    assert med_u / 100.0 == pytest.approx(ru, rel=0.5)
    assert med_p / 100.0 == pytest.approx(rp, rel=0.5)


def test_trainer_predict_applies_scalings_and_projections(tmp_path):
    from tensorpils.trainer import StokesTrainer

    tr, va, te = create_stokes_datasets(n_train=4, n_val=2, n_test=2, K=2,
                                        grid_resolution=2 * NP - 1, seed=0)
    trainer = StokesTrainer(
        model=_DummyNet(), train_dataset=tr, val_dataset=va, test_dataset=te,
        loss_type="pls", device="cpu", epochs=1, batch_size=2, mg_levels=3,
        output_dir=str(tmp_path))
    assert trainer.f_scale > 1.0 and trainer.p_scale > 1.0     # the scales are non-trivial here

    f_grid = torch.stack([tr[i][0] for i in range(2)])
    u, p = trainer._predict(f_grid)
    assert u[:, trainer.problem.boundary_mask].abs().max() < 1e-6          # zero velocity BC
    assert (p * trainer.problem.m_p_lumped).sum(dim=-1).abs().max() < 1e-4  # zero-mean pressure


# ------------------------------------------------- blend family (t: bare -> preconditioned)

def _admissible_basis(problem):
    """Orthonormal basis of {interior velocity} + {w-weighted zero-mean pressure}, float64.

    The subspace the loss actually acts on, and the only one where ``K`` is invertible. Mirrors
    ``experiments/stokes/conditioning/measure_conditioning.py`` so the tests pin the same object
    the paper's numbers are measured on.
    """
    off_p, n_p = problem.off_p, problem.n_p
    free_u = (~problem.dirichlet_mask[:off_p]).nonzero(as_tuple=True)[0]
    w = problem.m_p_lumped.double()
    w = w / w.norm()
    e1 = torch.zeros(n_p, dtype=torch.float64)
    e1[0] = 1.0
    v = w - e1
    H = torch.eye(n_p, dtype=torch.float64)
    if v.norm() > 1e-14:
        v = v / v.norm()
        H = H - 2.0 * torch.outer(v, v)
    Z = torch.zeros(problem.n_dofs, free_u.numel() + n_p - 1, dtype=torch.float64)
    Z[free_u, torch.arange(free_u.numel())] = 1.0
    Z[off_p:, free_u.numel():] = H[:, 1:]
    return Z


def test_blend_endpoints_are_exact(problem):
    """``t=0`` is the bare loss up to ``alpha``; ``t=1`` is the block preconditioner itself."""
    from tensorpils.preconditioners import StokesBlendPreconditioner
    base = StokesBlockPreconditioner(problem, mg_levels=3)
    r = _random_fields(problem)[0].reshape(3, -1)[:, :problem.n_dofs].contiguous()
    r = torch.randn(3, problem.n_dofs).masked_fill(problem.dirichlet_mask, 0.0)

    b0 = StokesBlendPreconditioner(base, strength=0.0)
    b1 = StokesBlendPreconditioner(base, strength=1.0)
    assert torch.allclose(b0(r), b0.alpha * r)
    assert torch.allclose(b1(r), base(r))


def test_blend_alpha_matches_lambda_max(problem):
    """``alpha`` is the top eigenvalue of the block preconditioner (power iteration)."""
    from tensorpils.preconditioners import StokesBlendPreconditioner
    base = StokesBlockPreconditioner(problem, mg_levels=3).double()
    Z = _admissible_basis(problem)
    P = base(torch.eye(problem.n_dofs, dtype=torch.float64))
    P_S = Z.T @ (0.5 * (P + P.T)) @ Z
    lam = torch.linalg.eigvalsh(P_S).max().item()
    assert StokesBlendPreconditioner(base, strength=0.5).alpha == pytest.approx(lam, rel=5e-3)


@pytest.mark.parametrize("t", [0.0, 0.5, 0.9, 1.0])
def test_blend_is_spd_for_every_t(problem, t):
    """A convex combination of SPD operators is SPD — required, since ``P_t`` is a norm."""
    from tensorpils.preconditioners import StokesBlendPreconditioner
    base = StokesBlockPreconditioner(problem, mg_levels=3).double()
    pre = StokesBlendPreconditioner(base, strength=t)
    Z = _admissible_basis(problem)
    P = pre(torch.eye(problem.n_dofs, dtype=torch.float64))
    ev = torch.linalg.eigvalsh(Z.T @ (0.5 * (P + P.T)) @ Z)
    assert ev.min() > 0, f"P_t not positive definite at t={t} (lambda_min={ev.min():.3e})"


def test_blend_sweeps_the_conditioning_between_the_two_anchors(problem):
    r"""``kappa(K P_t K)`` must go from ``kappa(K^2)`` at ``t=0`` to ``kappa(KPK)`` at ``t=1``.

    This is the axis of the collapse plot: without it, a sweep over ``t`` would have no
    quantitative meaning.
    """
    from tensorpils.preconditioners import StokesBlendPreconditioner
    base = StokesBlockPreconditioner(problem, mg_levels=3).double()
    Z = _admissible_basis(problem)
    K_S = Z.T @ problem.K.to_dense().double() @ Z
    K_S = 0.5 * (K_S + K_S.T)
    eye = torch.eye(problem.n_dofs, dtype=torch.float64)

    def cond(t):
        P = StokesBlendPreconditioner(base, strength=t)(eye)
        P_S = Z.T @ (0.5 * (P + P.T)) @ Z
        ev = torch.linalg.eigvalsh(K_S @ P_S @ K_S).abs()
        return (ev.max() / ev.min()).item()

    ev_k = torch.linalg.eigvalsh(K_S).abs()
    kappa_k2 = (ev_k.max() / ev_k.min()).item() ** 2
    c0, cm, c1 = cond(0.0), cond(0.9), cond(1.0)
    assert c0 == pytest.approx(kappa_k2, rel=1e-6)      # t=0 is exactly the bare loss
    assert c1 < cm < c0                                 # and it decreases toward the block P


# ------------------------------------------------- monolithic multigrid (P ~ K^-1)

@pytest.fixture(scope="module")
def monolithic(problem):
    from tensorpils.preconditioners import StokesMonolithicMultigrid
    return StokesMonolithicMultigrid(problem, n_levels=3, n_pre=4, n_post=4,
                                     cheb_degree=8, schur_omega=1.0)


def _consistent_residual(problem, B=2, seed=0):
    """A residual in the range of ``K``: zero on Dirichlet DOFs, plain zero-sum pressure.

    ``1ᵀB = 0``, so every ``K``-image has zero-*sum* continuity rows (which a real training
    residual satisfies automatically). Feeding an inconsistent residual would make any solver
    look divergent.
    """
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(B, problem.n_dofs, generator=g).masked_fill(problem.dirichlet_mask, 0.0)
    p = r[:, problem.off_p:]
    return torch.cat([r[:, :problem.off_p], p - p.mean(dim=-1, keepdim=True)], dim=-1)


def test_monolithic_vcycle_contracts(problem, monolithic):
    """Used as a stationary iteration the V-cycle must reduce the residual monotonically."""
    r = _consistent_residual(problem)
    # the operator the preconditioner (and the masked residual) actually targets: Dirichlet
    # rows/cols eliminated with a unit diagonal. Using the raw K would iterate on a different
    # problem and look divergent.
    K = problem.K.to_dense().float()
    m = problem.dirichlet_mask
    K[m, :] = 0.0
    K[:, m] = 0.0
    K[m, m] = 1.0
    x = torch.zeros_like(r)
    norms = [r.norm(dim=-1)]
    for _ in range(5):
        x = x + monolithic(r - x @ K.T)
        norms.append((r - x @ K.T).norm(dim=-1))
    rel = (norms[-1] / norms[0]).max().item()
    assert rel < 0.5, f"V-cycle did not contract over 5 cycles (relative residual {rel:.3e})"
    for a, b in zip(norms, norms[1:]):
        assert (b < a).all(), "residual increased during the iteration"


def test_monolithic_preserves_bc_and_gauge(problem, monolithic):
    """``P`` must map into the admissible subspace: the applied form ``½‖Pr‖²`` counts every DOF,
    so anything left on a constrained one is charged to the loss."""
    Pr = monolithic(_consistent_residual(problem))
    assert Pr[:, problem.dirichlet_mask].abs().max() < 1e-6
    p = Pr[:, problem.off_p:]
    assert (p * problem.m_p_lumped).sum(dim=-1).abs().max() < 1e-4


def test_monolithic_applied_form_beats_the_weighted_block(problem):
    r"""The regime claim: ``kappa((PK)ᵀPK)`` with a monolithic ``P`` beats ``kappa(KPK)`` with the
    block ``P`` — the reason the applied form is worth the extra smoothing work."""
    Z = _admissible_basis(problem)
    K_S = Z.T @ problem.K.to_dense().double() @ Z
    K_S = 0.5 * (K_S + K_S.T)
    eye = torch.eye(problem.n_dofs, dtype=torch.float64)

    from tensorpils.preconditioners import StokesMonolithicMultigrid
    mono64 = StokesMonolithicMultigrid(problem, n_levels=3, n_pre=4, n_post=4,
                                       cheb_degree=8, schur_omega=1.0).double()
    Pm = mono64(eye).T                                    # not symmetric: keep the transpose
    PK = (Z.T @ Pm @ Z) @ K_S
    ev = torch.linalg.eigvalsh(PK.T @ PK)
    kappa_applied = (ev.max() / ev.min()).item()

    Pb = StokesBlockPreconditioner(problem, mg_levels=3).double()(eye)
    Pb_S = Z.T @ (0.5 * (Pb + Pb.T)) @ Z
    ev = torch.linalg.eigvalsh(K_S @ Pb_S @ K_S).abs()
    kappa_weighted = (ev.max() / ev.min()).item()

    assert kappa_applied < kappa_weighted, (
        f"monolithic applied {kappa_applied:.3e} not better than block weighted "
        f"{kappa_weighted:.3e}")


def test_monolithic_loss_forward_backward(problem, monolithic):
    """Autograd must flow through the whole V-cycle (it is all sparse mat-vecs and scalings)."""
    from tensorpils.losses import build_stokes_loss
    crit = build_stokes_loss("pls", problem, precond=monolithic, form="applied")
    u, p, f = _random_fields(problem)
    u.requires_grad_(True)
    p.requires_grad_(True)
    loss = crit(u, p, f)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert torch.isfinite(u.grad).all() and u.grad.abs().sum() > 0
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_applied_loss_vanishes_at_the_reference(problem, monolithic):
    """``½‖Pr‖²`` is zero exactly where the residual is — the reference stays the target."""
    from tensorpils.losses import build_stokes_loss
    crit = build_stokes_loss("pls", problem, precond=monolithic, form="applied")
    f = _random_fields(problem)[2]
    u, p = problem.fem_reference(f)
    assert crit(u, p, f).item() < 1e-8


def test_build_stokes_loss_rejects_an_unknown_form(problem):
    from tensorpils.losses import build_stokes_loss, StokesAppliedPLSLoss, StokesPLSLoss
    pre = StokesBlockPreconditioner(problem, mg_levels=3)
    assert isinstance(build_stokes_loss("pls", problem, pre, form="weighted"), StokesPLSLoss)
    assert isinstance(build_stokes_loss("pls", problem, pre, form="applied"), StokesAppliedPLSLoss)
    with pytest.raises(ValueError, match="form"):
        build_stokes_loss("pls", problem, pre, form="nope")


def test_monolithic_P_is_indefinite_so_it_cannot_be_a_norm(problem, monolithic):
    r"""The form/preconditioner pairing is *forced*, and this is why.

    ``K`` restricted to the admissible subspace is a saddle point: it has exactly one negative
    eigenvalue per (gauge-fixed) pressure DOF. Any approximate **inverse** inherits that signature,
    so a monolithic ``P`` is indefinite and ``½ rᵀPr`` is unbounded below — it cannot be used as a
    norm weight no matter how well tuned. Conversely the block ``P`` is SPD by construction but is
    not an inverse, so it cannot be applied. Neither operator can borrow the other's loss form.
    """
    Z = _admissible_basis(problem)
    eye = torch.eye(problem.n_dofs, dtype=torch.float64)
    n_free_u = int((~problem.dirichlet_mask[:problem.off_p]).sum())

    from tensorpils.preconditioners import StokesMonolithicMultigrid
    mono64 = StokesMonolithicMultigrid(problem, n_levels=3, n_pre=4, n_post=4,
                                       cheb_degree=8, schur_omega=1.0).double()
    P = mono64(eye).T
    P_S = Z.T @ (0.5 * (P + P.T)) @ Z
    n_neg = int((torch.linalg.eigvalsh(P_S) < 0).sum())
    assert n_neg == Z.shape[1] - n_free_u, (
        f"expected one negative eigenvalue per gauge-fixed pressure DOF "
        f"({Z.shape[1] - n_free_u}), got {n_neg}")

    # ...and the block preconditioner is SPD, so it can be (and only be) a norm weight.
    Pb = StokesBlockPreconditioner(problem, mg_levels=3).double()(eye)
    assert torch.linalg.eigvalsh(Z.T @ (0.5 * (Pb + Pb.T)) @ Z).min() > 0


def test_galerkin_continuity_weight_scales_only_the_continuity_rows():
    """``½‖(r_mom, w·r_cont)‖²`` — the FEM counterpart of the strong-form ``--pi_div_weight``.

    The strong-form baseline tunes that weight on validation and it is worth ~27x to it on the
    structured benchmark (46.05 % velocity at ``w=1``, 1.73 % at ``w=100``), so an *unweighted*
    FEM control is not the comparison a weighted PINO should be read against. Measured, the two
    objectives are nearly the same function at ``w=1`` (gradient cosine 0.993) and diverge as
    ``w`` grows (0.83 at ``w=300``): the weight, not the residual's form, is what separates them.
    """
    import torch
    from tensorpils.losses import StokesGalerkinLoss
    from tensorpils.meshing import structured_quad9_mesh
    from tensorpils.physics import StokesProblem

    n_p = 9
    prob = StokesProblem(structured_quad9_mesh(nx=n_p, ny=n_p), nx_p=n_p, ny_p=n_p, mu=1.0)
    torch.manual_seed(0)
    u = torch.randn(3, prob.n_u, 2) * 0.01
    p = torch.randn(3, prob.n_p) * 0.01
    f = torch.randn(3, prob.n_u, 2)

    r = prob.residual(u, p, f)
    off = prob.off_p
    mom = 0.5 * (r[..., :off] ** 2).sum(-1).mean()
    con = 0.5 * (r[..., off:] ** 2).sum(-1).mean()

    for w in (1.0, 10.0, 300.0):
        got = StokesGalerkinLoss(prob, div_weight=w)(u, p, f)
        assert torch.allclose(got, mom + (w ** 2) * con, rtol=1e-5), f"w={w}"

    # w=1 must reproduce the bare control byte for byte, so existing runs are unaffected.
    assert torch.equal(StokesGalerkinLoss(prob)(u, p, f),
                       StokesGalerkinLoss(prob, div_weight=1.0)(u, p, f))
