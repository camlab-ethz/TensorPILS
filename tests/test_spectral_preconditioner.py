"""Endpoint and property tests for the spectral preconditioner (blend / power).

These verify the core mathematical claims in isolation, before any wiring:
    strength=0  ->  P = I           (residual / no-preconditioner anchor)
    strength=1  ->  P = A^{-1}      (supervised anchor)
plus the convex-combination identity, the closed-form condition number, and
differentiability. All in float64 for tight tolerances.
"""

import pytest
import torch

from tensorpils.meshing import structured_quad_mesh
from tensorpils.physics import PoissonProblem
from tensorpils.preconditioners import SpectralPreconditioner, SineSpectralPreconditioner

DT = torch.float64
ATOL = 1e-9


def _problem(n: int = 8) -> PoissonProblem:
    return PoissonProblem(structured_quad_mesh(nx=n, ny=n))


def _sine(prob, n: int = 8, kind: str = "blend", strength: float = 1.0):
    return SineSpectralPreconditioner(prob.boundary_mask, nx=n, ny=n,
                                      kind=kind, strength=strength, dtype=DT)


def _interior(problem):
    mask = problem.boundary_mask.to(torch.bool)
    interior = (~mask).nonzero(as_tuple=False).squeeze(-1)
    A_II = (problem.A.to_dense().to(DT)
            .index_select(0, interior).index_select(1, interior))
    return mask, interior, A_II


@pytest.mark.parametrize("kind", ["blend", "power"])
def test_identity_endpoint(kind):
    """strength=0 => P r = r for any boundary-zeroed residual."""
    prob = _problem()
    P = SpectralPreconditioner.from_problem(prob, kind=kind, strength=0.0, dtype=DT)
    mask, _, _ = _interior(prob)

    torch.manual_seed(0)
    r = torch.randn(4, prob.n_nodes, dtype=DT)
    r[:, mask] = 0.0
    assert torch.allclose(P(r), r, atol=ATOL)


@pytest.mark.parametrize("kind", ["blend", "power"])
def test_inverse_endpoint(kind):
    """strength=1 => P (A x) = x on the interior, and P = A_II^{-1}."""
    prob = _problem()
    P = SpectralPreconditioner.from_problem(prob, kind=kind, strength=1.0, dtype=DT)
    mask, interior, A_II = _interior(prob)
    N, n_int = prob.n_nodes, interior.numel()

    # P(A x) == x
    torch.manual_seed(1)
    x_I = torch.randn(3, n_int, dtype=DT)
    r = torch.zeros(3, N, dtype=DT)
    r[:, interior] = x_I @ A_II.T                       # A_II x, embedded
    out = P(r)
    assert torch.allclose(out[:, interior], x_I, atol=ATOL)
    assert torch.all(out[:, mask] == 0)                # boundary stays zero

    # P r == A_II^{-1} r  (matches a dense solve)
    r_I = torch.randn(3, n_int, dtype=DT)
    r2 = torch.zeros(3, N, dtype=DT)
    r2[:, interior] = r_I
    ref = torch.linalg.solve(A_II, r_I.T).T
    assert torch.allclose(P(r2)[:, interior], ref, atol=ATOL)


def test_blend_is_convex_combination():
    """P_t r = (1-t) r + t A^{-1} r on the interior."""
    prob = _problem()
    t = 0.37
    P = SpectralPreconditioner.from_problem(prob, kind="blend", strength=t, dtype=DT)
    _, interior, A_II = _interior(prob)

    torch.manual_seed(2)
    r_I = torch.randn(2, interior.numel(), dtype=DT)
    r = torch.zeros(2, prob.n_nodes, dtype=DT)
    r[:, interior] = r_I
    expected = (1 - t) * r_I + t * torch.linalg.solve(A_II, r_I.T).T
    assert torch.allclose(P(r)[:, interior], expected, atol=ATOL)


@pytest.mark.parametrize("s", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_power_condition_closed_form(s):
    """cond(PA) = kappa(A)^(1-s) for the fractional-power family."""
    prob = _problem()
    _, _, A_II = _interior(prob)
    evals = torch.linalg.eigvalsh(A_II)
    kappa_A = (evals.max() / evals.min()).item()

    P = SpectralPreconditioner.from_problem(prob, kind="power", strength=s, dtype=DT)
    assert P.cond_PA == pytest.approx(kappa_A ** (1 - s), rel=1e-6)


def test_1d_input_shape():
    prob = _problem()
    P = SpectralPreconditioner.from_problem(prob, kind="power", strength=0.5, dtype=DT)
    _, interior, _ = _interior(prob)
    r = torch.zeros(prob.n_nodes, dtype=DT)
    r[interior] = torch.randn(interior.numel(), dtype=DT)
    out = P(r)
    assert out.shape == (prob.n_nodes,)


def test_differentiable():
    prob = _problem()
    P = SpectralPreconditioner.from_problem(prob, kind="power", strength=0.5, dtype=DT)
    u = torch.randn(2, prob.n_nodes, dtype=DT, requires_grad=True)
    (P(u) ** 2).sum().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()


def test_invalid_args():
    prob = _problem()
    with pytest.raises(ValueError):
        SpectralPreconditioner.from_problem(prob, kind="nope", strength=0.5)
    with pytest.raises(ValueError):
        SpectralPreconditioner.from_problem(prob, kind="blend", strength=1.5)


# ============================ sine (fast DST) realization ============================

@pytest.mark.parametrize("kind", ["blend", "power"])
@pytest.mark.parametrize("strength", [0.0, 0.25, 1.0])
def test_sine_matches_dense(kind, strength):
    """The DST realization reproduces the dense eigendecomposition preconditioner exactly."""
    n = 10
    prob = _problem(n)
    P_dense = SpectralPreconditioner.from_problem(prob, kind=kind, strength=strength, dtype=DT)
    P_sine = _sine(prob, n, kind=kind, strength=strength)

    torch.manual_seed(0)
    r = torch.randn(4, prob.n_nodes, dtype=DT)
    r[:, prob.boundary_mask.bool()] = 0.0
    assert torch.allclose(P_sine(r), P_dense(r), atol=ATOL)
    if strength > 0:                                    # cond(PA) undefined shape aside, equal here
        assert P_sine.cond_PA == pytest.approx(P_dense.cond_PA, rel=1e-6)


@pytest.mark.parametrize("kind", ["blend", "power"])
def test_sine_identity_endpoint(kind):
    """strength=0 => P r = r on the interior (boundary zeroed)."""
    n = 8
    prob = _problem(n)
    P = _sine(prob, n, kind=kind, strength=0.0)
    mask = prob.boundary_mask.bool()
    torch.manual_seed(1)
    r = torch.randn(3, prob.n_nodes, dtype=DT)
    r[:, mask] = 0.0
    assert torch.allclose(P(r), r, atol=ATOL)


@pytest.mark.parametrize("kind", ["blend", "power"])
def test_sine_inverse_endpoint(kind):
    """strength=1 => P (A x) = x on the interior."""
    n = 8
    prob = _problem(n)
    P = _sine(prob, n, kind=kind, strength=1.0)
    mask, interior, A_II = _interior(prob)
    N, n_int = prob.n_nodes, interior.numel()
    torch.manual_seed(2)
    x_I = torch.randn(3, n_int, dtype=DT)
    r = torch.zeros(3, N, dtype=DT)
    r[:, interior] = x_I @ A_II.T
    out = P(r)
    assert torch.allclose(out[:, interior], x_I, atol=ATOL)
    assert torch.all(out[:, mask] == 0)


def test_sine_1d_input_and_grad():
    n = 8
    prob = _problem(n)
    P = _sine(prob, n, kind="blend", strength=0.5)
    _, interior, _ = _interior(prob)
    r = torch.zeros(prob.n_nodes, dtype=DT, requires_grad=True)
    out = P(r + 0.0)
    assert out.shape == (prob.n_nodes,)
    (P(torch.randn(2, prob.n_nodes, dtype=DT, requires_grad=True)) ** 2).sum().backward()


def test_sine_requires_frame_mask():
    """A mask that is not the outer frame (or wrong size) is rejected."""
    prob = _problem(8)
    with pytest.raises(ValueError):                     # all-False mask: interior nodes flagged
        SineSpectralPreconditioner(torch.zeros(64, dtype=torch.bool), nx=8, ny=8)
    with pytest.raises(ValueError):                     # wrong length
        SineSpectralPreconditioner(prob.boundary_mask, nx=8, ny=9)
