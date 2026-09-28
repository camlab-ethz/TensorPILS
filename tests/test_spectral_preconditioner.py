"""Endpoint and property tests for the spectral blend preconditioner P_t = (1-t) I + t A^{-1}.

These verify the core mathematical claims of the conditioning study in isolation:
    strength=0  ->  P = I           (residual / no-preconditioner anchor)
    strength=1  ->  P = A^{-1}      (supervised anchor)
plus the convex-combination identity, the closed-form condition number of the loss Hessian,
and differentiability. All in float64 for tight tolerances.
"""

import pytest
import torch

from tensorpils.meshing import structured_quad_mesh
from tensorpils.physics import PoissonProblem
from tensorpils.preconditioners import SpectralPreconditioner

DT = torch.float64
ATOL = 1e-9


def _problem(n: int = 8) -> PoissonProblem:
    return PoissonProblem(structured_quad_mesh(nx=n, ny=n))


def _interior(problem):
    mask = problem.boundary_mask.to(torch.bool)
    interior = (~mask).nonzero(as_tuple=False).squeeze(-1)
    A_II = (problem.A.to_dense().to(DT)
            .index_select(0, interior).index_select(1, interior))
    return mask, interior, A_II


def test_identity_endpoint():
    """strength=0 => P r = r for any boundary-zeroed residual."""
    prob = _problem()
    P = SpectralPreconditioner.from_problem(prob, kind="blend", strength=0.0, dtype=DT)
    mask, _, _ = _interior(prob)

    torch.manual_seed(0)
    r = torch.randn(4, prob.n_nodes, dtype=DT)
    r[:, mask] = 0.0
    assert torch.allclose(P(r), r, atol=ATOL)


def test_inverse_endpoint():
    """strength=1 => P (A x) = x on the interior, and P = A_II^{-1}."""
    prob = _problem()
    P = SpectralPreconditioner.from_problem(prob, kind="blend", strength=1.0, dtype=DT)
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


def test_blend_condition_numbers():
    """kappa(H_t) = kappa(P_t A)^2: kappa(A)^2 at t=0, 1 at t=1, and decreasing in between."""
    prob = _problem()
    _, _, A_II = _interior(prob)
    evals = torch.linalg.eigvalsh(A_II)
    kappa_A = (evals.max() / evals.min()).item()

    conds = [SpectralPreconditioner.from_problem(prob, kind="blend", strength=t, dtype=DT).cond_H
             for t in (0.0, 0.1, 0.5, 1.0)]
    assert conds[0] == pytest.approx(kappa_A ** 2, rel=1e-6)
    assert conds[-1] == pytest.approx(1.0, rel=1e-6)
    assert all(a > b for a, b in zip(conds, conds[1:]))


def test_1d_input_shape():
    prob = _problem()
    P = SpectralPreconditioner.from_problem(prob, kind="blend", strength=0.5, dtype=DT)
    _, interior, _ = _interior(prob)
    r = torch.zeros(prob.n_nodes, dtype=DT)
    r[interior] = torch.randn(interior.numel(), dtype=DT)
    out = P(r)
    assert out.shape == (prob.n_nodes,)


def test_differentiable():
    prob = _problem()
    P = SpectralPreconditioner.from_problem(prob, kind="blend", strength=0.5, dtype=DT)
    u = torch.randn(2, prob.n_nodes, dtype=DT, requires_grad=True)
    (P(u) ** 2).sum().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()


def test_invalid_args():
    prob = _problem()
    with pytest.raises(ValueError):
        SpectralPreconditioner.from_problem(prob, kind="power", strength=0.5)
    with pytest.raises(ValueError):
        SpectralPreconditioner.from_problem(prob, kind="blend", strength=1.5)
