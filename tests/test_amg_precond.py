r"""Algebraic (AmgX) preconditioner: the invariants the ``pls`` loss rests on.

The geometric V-cycle is written in plain differentiable torch ops, so autograd derives its vjp
and there is nothing to check. The algebraic one is an opaque CUDA library call whose vjp we
*assert* --- ``P`` symmetric, therefore backward = another forward cycle. A wrong vjp does not
raise; it silently trains to a worse answer and reads as "AMG is a worse preconditioner". These
tests are what separates those two readings.

Only the boundary-convention test runs without a GPU; AmgX is CUDA-only.
"""
import numpy as np
import pytest
import torch

from tensorpils.meshing import structured_quad_mesh
from tensorpils.physics import PoissonProblem, apply_zero_boundary
from tensorpils.preconditioners import GeometricMultigrid, build_preconditioner
from tensorpils.preconditioners.algebraic import bc_eliminated_matrix
from tensorpils.losses import build_loss

# Above AMG's direct-solve regime: below ~1000 interior DOFs AmgX coarsens the problem straight
# to an exact solve, which would make every check here pass for the wrong reason.
GRID = 65


def _amgx_available():
    try:
        import torch_amgx
        return torch_amgx.is_available()
    except Exception:
        return False


requires_amgx = pytest.mark.skipif(
    not _amgx_available(),
    reason="torch-amgx (CUDA-only) not importable; see experiments/poisson/amg_dropin/README.md")


@pytest.fixture(scope="module")
def problem():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return PoissonProblem(structured_quad_mesh(nx=GRID, ny=GRID)).to(dev)


@pytest.fixture(scope="module")
def amg(problem):
    """Exactly ONE AmgX preconditioner for the whole module.

    Two live AmgX solver objects abort the process at interpreter exit (exit code 134), so the
    fixture is module-scoped by necessity, not for speed.
    """
    return build_preconditioner(kind="amg", problem=problem, grid_size=(GRID, GRID),
                                amg_sweeps=2, device="cuda")


# ----------------------------------------------------------------- CPU-only


def test_bc_eliminated_matrix_matches_geometric_level0():
    """The operator handed to AmgX is the one the geometric V-cycle inverts at its finest level.

    GMG re-discretizes a structured mesh per level; the algebraic path reads ``problem.A``
    instead, which is what lets it work off-grid. They must agree on the fine level, boundary
    convention included, or the two arms are not solving the same problem.
    """
    n = 33
    prob = PoissonProblem(structured_quad_mesh(nx=n, ny=n))
    gmg = GeometricMultigrid(nx_fine=n, ny_fine=n, n_levels=3)

    ours = bc_eliminated_matrix(prob.A, prob.boundary_mask).toarray()
    theirs = torch.sparse_coo_tensor(gmg.A_indices_0, gmg.A_values_0,
                                     (n * n, n * n)).to_dense().numpy()
    assert ours.shape == theirs.shape
    np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-6)


def test_bc_eliminated_matrix_is_identity_on_the_boundary():
    """Boundary rows/cols are the identity, so a zero-boundary residual maps to a zero-boundary
    correction and the full-grid Preconditioner contract holds."""
    n = 17
    prob = PoissonProblem(structured_quad_mesh(nx=n, ny=n))
    A = bc_eliminated_matrix(prob.A, prob.boundary_mask).toarray()
    mask = prob.boundary_mask.numpy().astype(bool)
    np.testing.assert_allclose(A[mask][:, mask], np.eye(mask.sum()), atol=1e-12)
    assert np.abs(A[mask][:, ~mask]).max() == 0.0
    assert np.abs(A[~mask][:, mask]).max() == 0.0


# ----------------------------------------------------------------- needs AmgX


@requires_amgx
def test_mismatched_sweeps_are_refused():
    """Unequal pre/post sweeps break the symmetry the vjp assumes, so the factory refuses them
    rather than returning a preconditioner with a silently wrong gradient."""
    prob = PoissonProblem(structured_quad_mesh(nx=17, ny=17))
    with pytest.raises(ValueError, match="symmetric"):
        build_preconditioner(kind="amg", problem=prob, grid_size=(17, 17),
                             mg_pre_smooth=2, mg_post_smooth=1, device="cuda")


@requires_amgx
def test_cycle_is_symmetric(amg):
    """``P == P^T`` -- the identity :class:`_AMGXApply` uses for its backward pass."""
    assert amg.symmetry_defect() < 1e-5


@requires_amgx
def test_preserves_zero_boundary(amg, problem):
    r = torch.randn(4, GRID * GRID, device="cuda")
    r = apply_zero_boundary(r, problem.boundary_mask)
    assert amg(r)[:, problem.boundary_mask].abs().max().item() == 0.0


@requires_amgx
def test_one_cycle_contracts_the_residual(amg, problem):
    """A healthy V-cycle removes most of a random residual in one pass. A broken hierarchy (or a
    preconditioner that silently returned zero) would leave the ratio at 1."""
    r = apply_zero_boundary(torch.randn(8, GRID * GRID, device="cuda"), problem.boundary_mask)
    res = r - problem._spmm(problem.A, amg(r))
    assert (res.norm(dim=1) / r.norm(dim=1)).mean().item() < 0.3


@requires_amgx
@pytest.mark.parametrize("scale", [1.0, 1e-4, 1e-8, 1e-12])
def test_apply_is_linear_across_scales(amg, problem, scale):
    r"""``P(c r) == c P(r)`` for every ``c`` -- the defining property of a linear preconditioner,
    and the regression test for AmgX's convergence check.

    With ``max_iters=1`` and the binding's default ABSOLUTE ``tolerance=1e-8``, AmgX tests for
    convergence *before* iterating, so a right-hand side whose norm is below the tolerance comes
    back as ``x = 0``. A residual getting small is exactly what a *successfully training* model
    produces, so the loss and its gradient would vanish and training would silently freeze.

    Linearity rather than mere non-zero-ness. With ``max_iters=1`` the failure measured here is a
    clean cliff (fp32 roundoff above the tolerance, exactly zero below it), but an
    iterate-to-tolerance solver on the same binding instead partially converges and is already
    ~1e-3 wrong a decade *above* its tolerance while still returning a finite, nonzero,
    correctly-shaped answer. A scale sweep catches both; "is it nonzero" catches only the first.
    ``AMGXPreconditioner`` sets ``tolerance=0``.
    """
    r = apply_zero_boundary(torch.randn(4, GRID * GRID, device="cuda"), problem.boundary_mask)
    ref = amg(r)
    out = amg(r * scale)
    assert out.abs().max() > 0, f"AmgX returned zero at scale {scale} (tolerance short-circuit)"
    # compared in a norm: at fp32 the entries nearest zero carry a meaningless relative error
    assert (out - scale * ref).norm() / (scale * ref.norm()) < 1e-5


@requires_amgx
def test_pls_loss_gradient_matches_closed_form(amg, problem):
    r"""The end-to-end check: autograd through the assembled ``pls`` loss must equal

        dL/du = (1/B) * Mask A^T Mask P^T P r,

    which with the symmetric ``P`` is ``(1/B) Mask A (P (P r))``. This is what actually validates
    ``_AMGXApply.backward``; the symmetry test alone would not catch a missing batch factor or a
    dropped boundary mask.
    """
    B, N = 4, GRID * GRID
    crit = build_loss("pls", problem, lambda_bc=0.0, precond=amg)
    torch.manual_seed(0)
    u = torch.randn(B, N, device="cuda", requires_grad=True)
    f = torch.randn(B, N, device="cuda")

    crit(u, f).backward()

    with torch.no_grad():
        r = problem.residual(u, f)
        expected = apply_zero_boundary(problem._spmm(problem.A, amg(amg(r))),
                                       problem.boundary_mask) / B
        bare = apply_zero_boundary(problem._spmm(problem.A, r), problem.boundary_mask) / B

    assert (u.grad - expected).norm() / expected.norm() < 1e-4
    # guard against the check passing because P happened to be ~identity
    assert (u.grad - bare).norm() / bare.norm() > 0.1
