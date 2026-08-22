"""The ``solution="fem"`` dataset mode: labels are the discrete solution, not the closed form.

The point of the mode is that residual-based losses target ``A u = M f`` exactly, while the
analytic labels sit a discretisation error away from that. Under ``fem`` the label *is* the
target, so the floor disappears. These tests pin both halves of that claim, and that the
default is untouched.
"""

import numpy as np
import pytest
import torch

from tensorpils.data import (FEMPoissonSolver, PoissonDataset, StreamingPoissonDataset,
                             create_datasets, create_scaling_datasets)

GRID, K, SEED = 17, 3, 0          # small but not trivial: the mode structure is what matters


# Labels are stored float32, like every field in this repo, so a perfect label still leaves a
# residual at float32 roundoff amplified by kappa(A) -- measured 2e-7 relative at 17^2 and 4e-6 at
# 65^2. The solve itself is exact (3e-14 relative in float64). Hence a RELATIVE tolerance, well
# above float32 noise and far below the discretisation error the analytic labels carry.
FEM_RTOL = 1e-5


def _residual(ds, i):
    """‖A u − M f‖ / ‖M f‖ on interior rows — ~0 iff the label solves the discrete system."""
    f, u = ds.fs[i].double().numpy(), ds.us[i].double().numpy()
    A = ds.problem.A.to_scipy_coo().tocsr()
    M = ds.problem.M.to_scipy_coo().tocsr()
    interior = ~ds.problem.boundary_mask.numpy().astype(bool)
    r = (A @ u - M @ f)[interior]
    b = (M @ f)[interior]
    return float(np.linalg.norm(r) / np.linalg.norm(b))


def test_fem_labels_solve_the_discrete_system_and_analytic_labels_do_not():
    an = PoissonDataset(num_samples=4, K=K, seed=SEED, grid_resolution=GRID)
    fe = PoissonDataset(num_samples=4, K=K, seed=SEED, grid_resolution=GRID, solution="fem")

    # Same sources either way -- only the labels differ.
    for i in range(4):
        assert torch.allclose(an.fs[i], fe.fs[i])

    for i in range(4):
        r_fem, r_analytic = _residual(fe, i), _residual(an, i)
        assert r_fem < FEM_RTOL, "fem labels must solve A u = M f"
        # The analytic label misses by the discretisation error, orders above float32 noise.
        assert r_analytic > 1e-3
        assert r_analytic > 100 * r_fem


def test_fem_labels_vanish_on_the_dirichlet_boundary():
    fe = PoissonDataset(num_samples=2, K=K, seed=SEED, grid_resolution=GRID, solution="fem")
    mask = fe.problem.boundary_mask.numpy().astype(bool)
    for i in range(2):
        assert np.abs(fe.us[i].numpy()[mask]).max() == 0.0


def test_default_is_unchanged():
    """Guards the promise that adding the flag changed nothing for existing runs."""
    a = PoissonDataset(num_samples=3, K=K, seed=SEED, grid_resolution=GRID)
    b = PoissonDataset(num_samples=3, K=K, seed=SEED, grid_resolution=GRID, solution="analytic")
    assert a.solution == "analytic"
    for i in range(3):
        assert torch.equal(a.us[i], b.us[i]) and torch.equal(a.fs[i], b.fs[i])


def test_streaming_and_finite_agree_under_fem():
    """The stream must produce the same labels as the finite dataset for the same draw."""
    fin = PoissonDataset(num_samples=8, K=K, seed=SEED, grid_resolution=GRID, solution="fem")
    st = StreamingPoissonDataset(samples_per_epoch=8, K=K, seed=SEED,
                                 grid_resolution=GRID, solution="fem")
    fs, us = st._generate(8, torch.Generator().manual_seed(SEED))
    # The residual is the invariant that must hold for every streamed sample, whatever the draw.
    A = st.problem.A.to_scipy_coo().tocsr()
    M = st.problem.M.to_scipy_coo().tocsr()
    interior = ~st.problem.boundary_mask.numpy().astype(bool)
    for i in range(8):
        r = (A @ us[i].double().numpy() - M @ fs[i].double().numpy())[interior]
        b = (M @ fs[i].double().numpy())[interior]
        assert np.linalg.norm(r) / np.linalg.norm(b) < FEM_RTOL
    assert fin.us[0].shape == us[0].shape


@pytest.mark.parametrize("factory", ["create_datasets", "create_scaling_datasets"])
def test_factories_apply_the_mode_to_every_split(factory):
    """train, val AND test must carry the mode -- an eval split on analytic labels would
    reintroduce exactly the floor the mode exists to remove."""
    kw = dict(n_train=4, n_val=2, n_test=2, K=K, grid_resolution=GRID, seed=SEED)
    fn = {"create_datasets": create_datasets,
          "create_scaling_datasets": create_scaling_datasets}[factory]
    for ds in fn(solution="fem", **kw):
        assert ds.solution == "fem"
        assert _residual(ds, 0) < FEM_RTOL
    for ds in fn(**kw):
        assert ds.solution == "analytic"


def test_solver_matches_a_dense_solve():
    """FEMPoissonSolver against an independent dense solve of the same system."""
    ds = PoissonDataset(num_samples=2, K=K, seed=SEED, grid_resolution=GRID)
    solver = FEMPoissonSolver(ds.problem)
    u = solver(torch.stack([ds.fs[0], ds.fs[1]]))
    A = ds.problem.A.to_scipy_coo().tocsr().toarray()
    M = ds.problem.M.to_scipy_coo().tocsr().toarray()
    mask = ds.problem.boundary_mask.numpy().astype(bool)
    idx = np.flatnonzero(~mask)
    for i in range(2):
        want = np.zeros(mask.size)
        want[idx] = np.linalg.solve(A[np.ix_(idx, idx)], (M @ ds.fs[i].double().numpy())[idx])
        assert np.abs(u[i].numpy() - want).max() < 1e-6 * max(np.abs(want).max(), 1e-30)


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="solution must be one of"):
        PoissonDataset(num_samples=1, K=K, seed=SEED, grid_resolution=GRID, solution="exact")


class _TinyNet(torch.nn.Module):
    """Minimal grid-in/grid-out stand-in for the FNO -- this test is about provenance, not skill."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


def test_results_json_records_the_label_mode(tmp_path):
    """Provenance: a run must say which labels it used.

    Analytic and FEM runs share a run prefix, so they overwrite each other and are otherwise
    indistinguishable on disk -- mtime was the only signal, which is not provenance.
    """
    import json as _json

    from tensorpils.trainer import PoissonTrainer

    for mode in ("analytic", "fem"):
        out = tmp_path / mode
        tr, va, te = create_datasets(n_train=4, n_val=2, n_test=2, K=K,
                                     grid_resolution=GRID, seed=SEED, solution=mode)
        PoissonTrainer(model=_TinyNet(), train_dataset=tr, val_dataset=va, test_dataset=te,
                       loss_type="data", batch_size=2, epochs=1,
                       device="cpu", output_dir=str(out)).train()
        hits = sorted((out / "results").glob("*.json"))
        assert len(hits) == 1, hits
        assert _json.load(open(hits[0]))["dataset_solution"] == mode
