"""Datasets: analytical multi-frequency PDE fields on a structured grid.

* Poisson (static): source ``f`` and solution ``u`` from ``PoissonMultiFrequency`` (``r=-0.5``).
* Wave (time-dependent): the analytical trajectory ``u(t_k)`` of :math:`u_{tt}=c^2\\Delta u`
  from ``WaveMultiFrequency``, sampled on a uniform time grid.
* Allen–Cahn (time-dependent, nonlinear): a multi-frequency initial condition (reusing
  ``WaveMultiFrequency.initial_condition``) evolved by a FEM implicit-Euler + Newton reference
  solver (:meth:`ACProblem.fem_reference`), since Allen–Cahn has no analytical solution.

Every sample is exposed both as an image-shaped grid (for the FNO) and as the flat node
vector (for the FEM losses, which operate on node values via the physics operators).
"""

from typing import List, Optional

import numpy as np
import scipy.sparse.linalg as spla
import torch
from torch.utils.data import Dataset, IterableDataset

from tensormesh.dataset import PoissonMultiFrequency, WaveMultiFrequency

from .meshing import (structured_quad_mesh, structured_quad9_mesh, circle_mesh,
                      obstacle_mesh, node_to_grid)
from .physics import (PoissonProblem, WaveProblem, ACProblem, StokesProblem,
                      UnstructuredStokesProblem)

__all__ = ["FEMPoissonSolver", "PoissonDataset", "StreamingPoissonDataset", "create_datasets",
           "UnstructuredPoissonDataset", "create_unstructured_datasets",
           "create_scaling_datasets", "WaveDataset", "create_wave_datasets",
           "ACDataset", "create_ac_datasets",
           "StokesDataset", "create_stokes_datasets", "stokes_body_force",
           "UnstructuredStokesDataset", "create_unstructured_stokes_datasets"]


SOLUTION_MODES = ("analytic", "fem")


class FEMPoissonSolver:
    r"""Exact $Q_1$ FEM solve of :math:`A u = M f`, zero on the Dirichlet boundary.

    Backs ``solution="fem"``: labels become the *discrete* solution the FEM losses actually
    target, rather than the closed form ``PoissonMultiFrequency.solution`` sampled at nodes.
    The two differ by the discretisation error -- ~0.7 % relative at ``65^2`` with ``K=10`` --
    which is a floor the residual-based losses can never cross while the analytic labels are
    the reference. With ``fem`` labels that floor is gone, because the label *is* the target.

    **Why a sparse LU rather than the DST.** On a uniform grid the discrete sines are exact
    eigenvectors of the $Q_1$ stiffness, so a DST diagonalises it and is ~2.7x faster
    (measured, agreeing to 2.4e-14). It is not used, for two reasons: this factorises the
    *assembled* ``problem.A``, so a label can never silently disagree with the operator the
    losses use; and the DST would need the mesh to stay uniform, rectangular and $Q_1$, failing
    silently rather than loudly if that changed. The cost is ~2.3 min per million samples
    against ~0.8 -- negligible next to training.

    The factorisation is built once and reused for every sample, which is what keeps the
    streaming dataset affordable.
    """

    def __init__(self, problem: PoissonProblem):
        A = problem.A.to_scipy_coo().tocsr()
        self.M = problem.M.to_scipy_coo().tocsr()
        mask = problem.boundary_mask.numpy().astype(bool)
        self.interior = np.flatnonzero(~mask)
        self.n_nodes = mask.size
        self.lu = spla.splu(A[self.interior][:, self.interior].tocsc())

    def __call__(self, fs: torch.Tensor) -> torch.Tensor:
        """``[n, N]`` nodal source values -> ``[n, N]`` FEM solution, exactly 0 on the boundary."""
        f = fs.detach().cpu().numpy().astype(np.float64)
        b = (self.M @ f.T).T[:, self.interior]          # consistent load M f, interior rows
        u = np.zeros((f.shape[0], self.n_nodes), dtype=np.float64)
        u[:, self.interior] = self.lu.solve(b.T).T
        return torch.from_numpy(u).float()


def _check_solution_mode(solution: str) -> str:
    if solution not in SOLUTION_MODES:
        raise ValueError(f"solution must be one of {SOLUTION_MODES}, got {solution!r}")
    return solution


class PoissonDataset(Dataset):
    """Dataset of (source, solution) pairs on a structured quad grid.

    ``solution="analytic"`` (default) labels with the closed form; ``"fem"`` labels with the
    $Q_1$ FEM solution of the same source -- see :class:`FEMPoissonSolver` for why that matters.

    Each item is ``((f_grid [1,H,W], grid_size, f_node [N], u_node [N]), u_grid [H,W])``,
    where ``H = ny``, ``W = nx`` and ``N = nx*ny``. The grid form feeds the FNO; the node
    form feeds the FEM-based losses.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        K: int = 4,
        seed: int = 42,
        grid_resolution: int = 64,
        mesh=None,
        problem: Optional[PoissonProblem] = None,
        all_data: Optional[dict] = None,
        indices: Optional[List[int]] = None,
        solution: str = "analytic",
    ):
        super().__init__()
        self.K = K
        self.solution = _check_solution_mode(solution)
        self.grid_resolution = grid_resolution
        self.grid_size = (grid_resolution, grid_resolution)   # (nx, ny)

        if all_data is not None:
            # Reuse mesh/problem/fields generated by a sibling split.
            self.mesh = mesh
            self.problem = problem
            if indices is not None:
                self.fs = [all_data["fs"][i] for i in indices]
                self.us = [all_data["us"][i] for i in indices]
                self.l_a = all_data["l_a"][indices]
            else:
                self.fs = all_data["fs"]
                self.us = all_data["us"]
                self.l_a = all_data["l_a"]
            self._all_data = all_data
        else:
            torch.manual_seed(seed)
            self.mesh = mesh if mesh is not None else \
                structured_quad_mesh(nx=grid_resolution, ny=grid_resolution)
            self.problem = problem if problem is not None else PoissonProblem(self.mesh)

            # Random K x K coefficients in [-1, 1]
            self.l_a = (torch.rand(num_samples, K, K) * 2 - 1)
            equation = PoissonMultiFrequency(a=self.l_a, r=-0.5)
            points = self.mesh.points                              # [N, 2], float64
            all_fs = equation.source_term(points, domain="rectangle").float()   # [num, N]
            if self.solution == "fem":
                all_us = FEMPoissonSolver(self.problem)(all_fs)                 # [num, N]
            else:
                all_us = equation.solution(points).float()                      # [num, N]

            self.fs = [all_fs[i] for i in range(num_samples)]
            self.us = [all_us[i] for i in range(num_samples)]
            self._all_data = {"fs": self.fs, "us": self.us, "l_a": self.l_a}

    def __len__(self):
        return len(self.fs)

    def __getitem__(self, idx):
        f = self.fs[idx]                       # [N]
        u = self.us[idx]                       # [N]
        nx, ny = self.grid_size
        f_grid = node_to_grid(f, nx, ny).unsqueeze(0)   # [1, H=ny, W=nx]
        u_grid = node_to_grid(u, nx, ny)                # [H, W]
        return (f_grid, self.grid_size, f, u), u_grid

    def get_shared_resources(self):
        return self.mesh, self.problem, self._all_data


def create_datasets(n_train: int, n_val: int, n_test: int,
                    K: int, grid_resolution: int = 64, seed: int = 42,
                    solution: str = "analytic"):
    """Build train/val/test splits that share one mesh, one :class:`PoissonProblem`,
    and a single pool of generated samples."""
    total = n_train + (n_val if n_val > 0 else 0) + (n_test if n_test > 0 else 0)
    base = PoissonDataset(num_samples=total, K=K, seed=seed,
                          grid_resolution=grid_resolution, solution=solution)
    mesh, problem, all_data = base.get_shared_resources()

    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_train + n_val)) if n_val > 0 else train_idx
    if n_test > 0:
        start = n_train + (n_val if n_val > 0 else 0)
        test_idx = list(range(start, start + n_test))
    else:
        test_idx = train_idx

    def _make(indices):
        return PoissonDataset(
            num_samples=total, K=K, seed=seed, grid_resolution=grid_resolution,
            mesh=mesh, problem=problem, all_data=all_data, indices=indices,
            solution=solution,
        )
    return _make(train_idx), _make(val_idx), _make(test_idx)


class UnstructuredPoissonDataset(Dataset):
    r"""Poisson on an **unstructured** mesh — the same operator, no grid.

    Items are node vectors only: ``(f_node [N], u_node [N])``. There is no image form, which is
    the point — this is the dataset the FNO cannot consume and
    :class:`~tensorpils.gaot.GAOTModel` can, via ``forward_nodes``.

    Two things change relative to :class:`PoissonDataset`, and both are forced by the geometry
    rather than chosen:

    **Labels must be the FEM solution.** ``PoissonMultiFrequency`` is a sum of
    ``sin(i pi x) sin(j pi y)``; it vanishes on the boundary of the unit *square*, and on any
    other domain it does not. On the inscribed disc the closed form reaches ~90 % of its
    interior peak *on the boundary*, so it is not a solution of the problem being posed and
    labelling with it would be silently wrong. :class:`FEMPoissonSolver` already factorises the
    assembled ``A`` and never looked at the grid, so it carries over untouched — and the label
    it returns is exactly the discrete solution the residual losses target, which removes the
    discretisation floor as well.

    **The source is still evaluated from the closed form.** ``source_term`` is a pointwise
    formula, so it is defined wherever the nodes are; it only needs the points to lie in
    ``[0,1]^2``, which the default inscribed disc satisfies. Keeping ``K`` and ``r=-0.5``
    identical to the structured dataset is what makes the two runs comparable: same source
    distribution, same operator, different domain and different mesh.

    Parameters
    ----------
    num_samples, K, seed : as :class:`PoissonDataset`.
    chara_length : float
        Gmsh target edge length (``0.015`` -> ~4200 nodes, the ``64^2`` node budget).
    mesh, problem, all_data, indices :
        Shared-resource plumbing, so train/val/test reuse one mesh, one
        :class:`~tensorpils.physics.PoissonProblem` and one generated pool.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        K: int = 4,
        seed: int = 42,
        chara_length: float = 0.015,
        cx: float = 0.5,
        cy: float = 0.5,
        radius: float = 0.5,
        mesh=None,
        problem: Optional[PoissonProblem] = None,
        all_data: Optional[dict] = None,
        indices: Optional[List[int]] = None,
        cache_path: Optional[str] = None,
    ):
        super().__init__()
        self.K = K
        self.chara_length = chara_length
        self.solution = "fem"          # the only valid choice here; see the class docstring

        if all_data is not None:
            self.mesh = mesh
            self.problem = problem
            if indices is not None:
                self.fs = [all_data["fs"][i] for i in indices]
                self.us = [all_data["us"][i] for i in indices]
                self.l_a = all_data["l_a"][indices]
            else:
                self.fs, self.us, self.l_a = all_data["fs"], all_data["us"], all_data["l_a"]
            self._all_data = all_data
        else:
            torch.manual_seed(seed)
            self.mesh = mesh if mesh is not None else circle_mesh(
                chara_length=chara_length, cx=cx, cy=cy, r=radius, cache_path=cache_path)
            self.problem = problem if problem is not None else PoissonProblem(self.mesh)

            self.l_a = (torch.rand(num_samples, K, K) * 2 - 1)
            equation = PoissonMultiFrequency(a=self.l_a, r=-0.5)
            points = self.mesh.points                                   # [N, 2], float64
            all_fs = equation.source_term(points, domain="rectangle").float()
            all_us = FEMPoissonSolver(self.problem)(all_fs)             # the only valid label
            self.fs = [all_fs[i] for i in range(num_samples)]
            self.us = [all_us[i] for i in range(num_samples)]
            self._all_data = {"fs": self.fs, "us": self.us, "l_a": self.l_a}

        self.n_nodes = self.mesh.points.shape[0]
        # ``grid_size`` is what the trainers key their file names and eval metric off. There is
        # no grid, so it is None -- and anything that silently assumed a grid now raises.
        self.grid_size = None

    def __len__(self):
        return len(self.fs)

    def __getitem__(self, idx):
        return self.fs[idx], self.us[idx]

    def get_shared_resources(self):
        return self.mesh, self.problem, self._all_data


def create_unstructured_datasets(n_train: int, n_val: int, n_test: int,
                                 K: int, chara_length: float = 0.015, seed: int = 42,
                                 cx: float = 0.5, cy: float = 0.5, radius: float = 0.5,
                                 cache_path: Optional[str] = None):
    """Train/val/test splits over one mesh, one problem and one generated pool.

    The FEM solve is done once for the whole pool (one sparse factorisation, many right-hand
    sides), so the extra cost over the structured analytic dataset is a single LU.
    """
    total = n_train + (n_val if n_val > 0 else 0) + (n_test if n_test > 0 else 0)
    base = UnstructuredPoissonDataset(num_samples=total, K=K, seed=seed,
                                      chara_length=chara_length, cx=cx, cy=cy, radius=radius,
                                      cache_path=cache_path)
    mesh, problem, all_data = base.get_shared_resources()

    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_train + n_val)) if n_val > 0 else train_idx
    if n_test > 0:
        start = n_train + (n_val if n_val > 0 else 0)
        test_idx = list(range(start, start + n_test))
    else:
        test_idx = train_idx

    def _make(indices):
        return UnstructuredPoissonDataset(
            num_samples=total, K=K, seed=seed, chara_length=chara_length,
            mesh=mesh, problem=problem, all_data=all_data, indices=indices)
    return _make(train_idx), _make(val_idx), _make(test_idx)


class StreamingPoissonDataset(IterableDataset):
    """Infinite Poisson dataset: every sample is freshly drawn, none is ever seen twice.

    Items have the exact format of :class:`PoissonDataset` items. One pass of ``__iter__``
    yields ``samples_per_epoch`` samples (a "virtual epoch"); the coefficient generator's
    state persists across passes, so successive epochs continue the i.i.d. stream from the
    source distribution rather than replaying it. Samples are generated in chunks via the
    same closed-form ``PoissonMultiFrequency`` fields (manufactured solutions — no solve).

    A small fixed set of preview samples (an independent generator, not part of the stream)
    backs ``__getitem__`` so the end-of-training sample visualization keeps working.
    """

    def __init__(
        self,
        samples_per_epoch: int,
        K: int = 4,
        seed: int = 42,
        grid_resolution: int = 64,
        mesh=None,
        problem: Optional[PoissonProblem] = None,
        chunk_size: int = 256,
        n_preview: int = 8,
        solution: str = "analytic",
    ):
        super().__init__()
        self.solution = _check_solution_mode(solution)
        self.samples_per_epoch = samples_per_epoch
        self.K = K
        self.grid_resolution = grid_resolution
        self.grid_size = (grid_resolution, grid_resolution)   # (nx, ny)
        self.chunk_size = chunk_size

        self.mesh = mesh if mesh is not None else \
            structured_quad_mesh(nx=grid_resolution, ny=grid_resolution)
        self.problem = problem if problem is not None else PoissonProblem(self.mesh)

        # Factorised once here, not per chunk: the stream draws ~10^6 samples over a run.
        self._fem = FEMPoissonSolver(self.problem) if self.solution == "fem" else None

        self._generator = torch.Generator().manual_seed(seed)
        self._preview = self._generate(n_preview, torch.Generator().manual_seed(seed + 1))

    def _generate(self, n: int, generator: torch.Generator):
        """Draw ``n`` fresh (f, u) pairs: random K x K coefficients in [-1, 1] -> closed forms."""
        l_a = torch.rand(n, self.K, self.K, generator=generator) * 2 - 1
        equation = PoissonMultiFrequency(a=l_a, r=-0.5)
        points = self.mesh.points                                        # [N, 2], float64
        fs = equation.source_term(points, domain="rectangle").float()     # [n, N]
        us = self._fem(fs) if self._fem is not None else equation.solution(points).float()
        return fs, us

    def _item(self, f: torch.Tensor, u: torch.Tensor):
        nx, ny = self.grid_size
        f_grid = node_to_grid(f, nx, ny).unsqueeze(0)   # [1, H=ny, W=nx]
        u_grid = node_to_grid(u, nx, ny)                # [H, W]
        return (f_grid, self.grid_size, f, u), u_grid

    def __iter__(self):
        remaining = self.samples_per_epoch
        while remaining > 0:
            n = min(self.chunk_size, remaining)
            fs, us = self._generate(n, self._generator)
            for i in range(n):
                yield self._item(fs[i], us[i])
            remaining -= n

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        """Fixed preview samples — for visualization only, not part of the training stream."""
        fs, us = self._preview
        return self._item(fs[idx % len(fs)], us[idx % len(us)])


def create_scaling_datasets(n_train: int, n_val: int, n_test: int,
                            K: int, grid_resolution: int = 64, seed: int = 42,
                            stream_samples_per_epoch: Optional[int] = None,
                            solution: str = "analytic"):
    """Splits for a dataset-size sweep: val/test are drawn from dedicated seeds
    (``seed+1`` / ``seed+2``), so they are **identical for every** ``n_train`` — all runs
    of the sweep (including the streaming one) are scored on the same eval sets, making
    the comparisons paired rather than independently noisy.

    The train split uses ``seed``: finite by default (note the sets are nested across
    ``n_train`` values, which couples the draws and smooths the scaling curve), or the
    infinite stream when ``stream_samples_per_epoch`` is given (then ``n_train`` is ignored).
    """
    # The mode applies to val/test as well as train: with "fem" the eval metric measures against
    # the discrete solution, which is what removes the residual losses' discretisation floor.
    test_ds = PoissonDataset(num_samples=n_test, K=K, seed=seed + 2,
                             grid_resolution=grid_resolution, solution=solution)
    mesh, problem, _ = test_ds.get_shared_resources()
    val_ds = PoissonDataset(num_samples=n_val, K=K, seed=seed + 1,
                            grid_resolution=grid_resolution, mesh=mesh, problem=problem,
                            solution=solution)
    if stream_samples_per_epoch is not None:
        train_ds = StreamingPoissonDataset(
            samples_per_epoch=stream_samples_per_epoch, K=K, seed=seed,
            grid_resolution=grid_resolution, mesh=mesh, problem=problem, solution=solution)
    else:
        train_ds = PoissonDataset(num_samples=n_train, K=K, seed=seed,
                                  grid_resolution=grid_resolution, mesh=mesh, problem=problem,
                                  solution=solution)
    return train_ds, val_ds, test_ds


class WaveDataset(Dataset):
    """Dataset of analytical wave trajectories on a structured quad grid.

    Each item is ``(traj_grid [T+1, H, W], traj_node [T+1, N])`` where ``T = n_steps``,
    ``H = ny``, ``W = nx``, ``N = nx*ny``, and ``traj[k] = u(·, k·dt)`` is the analytical
    multi-frequency solution of :math:`u_{tt}=c^2\\Delta u`. The grid form feeds the FNO
    (which the trainer seeds with the first two frames and rolls forward); the node form
    feeds the central-difference Galerkin residual via :class:`WaveProblem`.

    The wave speed ``c``, time step ``dt`` and horizon ``n_steps`` are dataset-level
    scalars (shared by all samples); samples differ only in their frequency coefficients.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        K: int = 4,
        seed: int = 42,
        grid_resolution: int = 64,
        dt: float = 0.005,
        n_steps: int = 20,
        c: float = 1.0,
        r: float = 0.5,
        mesh=None,
        problem: Optional[WaveProblem] = None,
        all_data: Optional[dict] = None,
        indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.K = K
        self.grid_resolution = grid_resolution
        self.grid_size = (grid_resolution, grid_resolution)   # (nx, ny)
        self.dt = dt
        self.n_steps = n_steps
        self.c = c
        self.r = r

        if all_data is not None:
            # Reuse mesh/problem/fields generated by a sibling split.
            self.mesh = mesh
            self.problem = problem
            if indices is not None:
                self.trajs = [all_data["trajs"][i] for i in indices]
                self.l_a = all_data["l_a"][indices]
            else:
                self.trajs = all_data["trajs"]
                self.l_a = all_data["l_a"]
            self._all_data = all_data
        else:
            torch.manual_seed(seed)
            self.mesh = structured_quad_mesh(nx=grid_resolution, ny=grid_resolution)
            self.problem = WaveProblem(self.mesh)

            # Random K x K coefficients in [-1, 1] (one field per sample).
            self.l_a = (torch.rand(num_samples, K, K) * 2 - 1)
            equation = WaveMultiFrequency(a=self.l_a, c=c, r=r)
            points = self.mesh.points                              # [N, 2], float64
            # Analytical trajectory: [n_steps+1, num_samples, N] -> [num, T+1, N].
            frames = [equation.solution(points, dt * k).float() for k in range(n_steps + 1)]
            all_trajs = torch.stack(frames, dim=1)                 # [num, T+1, N]

            self.trajs = [all_trajs[i] for i in range(num_samples)]
            self._all_data = {"trajs": self.trajs, "l_a": self.l_a}

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        traj_node = self.trajs[idx]                    # [T+1, N]
        nx, ny = self.grid_size
        traj_grid = node_to_grid(traj_node, nx, ny)    # [T+1, H=ny, W=nx]
        return traj_grid, traj_node

    def get_shared_resources(self):
        return self.mesh, self.problem, self._all_data


def create_wave_datasets(n_train: int, n_val: int, n_test: int,
                         K: int, grid_resolution: int = 64,
                         dt: float = 0.005, n_steps: int = 20,
                         c: float = 1.0, r: float = 0.5, seed: int = 42):
    """Build train/val/test wave splits that share one mesh, one :class:`WaveProblem`,
    and a single pool of generated trajectories."""
    total = n_train + (n_val if n_val > 0 else 0) + (n_test if n_test > 0 else 0)
    base = WaveDataset(num_samples=total, K=K, seed=seed, grid_resolution=grid_resolution,
                       dt=dt, n_steps=n_steps, c=c, r=r)
    mesh, problem, all_data = base.get_shared_resources()

    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_train + n_val)) if n_val > 0 else train_idx
    if n_test > 0:
        start = n_train + (n_val if n_val > 0 else 0)
        test_idx = list(range(start, start + n_test))
    else:
        test_idx = train_idx

    def _make(indices):
        return WaveDataset(
            num_samples=total, K=K, seed=seed, grid_resolution=grid_resolution,
            dt=dt, n_steps=n_steps, c=c, r=r,
            mesh=mesh, problem=problem, all_data=all_data, indices=indices,
        )
    return _make(train_idx), _make(val_idx), _make(test_idx)


class ACDataset(Dataset):
    """Dataset of Allen–Cahn trajectories on a structured quad grid.

    Each item is ``(traj_grid [T+1, H, W], traj_node [T+1, N])`` where ``T = n_steps`` and
    ``traj[k] = u(·, k·dt)`` is the FEM implicit-Euler reference evolution of a multi-frequency
    initial condition under :math:`u_t = a^2\\Delta u + \\epsilon^2 u(1-u^2)` (zero Dirichlet).
    The grid form feeds the FNO (seeded with frame 0 and rolled forward); the node form feeds
    the backward-Euler Galerkin residual via :class:`ACProblem`.

    The diffusion ``a``, reaction strength ``eps``, time step ``dt`` and horizon ``n_steps`` are
    dataset-level scalars (shared by all samples); samples differ only in their initial condition.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        K: int = 4,
        seed: int = 42,
        grid_resolution: int = 64,
        dt: float = 1e-3,
        n_steps: int = 20,
        a: float = 1.0,
        eps: float = 2.0,
        r: float = 0.5,
        newton_tol: float = 1e-8,
        newton_max: int = 20,
        ref_chunk: int = 64,
        integrator: str = "convex_concave",
        ref_device: str = "cpu",
        mesh=None,
        problem: Optional[ACProblem] = None,
        all_data: Optional[dict] = None,
        indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.K = K
        self.grid_resolution = grid_resolution
        self.grid_size = (grid_resolution, grid_resolution)   # (nx, ny)
        self.dt = dt
        self.n_steps = n_steps
        self.a = a
        self.eps = eps
        self.r = r
        self.integrator = integrator                          # time discretisation of the reaction

        if all_data is not None:
            # Reuse mesh/problem/fields generated by a sibling split.
            self.mesh = mesh
            self.problem = problem
            if indices is not None:
                self.trajs = [all_data["trajs"][i] for i in indices]
                self.l_a = all_data["l_a"][indices]
            else:
                self.trajs = all_data["trajs"]
                self.l_a = all_data["l_a"]
            self._all_data = all_data
        else:
            torch.manual_seed(seed)
            self.mesh = structured_quad_mesh(nx=grid_resolution, ny=grid_resolution)
            self.problem = ACProblem(self.mesh)

            # Random K x K coefficients in [-1, 1]; multi-frequency initial condition.
            self.l_a = (torch.rand(num_samples, K, K) * 2 - 1)
            points = self.mesh.points                               # [N, 2], float64
            # WaveMultiFrequency.initial_condition broadcasts to [chunk, N, K, K] before summing the
            # K^2 modes, so doing all samples at once costs num*N*K^2 (~47 GB at K=16, 128^2, 1408
            # samples -> host OOM). Chunk over samples (reusing ref_chunk) to cap it at ref_chunk*N*K^2;
            # the per-sample computation is independent, so the result is identical to the single call.
            u0 = torch.cat([
                WaveMultiFrequency(a=self.l_a[s:s + ref_chunk], r=r).initial_condition(points).float()
                for s in range(0, num_samples, ref_chunk)
            ], dim=0)                                               # [num, N]
            # FEM implicit-Euler + Newton reference trajectory (no analytical solution exists).
            # ref_device="cpu" (default) solves each Newton system with scipy's sparse direct solver:
            # exact and deterministic. On CUDA without cupy, TensorMesh falls back to torch_sla's
            # iterative PBiCGStab, which breaks down nondeterministically and has produced NaN
            # reference trajectories (the data-driven AC finals at seeds 43/44). Results stored on CPU.
            solve_dev = "cuda" if ref_device == "cuda" and torch.cuda.is_available() else "cpu"
            all_trajs = self.problem.fem_reference(
                u0.to(solve_dev), a=a, eps=eps, dt=dt, n_steps=n_steps,
                newton_tol=newton_tol, newton_max=newton_max, chunk=ref_chunk,
                integrator=integrator,
            ).cpu()                                                 # [num, T+1, N]
            # A single non-finite label poisons data-driven training (NaN loss from the first batch)
            # and any validation/test metric it enters, silently. Refuse to build instead.
            bad = (~torch.isfinite(all_trajs)).flatten(1).any(1).nonzero().flatten().tolist()
            if bad:
                raise RuntimeError(
                    f"{len(bad)} of {num_samples} FEM reference trajectories are non-finite "
                    f"(samples {bad[:10]}{' ...' if len(bad) > 10 else ''}; solved on {solve_dev}). "
                    "Refusing to build the dataset. "
                    + ("Rebuild with ref_device='cpu' (--ac_ref_device cpu)." if solve_dev == "cuda" else
                       "The direct solver should not do this: inspect these initial conditions."))

            self.trajs = [all_trajs[i] for i in range(num_samples)]
            self._all_data = {"trajs": self.trajs, "l_a": self.l_a}

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        traj_node = self.trajs[idx]                    # [T+1, N]
        nx, ny = self.grid_size
        traj_grid = node_to_grid(traj_node, nx, ny)    # [T+1, H=ny, W=nx]
        return traj_grid, traj_node

    def get_shared_resources(self):
        return self.mesh, self.problem, self._all_data


def create_ac_datasets(n_train: int, n_val: int, n_test: int,
                       K: int, grid_resolution: int = 64,
                       dt: float = 1e-3, n_steps: int = 20,
                       a: float = 1.0, eps: float = 2.0, r: float = 0.5,
                       newton_tol: float = 1e-8, newton_max: int = 20,
                       ref_chunk: int = 64, integrator: str = "convex_concave",
                       seed: int = 42, ref_device: str = "cpu"):
    """Build train/val/test Allen–Cahn splits that share one mesh, one :class:`ACProblem`,
    and a single pool of FEM reference trajectories (solved on ``ref_device``; see ACDataset)."""
    total = n_train + (n_val if n_val > 0 else 0) + (n_test if n_test > 0 else 0)
    base = ACDataset(num_samples=total, K=K, seed=seed, grid_resolution=grid_resolution,
                     dt=dt, n_steps=n_steps, a=a, eps=eps, r=r,
                     newton_tol=newton_tol, newton_max=newton_max, ref_chunk=ref_chunk,
                     integrator=integrator, ref_device=ref_device)
    mesh, problem, all_data = base.get_shared_resources()

    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_train + n_val)) if n_val > 0 else train_idx
    if n_test > 0:
        start = n_train + (n_val if n_val > 0 else 0)
        test_idx = list(range(start, start + n_test))
    else:
        test_idx = train_idx

    def _make(indices):
        return ACDataset(
            num_samples=total, K=K, seed=seed, grid_resolution=grid_resolution,
            dt=dt, n_steps=n_steps, a=a, eps=eps, r=r, integrator=integrator,
            mesh=mesh, problem=problem, all_data=all_data, indices=indices,
        )
    return _make(train_idx), _make(val_idx), _make(test_idx)


# ================================ Stokes ================================

def stokes_body_force(coeffs: torch.Tensor, points: torch.Tensor,
                      r: float = -0.5) -> torch.Tensor:
    r"""Random multi-frequency body force ``f`` on the unit square — ``[B, N, 2]``.

    Each component is a sine series
    :math:`f_c(x,y)=\sum_{k,l=1}^{K} a_{ckl}\,(k^2+l^2)^{r}\sin(k\pi x)\sin(l\pi y)`,
    the vector-valued analogue of TensorMesh's ``PoissonMultiFrequency`` source (same
    ``r=-0.5`` spectral decay, so the forcing is smooth and dominated by low modes).

    Unlike the Poisson / wave datasets there is no manufactured solution here: the Stokes
    reference is the *discrete* Taylor-Hood solve of this ``f``
    (:meth:`~tensorpils.physics.StokesProblem.fem_reference`), which is exactly the state
    the label-free residual drives toward. ``f`` therefore needs no boundary or
    divergence constraint of its own.

    Parameters
    ----------
    coeffs : torch.Tensor
        ``[B, 2, K, K]`` coefficients (component, k, l).
    points : torch.Tensor
        ``[N, 2]`` node coordinates.
    """
    B, _, K, _ = coeffs.shape
    x, y = points[..., 0], points[..., 1]                                # [N]
    ks = torch.arange(1, K + 1, dtype=points.dtype, device=points.device)
    sx = torch.sin(torch.pi * ks[:, None] * x[None, :])                  # [K, N]
    sy = torch.sin(torch.pi * ks[:, None] * y[None, :])                  # [K, N]
    decay = ((ks[:, None] ** 2 + ks[None, :] ** 2) ** r)                 # [K, K]
    w = coeffs.to(points.dtype) * decay                                  # [B, 2, K, K]
    # f[b, c, n] = sum_{k,l} w[b,c,k,l] * sx[k,n] * sy[l,n]
    f = torch.einsum("bckl,kn,ln->bcn", w, sx, sy)                       # [B, 2, N]
    return f.permute(0, 2, 1).contiguous()                               # [B, N, 2]


def _rms(t: torch.Tensor) -> float:
    """Root-mean-square magnitude of a field, over samples and nodes."""
    return float(t.pow(2).mean().sqrt().clamp_min(1e-30))


class StokesDataset(Dataset):
    r"""Dataset of Taylor-Hood Stokes solutions on a structured Q2/Q1 grid.

    Item: ``(f_grid [2, Ny, Nx], f_node [n_u, 2], u_node [n_u, 2], p_node [n_p])`` —
    a random body force and the **discrete** velocity/pressure that solve
    :math:`\mathcal K c = (M_u f, 0)`.

    ``grid_resolution`` is the **velocity** (fine) grid and must be odd; the pressure grid
    is ``(grid_resolution + 1) // 2``.

    Scaling
    -------
    Each sample is first rescaled so its reference velocity has unit FE :math:`L^2` norm.
    Stokes is linear, so scaling ``f``, ``u`` and ``p`` by one constant is exact.

    That fixes velocity but *not* the other two: the three magnitudes are tied by the
    physics and cannot be normalized independently — for this forcing
    :math:`u\sim f/(\mu k^2)` and :math:`p\sim f/k`, so at unit velocity one finds
    :math:`\|f\|\sim 10^2` and :math:`\|p\|\sim 10^1`. Handing an FNO an input of magnitude
    :math:`10^2` and asking for two output channels three orders of magnitude apart does not
    train. The dataset therefore also publishes two **network-facing** constants,

    * ``f_scale`` — RMS of the body force; the trainer feeds ``f / f_scale`` to the FNO,
    * ``p_scale`` — RMS of the pressure; the trainer reads pressure as ``channel * p_scale``,

    which leave the FEM operator, the residual and the reference untouched — they only put
    the network's input and both output channels at :math:`O(1)`.
    """

    def __init__(
        self,
        num_samples: int = 512,
        K: int = 4,
        seed: int = 42,
        grid_resolution: int = 65,
        mu: float = 1.0,
        r: float = -0.5,
        ref_chunk: int = 64,
        normalize: bool = True,
        mesh=None,
        problem: Optional[StokesProblem] = None,
        all_data: Optional[dict] = None,
        indices: Optional[List[int]] = None,
    ):
        super().__init__()
        if grid_resolution % 2 == 0:
            raise ValueError(
                f"Stokes needs an odd velocity grid (Q2 nodes = 2*n_p - 1), got "
                f"{grid_resolution}. Try {grid_resolution + 1}.")
        self.K = K
        self.mu = mu
        self.r = r
        self.grid_resolution = grid_resolution
        self.grid_size = (grid_resolution, grid_resolution)          # velocity (nx, ny)
        nx_p = (grid_resolution + 1) // 2
        self.pgrid_size = (nx_p, nx_p)

        if all_data is not None:
            self.mesh = mesh
            self.problem = problem
            sel = indices if indices is not None else range(len(all_data["u"]))
            self.f = all_data["f"][list(sel)]
            self.u = all_data["u"][list(sel)]
            self.p = all_data["p"][list(sel)]
            self._all_data = all_data                 # p_scale stays the shared pool's
        else:
            torch.manual_seed(seed)
            self.mesh = structured_quad9_mesh(nx=nx_p, ny=nx_p)
            self.problem = StokesProblem(self.mesh, nx_p=nx_p, ny_p=nx_p, mu=mu)

            coeffs = torch.rand(num_samples, 2, K, K) * 2 - 1
            f = stokes_body_force(coeffs, self.mesh.points, r=r).float()   # [num, n_u, 2]

            # Discrete Taylor-Hood reference (batched sparse solve, float64 internally).
            solve_dev = "cuda" if torch.cuda.is_available() else "cpu"
            prob = self.problem.to(solve_dev)
            u, p = prob.fem_reference(f.to(solve_dev), chunk=ref_chunk)
            self.problem = prob.to("cpu")
            f, u, p = f.cpu(), u.cpu(), p.cpu()

            if normalize:
                scale = self.problem.velocity_l2(u).clamp_min(1e-30)       # [num]
                f = f / scale[:, None, None]
                u = u / scale[:, None, None]
                p = p / scale[:, None]

            self.f, self.u, self.p = f, u, p
            self._all_data = {
                "f": f, "u": u, "p": p, "coeffs": coeffs,
                "f_scale": _rms(f), "p_scale": _rms(p),
                # RMS FE-L2 norms: the weights that balance the two fields in any
                # absolute-error loss or metric.
                "u_l2_scale": float(self.problem.velocity_l2(u).pow(2).mean().sqrt()),
                "p_l2_scale": float(self.problem.pressure_l2(p).pow(2).mean().sqrt()),
            }

        # Network-facing scales (see the class docstring): fixed constants that leave the
        # physics untouched and only make the FNO's input and output channels O(1).
        self.f_scale = float(self._all_data["f_scale"])
        self.p_scale = float(self._all_data["p_scale"])
        self.u_l2_scale = float(self._all_data["u_l2_scale"])
        self.p_l2_scale = float(self._all_data["p_l2_scale"])

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        nx, ny = self.grid_size
        f_node = self.f[idx]                                   # [n_u, 2]
        f_grid = torch.stack([node_to_grid(f_node[..., 0], nx, ny),
                              node_to_grid(f_node[..., 1], nx, ny)], dim=0)   # [2, ny, nx]
        return f_grid, f_node, self.u[idx], self.p[idx]

    def get_shared_resources(self):
        return self.mesh, self.problem, self._all_data


def create_stokes_datasets(n_train: int, n_val: int, n_test: int,
                           K: int, grid_resolution: int = 65, mu: float = 1.0,
                           r: float = -0.5, ref_chunk: int = 64,
                           normalize: bool = True, seed: int = 42):
    """Build train/val/test Stokes splits sharing one mesh, one :class:`StokesProblem`,
    and a single pool of reference solutions (one batched factorization for all splits)."""
    total = n_train + (n_val if n_val > 0 else 0) + (n_test if n_test > 0 else 0)
    base = StokesDataset(num_samples=total, K=K, seed=seed,
                         grid_resolution=grid_resolution, mu=mu, r=r,
                         ref_chunk=ref_chunk, normalize=normalize)
    mesh, problem, all_data = base.get_shared_resources()

    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_train + n_val)) if n_val > 0 else train_idx
    if n_test > 0:
        start = n_train + (n_val if n_val > 0 else 0)
        test_idx = list(range(start, start + n_test))
    else:
        test_idx = train_idx

    def _make(indices):
        return StokesDataset(
            num_samples=total, K=K, seed=seed, grid_resolution=grid_resolution,
            mu=mu, r=r, mesh=mesh, problem=problem, all_data=all_data, indices=indices,
        )
    return _make(train_idx), _make(val_idx), _make(test_idx)


class UnstructuredStokesDataset(Dataset):
    r"""Taylor-Hood Stokes on an **unstructured** mesh — the same problem, no image.

    Item: ``(f_node [n_u, 2], u_node [n_u, 2], p_node [n_p])``. The grid form
    :class:`StokesDataset` also carries is simply absent, which is the point: this is the
    dataset an FNO cannot consume and :class:`~tensorpils.gaot.GAOTModel` can.

    Everything that made the structured dataset work is reused verbatim, because none of it
    looked at the grid:

    * the source, :func:`stokes_body_force` — a pointwise sine series, defined wherever the
      nodes are, needing only that they lie in :math:`[0,1]^2`;
    * the reference, :meth:`~tensorpils.physics.StokesProblem.fem_reference` — a batched sparse
      float64 solve of the assembled ``K``, which zeroes exactly the training residual
      (measured 4.4e-06 relative on this mesh, against 6.7e-06 structured);
    * the three scalings — per-sample normalisation to unit velocity FE-``L²``, plus the
      network-facing ``f_scale`` / ``p_scale`` constants. Stokes is linear, so one constant per
      sample is exact, and the physics still fixes the other two magnitudes.

    There was never a manufactured solution here, so unlike the Poisson disc nothing has to be
    given up: the structured Stokes labels were already the discrete solve.

    Parameters
    ----------
    num_samples, K, seed, mu, r, ref_chunk, normalize : as :class:`StokesDataset`.
    chara_length : float
        Gmsh target edge length. ``0.035`` gives ~4000 P2 / ~1035 P1 nodes on the default
        obstacle geometry, against the structured benchmark's ``65² = 4225`` / ``33² = 1089``.
    obstacle, cx, cy, radius : geometry, see :func:`~tensorpils.meshing.obstacle_mesh`.
    """

    def __init__(
        self,
        num_samples: int = 512,
        K: int = 10,
        seed: int = 42,
        chara_length: float = 0.035,
        mu: float = 1.0,
        r: float = -0.5,
        ref_chunk: int = 64,
        normalize: bool = True,
        obstacle: str = "circle",
        cx: float = 0.40,
        cy: float = 0.50,
        radius: float = 0.14,
        mesh=None,
        problem: Optional[UnstructuredStokesProblem] = None,
        all_data: Optional[dict] = None,
        indices: Optional[List[int]] = None,
        cache_path: Optional[str] = None,
    ):
        super().__init__()
        self.K = K
        self.mu = mu
        self.r = r
        self.chara_length = chara_length
        # No grid, and nothing downstream may pretend otherwise.
        self.grid_size = None
        self.pgrid_size = None

        if all_data is not None:
            self.mesh = mesh
            self.problem = problem
            sel = list(indices) if indices is not None else list(range(len(all_data["u"])))
            self.f = all_data["f"][sel]
            self.u = all_data["u"][sel]
            self.p = all_data["p"][sel]
            self._all_data = all_data
        else:
            torch.manual_seed(seed)
            self.mesh = mesh if mesh is not None else obstacle_mesh(
                chara_length=chara_length, order=2, obstacle=obstacle,
                cx=cx, cy=cy, r=radius, cache_path=cache_path)
            self.problem = problem if problem is not None else \
                UnstructuredStokesProblem(self.mesh, mu=mu)

            coeffs = torch.rand(num_samples, 2, K, K) * 2 - 1
            f = stokes_body_force(coeffs, self.mesh.points, r=r).float()   # [num, n_u, 2]

            solve_dev = "cuda" if torch.cuda.is_available() else "cpu"
            prob = self.problem.to(solve_dev)
            u, p = prob.fem_reference(f.to(solve_dev), chunk=ref_chunk)
            self.problem = prob.to("cpu")
            f, u, p = f.cpu(), u.cpu(), p.cpu()

            if normalize:
                scale = self.problem.velocity_l2(u).clamp_min(1e-30)
                f = f / scale[:, None, None]
                u = u / scale[:, None, None]
                p = p / scale[:, None]

            self.f, self.u, self.p = f, u, p
            self._all_data = {
                "f": f, "u": u, "p": p, "coeffs": coeffs,
                "f_scale": _rms(f), "p_scale": _rms(p),
                "u_l2_scale": float(self.problem.velocity_l2(u).pow(2).mean().sqrt()),
                "p_l2_scale": float(self.problem.pressure_l2(p).pow(2).mean().sqrt()),
            }

        self.f_scale = float(self._all_data["f_scale"])
        self.p_scale = float(self._all_data["p_scale"])
        self.u_l2_scale = float(self._all_data["u_l2_scale"])
        self.p_l2_scale = float(self._all_data["p_l2_scale"])
        self.n_u = self.problem.n_u
        self.n_p = self.problem.n_p

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        return self.f[idx], self.u[idx], self.p[idx]

    def get_shared_resources(self):
        return self.mesh, self.problem, self._all_data


def create_unstructured_stokes_datasets(n_train: int, n_val: int, n_test: int,
                                        K: int = 10, chara_length: float = 0.035,
                                        mu: float = 1.0, r: float = -0.5,
                                        ref_chunk: int = 64, normalize: bool = True,
                                        obstacle: str = "circle", cx: float = 0.40,
                                        cy: float = 0.50, radius: float = 0.14,
                                        seed: int = 42, cache_path: Optional[str] = None):
    """Train/val/test over one mesh, one problem and one pool (one factorisation for all)."""
    total = n_train + (n_val if n_val > 0 else 0) + (n_test if n_test > 0 else 0)
    base = UnstructuredStokesDataset(
        num_samples=total, K=K, seed=seed, chara_length=chara_length, mu=mu, r=r,
        ref_chunk=ref_chunk, normalize=normalize, obstacle=obstacle, cx=cx, cy=cy,
        radius=radius, cache_path=cache_path)
    mesh, problem, all_data = base.get_shared_resources()

    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_train + n_val)) if n_val > 0 else train_idx
    if n_test > 0:
        start = n_train + (n_val if n_val > 0 else 0)
        test_idx = list(range(start, start + n_test))
    else:
        test_idx = train_idx

    def _make(indices):
        return UnstructuredStokesDataset(
            num_samples=total, K=K, seed=seed, chara_length=chara_length, mu=mu, r=r,
            mesh=mesh, problem=problem, all_data=all_data, indices=indices)
    return _make(train_idx), _make(val_idx), _make(test_idx)
