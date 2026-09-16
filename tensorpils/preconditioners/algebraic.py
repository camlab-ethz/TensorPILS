r"""Algebraic multigrid preconditioner :math:`P \approx A^{-1}` --- built from the MATRIX.

This is the mesh-free counterpart of :class:`~tensorpils.preconditioners.multigrid.
GeometricMultigrid`. The geometric V-cycle needs a structured grid twice over: it
re-discretizes each level with :func:`~tensorpils.meshing.structured_quad_mesh`, and its
prolongation :func:`~tensorpils.preconditioners.multigrid._build_2d_prolongation` is bilinear
interpolation that assumes the row-major node order. Neither survives an unstructured mesh.

An AMG V-cycle builds its own hierarchy from the sparsity and values of ``A`` alone, so the
only thing this class needs is the assembled operator and the Dirichlet mask --- no mesh, no
node ordering, no grid size. That is the whole point: **the same object preconditions a
structured and an unstructured discretisation**, and swapping it in on a structured grid is
the controlled experiment that isolates "AMG instead of GMG" from "unstructured instead of
structured".

Backend
-------
One AmgX V-cycle through ``torch_amgx`` (NVIDIA AmgX 2.5). Four facts about that binding
shape this code, all measured rather than assumed (see ``notes/`` and the experiment README):

* **We own the autograd.** ``torch_amgx`` is a binding layer with no autograd; its own module
  docstring says to wrap the solve in a :class:`torch.autograd.Function`. We do, in
  :class:`_AMGXApply`.
* **The vjp is another forward apply, but only if ``presweeps == postsweeps``.** A V-cycle with
  a symmetric smoother and matching sweep counts is a symmetric operator, so
  :math:`P^\top = P` and the backward pass is the same cycle. Measured
  :math:`\|P-P^\top\|_\infty/\|P\|_\infty`: ``5.7e-16`` (fp64) / ``2.5e-7`` (fp32) at matching
  sweeps, against ``5.4e-2`` at 2/1 and ``1.4e-1`` at 1/0. The constructor therefore *refuses*
  mismatched sweeps rather than silently returning a wrong gradient --- there is no transpose
  apply in the binding to fall back on.
* **``tolerance`` must be 0.** With ``max_iters=1`` AmgX still runs its convergence check, and
  the default ``tolerance=1e-8`` is ABSOLUTE. A residual that falls below it --- which is
  exactly what a *successfully training* model produces --- makes AmgX return ``x = 0`` at
  iteration 0, so ``Pr = 0``, the loss is 0, the gradient vanishes and training silently
  freezes. On *this* path the failure is a clean cliff --- measured at ``65^2``, the relative
  linearity error sits at fp32 roundoff (~1e-7) for every ``\|r\|`` down to the tolerance and then
  goes to exactly zero below it --- because ``max_iters=1`` leaves the check nothing to do but
  skip the single cycle outright. (An iterate-to-tolerance solver instead *partially* converges
  and degrades gradually for a decade above its tolerance; that is the same trap with a softer
  edge, and it is what makes ``torch_sla.spsolve``'s ``atol`` default dangerous for adjoints.)
  ``tolerance=0`` removes it at no cost, and ``tests/test_amg_precond.py`` guards it by checking
  linearity ``P(cr) = cP(r)`` across twelve decades of ``c`` --- a test that covers both edges,
  where asserting merely that the output is nonzero would not.
* **``monitor_residual`` and ``store_res_history`` must both be 1**, however useless the history
  is here. The binding calls ``AMGX_solver_get_iteration_residual`` unconditionally after every
  solve and throws without it, printing a ~3.5 KB stack trace *per apply* (measured harmless to
  the results, but ~3.5 MB of log per 1000 applies, and it distorts timings). Turning only the
  history off is not an option either: AmgX rejects ``store_res_history=1, monitor_residual=0``
  outright at solver creation, so the pair moves together.

Two operational constraints come with AmgX: it is **CUDA-only**, and **two live solver objects
abort the process at exit** (an un-refcounted ``AMGX_initialize``/``finalize`` in the binding;
the run's numbers are correct but the exit code is 134 and Slurm marks the job FAILED). One
fixed ``A`` needs exactly one solver, so ordinary use is in the safe case --- but do not build
a second one for :math:`A^\top` (use the symmetry above), and do not mix with
``torch_sla.solve(backend="amgx")`` in the same process.

Cost
----
AmgX takes one right-hand side at a time, so a batch is a Python loop. Measured on an RTX 4090
the per-apply cost is dominated by host-side launch overhead and is nearly flat in ``N``
(6.2 ms per batch of 32 at ``N=289`` against 7.0 ms at ``N=16641``), against 4.2 ms for the
batched geometric V-cycle at ``65^2``. The geometric cycle is cheaper on a small structured
grid; the algebraic one stops caring about ``N``, which is the regime the unstructured case
lives in.
"""

from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch

from .base import Preconditioner

__all__ = ["AMGXPreconditioner", "bc_eliminated_matrix"]


def bc_eliminated_matrix(A, boundary_mask) -> sp.csr_matrix:
    r"""``A`` with Dirichlet rows/columns zeroed and a unit diagonal there, as SciPy CSR.

    This is the same level-0 operator :meth:`GeometricMultigrid._assemble_operator` builds,
    but read off the *assembled* ``problem.A`` instead of re-discretizing a structured mesh ---
    which is what makes it work for any mesh. On the boundary block the operator is the
    identity, so a residual carrying the repo's zero-boundary convention maps to a correction
    that is still zero there, and the :class:`~tensorpils.preconditioners.base.Preconditioner`
    full-grid contract is preserved exactly.

    ``A`` may be a TensorMesh ``SparseMatrix``, a torch (sparse or dense) tensor, or a SciPy
    matrix. ``boundary_mask`` is a boolean over the full node set.
    """
    if hasattr(A, "to_scipy_coo"):                       # tensormesh.SparseMatrix
        A = A.to_scipy_coo()
    elif isinstance(A, torch.Tensor):
        A = sp.coo_matrix(A.to_dense().detach().cpu().numpy() if A.is_sparse
                          else A.detach().cpu().numpy())
    A = sp.csr_matrix(A, dtype=np.float64)

    mask = boundary_mask
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask, dtype=bool)
    keep = (~mask).astype(np.float64)

    out = (sp.diags(keep) @ A @ sp.diags(keep) + sp.diags(mask.astype(np.float64))).tocsr()
    out.sort_indices()
    return out


class _AMGXApply(torch.autograd.Function):
    r"""One AmgX V-cycle as a differentiable linear map, batched by looping.

    ``P`` is a *constant* linear operator (the matrix is fixed for a run), so there is no
    gradient with respect to it and the vjp is :math:`P^\top g = P g` --- valid because the
    cycle is symmetric, which :class:`AMGXPreconditioner` enforces at construction.
    """

    @staticmethod
    def forward(ctx, r, solver, dtype):
        ctx.solver, ctx.dtype = solver, dtype
        return _AMGXApply._apply_rows(r.detach(), solver, dtype)

    @staticmethod
    def backward(ctx, g):
        return _AMGXApply._apply_rows(g.detach(), ctx.solver, ctx.dtype), None, None

    @staticmethod
    def _apply_rows(x, solver, dtype):
        out_dtype = x.dtype
        flat = x.reshape(-1, x.shape[-1]).to(dtype)
        rows = [solver.solve(flat[i].contiguous()) for i in range(flat.shape[0])]
        return torch.stack(rows).to(out_dtype).reshape(x.shape)


class AMGXPreconditioner(Preconditioner):
    r"""One AmgX V-cycle as :math:`P \approx A^{-1}`, built from the matrix alone.

    Parameters
    ----------
    A : matrix
        The assembled operator to invert approximately --- ``problem.A`` for Poisson, or
        ``a^2 A + c M`` for the Allen--Cahn screened operator. Dirichlet rows/columns are
        eliminated internally by :func:`bc_eliminated_matrix`.
    boundary_mask : torch.Tensor
        Boolean over the full node set; ``True`` on Dirichlet nodes.
    sweeps : int
        Pre- **and** post-smoothing sweeps. A single number, not two, because the vjp is only
        correct when they match (see the module docstring).
    algorithm : {"CLASSICAL", "AGGREGATION"}
        AmgX coarsening. ``CLASSICAL`` (Ruge--Stueben) is the default and the right analogue of
        the geometric hierarchy for an elliptic operator.
    smoother : str
        AmgX smoother token. Must be symmetric for the vjp to be valid; ``BLOCK_JACOBI``
        (the default) is.
    relaxation : float
        Smoother damping.
    device : str
        CUDA device. AmgX has no CPU path.
    dtype : torch.dtype
        Arithmetic of the cycle. ``float32`` matches training and is ~20 % faster per apply;
        ``float64`` is available for conditioning measurements.
    """

    def __init__(self, A, boundary_mask, sweeps: int = 2,
                 algorithm: str = "CLASSICAL", smoother: str = "BLOCK_JACOBI",
                 relaxation: float = 0.8, device: str = "cuda:0",
                 dtype: torch.dtype = torch.float32,
                 max_levels: int = 24, min_coarse_rows: int = 16,
                 coarse_solver: str = "DENSE_LU_SOLVER"):
        super().__init__()
        try:
            import torch_amgx
        except ImportError as e:                                    # pragma: no cover
            raise ImportError(
                "AMGXPreconditioner needs torch-amgx (NVIDIA AmgX), which is CUDA-only."
            ) from e
        if not torch_amgx.is_available():                           # pragma: no cover
            # `import torch_amgx` is NOT where a broken install shows up: the native extension is
            # loaded lazily and the failure is swallowed into is_available() -> False, so the
            # package imports cleanly and the first real error arrives later as a misleading
            # "install a prebuilt wheel". Surface the actual cause here instead.
            if not torch.cuda.is_available():
                raise RuntimeError("AmgX is CUDA-only and no CUDA device is visible.")
            try:
                from torch_amgx import _C                           # noqa: F401
            except ImportError as e:
                raise ImportError(
                    f"torch_amgx imported but its native extension could not load: {e}\n"
                    "On Euler this is a module-loaded libstdc++ shadowing the system one "
                    "(gcc/8.5.0 provides GLIBCXX <= 3.4.25; libamgxsh.so needs 3.4.29). Prepend "
                    "/usr/lib/x86_64-linux-gnu to LD_LIBRARY_PATH -- see "
                    "experiments/poisson/amg_dropin/README.md."
                ) from e
            raise RuntimeError("torch_amgx.is_available() is False for an unknown reason.")

        if int(sweeps) < 1:
            raise ValueError(f"sweeps must be >= 1, got {sweeps}")
        dev = torch.device(device)
        if dev.type != "cuda":
            raise RuntimeError(f"AmgX is CUDA-only; got device {device!r}")
        if dev.index is None:                  # the binding compares device objects exactly
            dev = torch.device("cuda", torch.cuda.current_device())

        self.sweeps = int(sweeps)
        self.algorithm = algorithm
        self.smoother = smoother
        self.relaxation = float(relaxation)
        self._dtype = dtype
        self._device = dev

        Acsr = bc_eliminated_matrix(A, boundary_mask)
        self.n_dofs = int(Acsr.shape[0])
        self.nnz = int(Acsr.nnz)

        cfg = torch_amgx.Config(amgx_config_str=(
            "config_version=2,"
            "solver(main)=AMG,"
            f"main:algorithm={algorithm},"
            "main:max_iters=1,"                     # one V-cycle, not a solve
            "main:cycle=V,"
            f"main:presweeps={self.sweeps},main:postsweeps={self.sweeps},"
            f"main:smoother={smoother},"
            f"main:relaxation_factor={self.relaxation},"
            # tolerance=0: never let the convergence check short-circuit the cycle and
            # return x=0 -- a small residual is success, not a reason to stop.
            "main:tolerance=0.0,main:convergence=ABSOLUTE,main:norm=L2,"
            # both must stay on: the binding always reads the iteration residual back.
            "main:monitor_residual=1,main:store_res_history=1,main:print_solve_stats=0,"
            f"main:max_levels={max_levels},main:min_coarse_rows={min_coarse_rows},"
            f"main:coarse_solver={coarse_solver}"
        ))
        # Exactly one live Solver per process: two abort at interpreter exit.
        self._solver = torch_amgx.Solver(cfg, device=dev)
        self._solver.setup_csr(
            torch.from_numpy(Acsr.indptr.astype(np.int32)).to(dev),
            torch.from_numpy(Acsr.indices.astype(np.int32)).to(dev),
            torch.from_numpy(Acsr.data).to(dtype).to(dev),
            self.n_dofs)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Apply ``P``. ``r``: ``[B, N_full]`` or ``[N_full]``; same shape out."""
        if r.shape[-1] != self.n_dofs:
            raise ValueError(f"residual has {r.shape[-1]} entries, expected {self.n_dofs}")
        if r.device != self._device:
            r = r.to(self._device)
        return _AMGXApply.apply(r, self._solver, self._dtype)

    def report(self) -> None:
        print(f"[amgx precond] one V({self.sweeps},{self.sweeps}) cycle  "
              f"algorithm={self.algorithm}  smoother={self.smoother}  "
              f"relax={self.relaxation:g}  n={self.n_dofs}  nnz={self.nnz}  "
              f"dtype={str(self._dtype).replace('torch.', '')}  device={self._device}")

    @torch.no_grad()
    def symmetry_defect(self, k: int = 8, seed: int = 0) -> float:
        r"""Measured :math:`\max_i |\langle Px_i,y_i\rangle-\langle x_i,Py_i\rangle| /
        |\langle Px_i,y_i\rangle|` --- the assumption the backward pass rests on.

        Expect ~1e-7 in float32 and ~1e-15 in float64. Anything larger means the cycle is not
        symmetric and the gradient of any loss using this ``P`` is wrong.
        """
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(k, self.n_dofs, generator=g).to(self._device)
        y = torch.randn(k, self.n_dofs, generator=g).to(self._device)
        a = (self(x) * y).sum(-1)
        b = (x * self(y)).sum(-1)
        return float(((a - b).abs() / a.abs().clamp_min(1e-30)).max())
