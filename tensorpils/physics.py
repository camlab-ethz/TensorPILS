"""FEM operators built on TensorMesh's assembled matrices.

Two PDE operators live here, both assembled once from a structured
:class:`tensormesh.Mesh` and both exposing a batched, autograd-differentiable
interface used by the physics-informed losses:

* :class:`PoissonProblem` — static :math:`-\\Delta u = f`
* :class:`WaveProblem`    — time-dependent :math:`u_{tt} = c^2 \\Delta u`
* :class:`ACProblem`      — time-dependent Allen–Cahn :math:`u_t = a^2 \\Delta u + \\epsilon^2 u(1-u^2)`
* :class:`StokesProblem`  — static Stokes saddle point :math:`-\\mu\\Delta u + \\nabla p = f`,
  :math:`\\nabla\\cdot u = 0`, on a Taylor-Hood Q2/Q1 pair

They share :class:`FEMOperator`, which assembles the stiffness ``A`` and mass ``M``
matrices (via TensorMesh's ``LaplaceElementAssembler`` / ``MassElementAssembler``),
holds the Dirichlet ``boundary_mask``, and confines the ``[n_nodes, batch]`` transpose
required by sparse mat–vec products. All public methods take/return node fields in
``[batch, n_nodes]`` (or ``[n_nodes]``) convention.
"""

import torch
import torch.nn as nn

from tensormesh import (Condenser, Field, LaplaceElementAssembler,
                        MassElementAssembler, MixedElementAssembler)

from .meshing import structured_quad_mesh, node_to_grid, grid_to_node

__all__ = ["FEMOperator", "PoissonProblem", "WaveProblem", "ACProblem", "StokesProblem",
           "apply_zero_boundary"]


def apply_zero_boundary(u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out boundary entries, keeping interior values. Supports ``[N]`` and ``[B, N]``."""
    out = torch.zeros_like(u)
    if u.dim() == 1:
        out[~mask] = u[~mask]
    else:
        out[:, ~mask] = u[:, ~mask]
    return out


class FEMOperator(nn.Module):
    r"""Shared FEM machinery: assembled stiffness ``A`` and mass ``M`` + Dirichlet mask.

    Parameters
    ----------
    mesh : tensormesh.Mesh
        The (structured) quad mesh. ``mesh.boundary_mask`` defines the Dirichlet nodes.
    quadrature_order : int, optional
        Gauss order per axis for assembling ``A`` and ``M``. Default ``4`` integrates
        the bilinear-quad stiffness and mass exactly on rectangular cells.

    Notes
    -----
    ``A`` and ``M`` are plain attributes (TensorMesh ``SparseMatrix``, not registered
    buffers), so :meth:`to` is overridden to move them alongside the module.
    """

    def __init__(self, mesh, quadrature_order: int = 4):
        super().__init__()
        self.n_nodes = int(mesh.n_points)
        self.register_buffer("boundary_mask", mesh.boundary_mask.bool())
        # Assemble once; matrices inherit the mesh dtype/device and are cast at use.
        self.A = LaplaceElementAssembler.from_mesh(mesh, quadrature_order=quadrature_order)(mesh.points)
        self.M = MassElementAssembler.from_mesh(mesh, quadrature_order=quadrature_order)(mesh.points)

    # -- device/dtype management: move the (plain-attribute) sparse matrices too --
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.A = self.A.to(*args, **kwargs)
        self.M = self.M.to(*args, **kwargs)
        return self

    @staticmethod
    def _spmm(mat, x: torch.Tensor) -> torch.Tensor:
        """Differentiable sparse mat–vec / mat–mat. ``x``: ``[N]`` or ``[B, N]``."""
        if mat.dtype != x.dtype or mat.device != x.device:
            mat = mat.to(dtype=x.dtype, device=x.device)
        if x.dim() == 1:
            return mat @ x                 # [N]
        return (mat @ x.T).T               # [B, N]

    def load_vector(self, f: torch.Tensor) -> torch.Tensor:
        r"""Consistent load :math:`b = M f`. ``f``: ``[N]`` or ``[B, N]``; returns same shape."""
        return self._spmm(self.M, f)


class PoissonProblem(FEMOperator):
    r"""Discrete FEM operator for :math:`-\Delta u = f` with zero Dirichlet BC.

    Notes
    -----
    The energy uses the matrix identity
    :math:`E(u) = \tfrac12 u^\top A u - u^\top (M f)`, which is exactly the FE energy
    :math:`\int(\tfrac12|\nabla u_h|^2 - f_h u_h)` and whose gradient w.r.t. ``u`` is the
    residual :math:`A u - M f` — the property the Deep Ritz / preconditioned losses rely on.
    """

    # ------------------------------------------------------------------ residual
    def residual(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        r"""Boundary-masked Galerkin residual :math:`R = A\,u_{\mathrm{bc}} - M f`.

        ``u`` is first projected to zero on the Dirichlet boundary; the returned
        residual is also zeroed there. Fully differentiable in ``u``.
        """
        u_bc = apply_zero_boundary(u, self.boundary_mask)
        r = self._spmm(self.A, u_bc) - self.load_vector(f)
        return apply_zero_boundary(r, self.boundary_mask)

    # -------------------------------------------------------------------- energy
    def energy(self, u: torch.Tensor, f: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
        r"""Deep Ritz energy :math:`E(u) = \tfrac12 u^\top A u - u^\top (M f)`.

        ``u``, ``f``: ``[N]`` or ``[B, N]``. Returns a scalar (``reduce="mean"``/``"sum"``)
        or the per-sample energy ``[B]`` (``reduce="none"``). Apply any boundary
        projection to ``u`` *before* calling for hard-BC behaviour.
        """
        Au = self._spmm(self.A, u)
        b = self.load_vector(f)
        if u.dim() == 1:
            return 0.5 * (u * Au).sum() - (u * b).sum()
        e = 0.5 * (u * Au).sum(dim=-1) - (u * b).sum(dim=-1)   # [B]
        if reduce == "mean":
            return e.mean()
        if reduce == "sum":
            return e.sum()
        return e


class WaveProblem(FEMOperator):
    r"""Discrete FEM operator for the wave equation :math:`u_{tt} = c^2 \Delta u`
    with zero Dirichlet BC, in central-difference (explicit) time.

    The semi-discrete form is :math:`M\ddot u + c^2 A u = 0`; replacing
    :math:`\ddot u` by the second-order central difference gives the nodal residual

    .. math::
        R = M\,\frac{u^{n+1} - 2u^n + u^{n-1}}{\Delta t^2} + c^2 A\,u^n,

    which the label-free Galerkin loss drives to zero. This mirrors the update
    :math:`M u^{n+1} = 2M u^n - M u^{n-1} - \Delta t^2 c^2 A u^n` of TensorMesh's
    ``examples/wave/wave.py``. ``A`` (stiffness) and ``M`` (mass) are the same
    matrices :class:`PoissonProblem` assembles.
    """

    # ------------------------------------------------------------------ residual
    def residual(self, u_prev: torch.Tensor, u_curr: torch.Tensor,
                 u_next: torch.Tensor, c: float, dt: float) -> torch.Tensor:
        r"""Boundary-masked central-difference wave residual.

        ``u_prev``/``u_curr``/``u_next`` are :math:`u^{n-1}, u^n, u^{n+1}` in ``[N]`` or
        ``[B, N]``. Each is projected to zero on the Dirichlet boundary before use and
        the returned residual is also zeroed there. Fully differentiable in the inputs.
        """
        mask = self.boundary_mask
        up = apply_zero_boundary(u_prev, mask)
        uc = apply_zero_boundary(u_curr, mask)
        un = apply_zero_boundary(u_next, mask)
        accel = (un - 2.0 * uc + up) / (dt * dt)
        r = self._spmm(self.M, accel) + (c * c) * self._spmm(self.A, uc)
        return apply_zero_boundary(r, mask)

    # -------------------------------------------------------------------- energy
    def energy(self, u: torch.Tensor, v: torch.Tensor, c: float,
               reduce: str = "mean") -> torch.Tensor:
        r"""Mechanical energy :math:`E = \tfrac12 v^\top M v + \tfrac12 c^2 u^\top A u`
        (kinetic + potential). ``u``, ``v``: ``[N]`` or ``[B, N]``. Diagnostic only."""
        Mv = self._spmm(self.M, v)
        Au = self._spmm(self.A, u)
        if u.dim() == 1:
            return 0.5 * (v * Mv).sum() + 0.5 * (c * c) * (u * Au).sum()
        e = 0.5 * (v * Mv).sum(dim=-1) + 0.5 * (c * c) * (u * Au).sum(dim=-1)   # [B]
        if reduce == "mean":
            return e.mean()
        if reduce == "sum":
            return e.sum()
        return e


class ACProblem(FEMOperator):
    r"""Discrete FEM operator for the Allen–Cahn equation

    .. math::
        \partial_t u = a^2 \Delta u + \epsilon^2 u(1-u^2)

    on the unit square with zero Dirichlet BC, in fully-implicit backward Euler time.
    ``a`` is the diffusion coefficient and ``\epsilon`` the reaction strength (following
    TensorGalerkin's ``Trainer/ac.py`` and TensorMesh's ``examples/diffusion/allen-cahn``:
    the large factor multiplies the double-well term, not the Laplacian).

    Backward Euler with the reaction evaluated at the new level gives the weak-form nodal
    residual (group / product FEM, consistent mass ``M`` and stiffness ``A``):

    .. math::
        R = M\,\frac{u^{n+1} - u^n}{\Delta t} + a^2 A\,u^{n+1}
            - \epsilon^2 M\,\big(u^{n+1} - (u^{n+1})^3\big),

    which the label-free Galerkin loss drives to zero. The FEM reference trajectory
    (:meth:`fem_reference`) solves ``R = 0`` per step by Newton, so it exactly zeroes this
    residual — the discrete target the physics loss is trained toward. ``A``/``M`` are the
    same matrices :class:`PoissonProblem` / :class:`WaveProblem` assemble.
    """

    # ------------------------------------------------------------------ residual
    def residual(self, u_curr: torch.Tensor, u_next: torch.Tensor,
                 a: float, eps: float, dt: float,
                 integrator: str = "backward_euler") -> torch.Tensor:
        r"""Boundary-masked weak-form Allen–Cahn step residual (``/dt`` scaling), by integrator.

        ``u_curr``/``u_next`` are :math:`u^n, u^{n+1}` in ``[N]`` or ``[B, N]``. Both are projected
        to zero on the Dirichlet boundary before use and the returned residual is also zeroed
        there. Fully differentiable in the inputs. The two schemes differ only in the *linear*
        reaction term (the cubic is implicit in both):

        * ``"backward_euler"`` — fully implicit:
          :math:`M\frac{u^{n+1}-u^n}{\dt} + a^2Au^{n+1} - \epsilon^2 M(u^{n+1}-(u^{n+1})^3)`.
        * ``"convex_concave"`` — Eyre split, linear term explicit at :math:`u^n`:
          :math:`M\frac{u^{n+1}-u^n}{\dt} + a^2Au^{n+1} + \epsilon^2 M((u^{n+1})^3-u^n)`.

        Both are the ``/dt``-scaled residual (matching :meth:`fem_reference`), so a least-squares
        loss built on either has a ``dt``-independent gradient scale. This is the same residual
        the reference solve zeros, so the ``convex_concave`` reference exactly zeros the
        ``convex_concave`` residual (and likewise for backward Euler)."""
        mask = self.boundary_mask
        uc = apply_zero_boundary(u_curr, mask)
        un = apply_zero_boundary(u_next, mask)
        if integrator == "backward_euler":
            reaction = (eps * eps) * (un - un ** 3)               # linear term implicit (uⁿ⁺¹)
        elif integrator == "convex_concave":
            reaction = (eps * eps) * (uc - un ** 3)               # linear term explicit (uⁿ)
        else:
            raise ValueError(f"unknown integrator {integrator!r}; "
                             "expected 'backward_euler' or 'convex_concave'")
        r = self._spmm(self.M, (un - uc) / dt) \
            + (a * a) * self._spmm(self.A, un) \
            - self._spmm(self.M, reaction)
        return apply_zero_boundary(r, mask)

    # ---------------------------------------------- minimizing-movement objective
    def mm_objective(self, u_curr: torch.Tensor, u_next: torch.Tensor,
                     a: float, eps: float, dt: float) -> torch.Tensor:
        r"""Convex–concave minimizing-movement (JKO) objective for one Allen–Cahn step.

        With the Ginzburg–Landau energy split :math:`E = E_{\mathrm{cvx}} + E_{\mathrm{ccv}}`
        (:math:`E_{\mathrm{cvx}}=\tfrac{a^2}{2}\!\int|\nabla u|^2+\tfrac{\epsilon^2}{4}\!\int u^4
        + \tfrac{\epsilon^2}{4}|\Omega|`, :math:`E_{\mathrm{ccv}}=-\tfrac{\epsilon^2}{2}\!\int u^2`,
        so :math:`DE_{\mathrm{ccv}}(u^n)=-\epsilon^2 u^n`), the convex–concave step is the unique
        minimiser of

        .. math::
            J(u) = E_{\mathrm{cvx}}(u) + \langle DE_{\mathrm{ccv}}(u^n), u\rangle
                   + \tfrac{1}{2\dt}\|u-u^n\|_{L^2}^2 .

        Discretised (nodal quartic :math:`\int u^4 \approx (u^4)^\top M\mathbf 1`, lumped mass
        :math:`M\mathbf 1`), for ``u_next``:math:`=u`, ``u_curr``:math:`=u^n`:

        .. math::
            J = \tfrac{a^2}{2}u^\top A u + \tfrac{\epsilon^2}{4}\big[(u^4)^\top M\mathbf 1+|\Omega|\big]
                - \epsilon^2 (u^n)^\top M u + \tfrac{1}{2\dt}(u-u^n)^\top M (u-u^n).

        Its gradient is the convex–concave step residual with the *lumped* mass on the cubic
        (:math:`\nabla_u J = M\frac{u-u^n}{\dt}+a^2Au+\epsilon^2\operatorname{diag}(M\mathbf 1)u^3
        -\epsilon^2 M u^n`); minimising :math:`J` is the Deep-Ritz analogue of the least-squares
        residual loss. Returns ``[B]`` (or scalar for ``[N]``). ``|\Omega|`` is a `u`-independent
        constant, kept for fidelity to the objective. Boundary-projected like :meth:`residual`."""
        mask = self.boundary_mask
        uc = apply_zero_boundary(u_curr, mask)
        un = apply_zero_boundary(u_next, mask)
        ones = torch.ones(self.n_nodes, dtype=un.dtype, device=un.device)
        m1 = self._spmm(self.M, ones)                     # lumped mass vector M·1  [N]
        omega = m1.sum()                                  # |Ω| = 1ᵀM1
        Aun = self._spmm(self.A, un)
        Mun = self._spmm(self.M, un)
        Muc = self._spmm(self.M, uc)
        grad_term = 0.5 * (a * a) * (un * Aun).sum(dim=-1)              # a²/2 uᵀA u
        quartic = 0.25 * (eps * eps) * ((un ** 4) * m1).sum(dim=-1)     # ε²/4 (u⁴)ᵀM1
        concave = -(eps * eps) * (uc * Mun).sum(dim=-1)                 # -ε² (uⁿ)ᵀM u
        prox = (0.5 / dt) * ((un - uc) * (Mun - Muc)).sum(dim=-1)       # 1/(2dt)‖u-uⁿ‖²_M
        return grad_term + quartic + concave + prox + 0.25 * (eps * eps) * omega

    # -------------------------------------------------------------------- energy
    def energy(self, u: torch.Tensor, a: float, eps: float,
               reduce: str = "mean") -> torch.Tensor:
        r"""Ginzburg–Landau energy :math:`E = \tfrac12 a^2 u^\top A u + \epsilon^2 \tfrac14 u^\top M (u^2-1)^2`
        (gradient + double-well). ``u``: ``[N]`` or ``[B, N]``. Diagnostic only; Allen–Cahn is
        the ``L^2`` gradient flow of this energy, so it is non-increasing in time."""
        Au = self._spmm(self.A, u)
        well = 0.25 * (u * u - 1.0) ** 2                           # W(u)=¼(u²-1)²
        Mw = self._spmm(self.M, well)
        if u.dim() == 1:
            return 0.5 * (a * a) * (u * Au).sum() + (eps * eps) * Mw.sum()
        e = 0.5 * (a * a) * (u * Au).sum(dim=-1) + (eps * eps) * Mw.sum(dim=-1)   # [B]
        if reduce == "mean":
            return e.mean()
        if reduce == "sum":
            return e.sum()
        return e

    # ------------------------------------------ FEM reference solve (legacy dense, kept for now)
    @torch.no_grad()
    def fem_reference_old(self, u0: torch.Tensor, a: float, eps: float, dt: float,
                          n_steps: int, newton_tol: float = 1e-8, newton_max: int = 20,
                          chunk: int = 64, integrator: str = "convex_concave") -> torch.Tensor:
        r"""Legacy dense batched Newton reference (superseded by :meth:`fem_reference`; kept for
        comparison/benchmarking, to be removed). Dense ``torch.linalg.solve`` on the interior
        block, solves in the caller's dtype.

        Batched implicit + Newton reference trajectory (build-time ground truth).

        Solves ``R(u^{n+1}; u^n) = 0`` each step by Newton on the interior DOFs (Dirichlet nodes
        held at 0). Uses dense linear algebra on the interior block — appropriate for structured
        grids up to a few thousand nodes; process ``chunk`` samples at a time to bound memory.
        ``u0``: ``[N]`` or ``[B, N]``. Returns the trajectory ``[B, n_steps+1, N]`` (or
        ``[n_steps+1, N]`` for a single sample).

        ``integrator`` selects the time discretisation of the reaction ε²(u−u³):

        * ``"convex_concave"`` (default) — Eyre convex splitting: the convex quartic (cubic term
          ε²u³) is implicit, the concave part (linear term ε²u) is explicit at ``u^n``. Residual
          :math:`(M+\Delta t\,a^2A)u+\Delta t\,\varepsilon^2 M u^3-(I+\Delta t\,\varepsilon^2)Mu^n`,
          Jacobian :math:`J = M/\Delta t + a^2A + 3\varepsilon^2 M\,\mathrm{diag}(u^2)`, which is
          SPD for every ``u`` and ``Δt`` — unconditionally energy-stable.
        * ``"backward_euler"`` — fully implicit; the whole reaction is at the new level. Jacobian
          :math:`J = M/\Delta t + a^2A - \varepsilon^2 M\,\mathrm{diag}(1-3u^2)`.

        The constant part ``L = M/\Delta t + a^2 A`` is formed once; the reaction Jacobian is a
        per-iterate column scaling of ``M``."""
        if integrator not in ("convex_concave", "backward_euler"):
            raise ValueError(f"unknown integrator {integrator!r}; "
                             "expected 'convex_concave' or 'backward_euler'")
        single = (u0.dim() == 1)
        if single:
            u0 = u0.unsqueeze(0)
        device, dtype = u0.device, u0.dtype
        mask = self.boundary_mask.to(device)
        inner = ~mask
        idx = torch.nonzero(inner, as_tuple=False).squeeze(1)

        # Dense operators on the interior block (assembled once).
        M = self.M.to_dense().to(device=device, dtype=dtype)
        A = self.A.to_dense().to(device=device, dtype=dtype)
        M_ii = M[idx][:, idx]                                      # [Ni, Ni]
        A_ii = A[idx][:, idx]                                      # [Ni, Ni]
        L_ii = M_ii / dt + (a * a) * A_ii                          # constant Newton LHS part

        B = u0.shape[0]
        traj = torch.zeros(B, n_steps + 1, self.n_nodes, device=device, dtype=dtype)
        traj[:, 0] = apply_zero_boundary(u0, mask)

        for lo in range(0, B, chunk):
            hi = min(lo + chunk, B)
            u_old = traj[lo:hi, 0, idx].clone()                   # [b, Ni]
            for step in range(1, n_steps + 1):
                u = u_old.clone()                                 # Newton initial guess
                for _ in range(newton_max):
                    # Reaction term and its Jacobian column-scale w (so J = L_ii + M_ii·diag(w)),
                    # per integrator. Both share the diffusion+mass block L_ii = M/dt + a²A.
                    if integrator == "backward_euler":
                        reac = (eps * eps) * (u - u ** 3)         # ε²(u-u³) fully implicit
                        r_sign = -1.0                             # residual: ... - M·reac
                        w = (eps * eps) * (3.0 * u * u - 1.0)     # J = L - ε²M diag(1-3u²)
                    else:                                         # convex_concave (Eyre split)
                        reac = (eps * eps) * (u ** 3 - u_old)     # cubic implicit, linear explicit
                        r_sign = +1.0                             # residual: ... + M·reac
                        w = 3.0 * (eps * eps) * (u * u)           # J = L + 3ε²M diag(u²)  (SPD)
                    r = (M_ii @ ((u - u_old) / dt).T).T \
                        + (a * a) * (A_ii @ u.T).T \
                        + r_sign * (M_ii @ reac.T).T              # [b, Ni]
                    if r.norm(dim=1).max() < newton_tol:
                        break
                    J = L_ii.unsqueeze(0) + (M_ii.unsqueeze(0) * w.unsqueeze(1))   # column-scale
                    du = torch.linalg.solve(J, -r.unsqueeze(-1)).squeeze(-1)   # [b, Ni]
                    u = u + du
                traj[lo:hi, step, idx] = u
                u_old = u
        return traj[0] if single else traj

    # ------------------------------------------ FEM reference solve (sparse float64, standard)
    @torch.no_grad()
    def fem_reference(self, u0: torch.Tensor, a: float, eps: float, dt: float,
                      n_steps: int, newton_tol: float = 1e-8, newton_max: int = 20,
                      chunk: int = 64, integrator: str = "convex_concave",
                      solver_backend: str = "auto") -> torch.Tensor:
        r"""Sparse-solve Newton reference trajectory --- the standard build-time ground truth.

        Same contract and output as the legacy dense :meth:`fem_reference_old`, but each
        Newton system is solved with a **batched sparse** solve (``SparseMatrix.solve_batch``:
        shared sparsity, per-sample values) instead of a dense ``torch.linalg.solve``. This drops
        the :math:`O(N_i^2)` memory / :math:`O(N_i^3)` factorisation and sidesteps MAGMA's batched
        dense LU (which fails at large grids). ``solve_batch`` loops per sample internally, so it is
        genuinely sparse but not GPU-parallel; SPD is auto-detected for the convex--concave path.

        The interior Jacobian is ``J = M/dt + a^2 A + M diag(w)`` with ``w = 3 eps^2 u^2``
        (convex_concave) or ``w = eps^2 (3u^2 - 1)`` (backward_euler); its values are assembled per
        Newton step as ``L.values + M.values * w[:, col]`` on the shared ``(row,col)`` layout of
        ``L = M/dt + a^2 A``. ``solver_backend`` is passed through (``"auto"`` picks scipy on CPU,
        cupy/cuDSS on CUDA if installed)."""
        if integrator not in ("convex_concave", "backward_euler"):
            raise ValueError(f"unknown integrator {integrator!r}; "
                             "expected 'convex_concave' or 'backward_euler'")
        import numpy as np
        from tensormesh.sparse.matrix import SparseMatrix

        single = (u0.dim() == 1)
        if single:
            u0 = u0.unsqueeze(0)
        device, out_dtype = u0.device, u0.dtype
        # The reference is build-time ground truth, so always solve in float64 regardless of the
        # caller's dtype: fp32 cannot reach newton_tol=1e-8 / the solver's inner tolerance (its
        # residual floor is ~1e-6), which leaves Newton non-converged AND spinning to newton_max
        # every step. Cast the trajectory back to the caller's dtype only on return.
        dtype = torch.float64
        u0 = u0.to(dtype)
        mask = self.boundary_mask.to(device)
        idx = torch.nonzero(~mask, as_tuple=False).squeeze(1)
        idx_np = idx.cpu().numpy()

        # Interior sparse blocks, assembled once, in float64. L = M/dt+a²A is the constant
        # template; M_ii shares its layout so the reaction is a value update.
        M_csr = self.M.to_scipy_coo().tocsr()[idx_np][:, idx_np].astype(np.float64).tocoo()
        A_csr = self.A.to_scipy_coo().tocsr()[idx_np][:, idx_np].astype(np.float64).tocoo()
        M_ii = SparseMatrix.from_scipy_coo(M_csr).to(dtype=dtype, device=device)
        A_ii = SparseMatrix.from_scipy_coo(A_csr).to(dtype=dtype, device=device)
        L_ii = (M_ii * (1.0 / dt) + A_ii * (a * a)).to(dtype=dtype, device=device)
        col, L_vals, M_vals = L_ii.col_indices, L_ii.values, M_ii.values

        def matvec(mat, x):                                   # [b, Ni] -> [b, Ni]
            return (mat @ x.T).T

        B = u0.shape[0]
        traj = torch.zeros(B, n_steps + 1, self.n_nodes, device=device, dtype=dtype)
        traj[:, 0] = apply_zero_boundary(u0, mask)

        for lo in range(0, B, chunk):
            hi = min(lo + chunk, B)
            u_old = traj[lo:hi, 0, idx].clone()               # [b, Ni]
            for step in range(1, n_steps + 1):
                u = u_old.clone()
                for _ in range(newton_max):
                    if integrator == "backward_euler":
                        reac = (eps * eps) * (u - u ** 3)     # ε²(u-u³) fully implicit
                        r_sign = -1.0
                        w = (eps * eps) * (3.0 * u * u - 1.0)
                    else:                                     # convex_concave (Eyre split)
                        reac = (eps * eps) * (u ** 3 - u_old)
                        r_sign = +1.0
                        w = 3.0 * (eps * eps) * (u * u)
                    r = matvec(M_ii, (u - u_old) / dt) \
                        + (a * a) * matvec(A_ii, u) \
                        + r_sign * matvec(M_ii, reac)         # [b, Ni]
                    if r.norm(dim=1).max() < newton_tol:
                        break
                    J_vals = L_vals.unsqueeze(0) + M_vals.unsqueeze(0) * w[:, col]   # [b, nnz]
                    du = L_ii.solve_batch(J_vals, -r, backend=solver_backend)        # [b, Ni]
                    u = u + du
                traj[lo:hi, step, idx] = u
                u_old = u
        traj = traj[0] if single else traj
        return traj.to(out_dtype)                     # cast fp64 solve back to caller's dtype


# ============================== Stokes (saddle point) ==============================

class _StokesBilinearForm(MixedElementAssembler):
    r"""Taylor-Hood Stokes bilinear form
    :math:`\mu\,\nabla u : \nabla v - p\,\nabla\cdot v - q\,\nabla\cdot u`.

    Mirrors TensorMesh's ``examples/fluid/stokes_taylor_hood``; the only difference is
    that we hand it a *structured* ``quad9`` mesh, so the Q2 velocity field is Q2 and the
    order-1 pressure field is Q1 on the cell corners.
    """

    fields = [Field(trial="u", test="v", order=2, components=2),
              Field(trial="p", test="q", order=1)]

    def __post_init__(self, mu=1.0):
        self.mu = mu

    def forward(self, gradu, p, gradv, q):
        return (self.mu * (gradu * gradv).sum()
                - p * gradv.diagonal().sum()
                - q * gradu.diagonal().sum())


class StokesProblem(FEMOperator):
    r"""Discrete Taylor-Hood (Q2/Q1) operator for the stationary Stokes equations

    .. math::
        -\mu\Delta u + \nabla p = f,\qquad \nabla\cdot u = 0,\qquad u|_{\partial\Omega}=0,

    assembled as the symmetric **saddle-point** system

    .. math::
        \mathcal K c = \begin{pmatrix} A & B^\top \\ B & 0\end{pmatrix}
        \begin{pmatrix}\mathbf u \\ \mathbf p\end{pmatrix}
        = \begin{pmatrix}\mathbf b \\ \mathbf 0\end{pmatrix}.

    Unlike the three scalar PDEs above, the unknown is a *pair* of fields living on two
    grids: velocity on the fine Q2 node grid ``[Ny, Nx]`` (2 components) and pressure on
    the Q1 corner grid ``[ny_p, nx_p] = [(Ny+1)/2, (Nx+1)/2]`` — the ``[::2, ::2]``
    subgrid. :meth:`from_grid` performs exactly that split on the FNO's 3-channel output,
    and :meth:`pack` assembles the monolithic DOF vector in TensorMesh's block layout
    (velocity first, node-major/component-minor; then pressure).

    Two structural facts drive the design:

    * ``A`` decouples across velocity components and equals :math:`\mu` times the Q2
      scalar stiffness that :class:`FEMOperator` already assembles — so the geometric
      multigrid built for the scalar Laplacian preconditions the velocity block directly.
    * :math:`B^\top \mathbf 1 = 0` on the zero-BC velocity space, i.e. the residual is
      **invariant** under a constant pressure shift. The constant mode is therefore a pure
      gauge: it is fixed by :meth:`project_pressure_gauge` (zero mean in the FE
      :math:`L^2` inner product), never by the loss.

    Parameters
    ----------
    mesh : tensormesh.Mesh
        A structured ``quad9`` mesh from :func:`meshing.structured_quad9_mesh`.
    nx_p, ny_p : int
        Pressure (cell-corner) grid size; the fine grid is ``2nx_p-1`` by ``2ny_p-1``.
    mu : float
        Dynamic viscosity :math:`\mu`.
    """

    def __init__(self, mesh, nx_p: int, ny_p: int, mu: float = 1.0,
                 quadrature_order: int = 5):
        # FEMOperator gives us the Q2 scalar stiffness A and Q2 scalar mass M on the fine
        # grid. A is the per-component velocity block (up to mu); M builds the load vector.
        super().__init__(mesh, quadrature_order=quadrature_order)
        self.mu = float(mu)
        self.nx_p, self.ny_p = int(nx_p), int(ny_p)
        self.nx_u, self.ny_u = 2 * nx_p - 1, 2 * ny_p - 1
        self.n_u = self.n_nodes                                   # fine (Q2) nodes
        self.n_p = self.nx_p * self.ny_p

        # --- monolithic saddle-point operator -------------------------------------
        asm = _StokesBilinearForm.from_mesh(mesh, quadrature_order=quadrature_order, mu=mu)
        layout = asm.layout
        assert layout.n_nodes("u") == self.n_u, "velocity field is not the mesh node set"
        assert layout.n_nodes("p") == self.n_p, (
            f"pressure field has {layout.n_nodes('p')} nodes, expected {self.n_p}")
        self.K = asm()
        self.n_dofs = int(layout.n_dofs)
        self.off_p = int(layout.offsets["p"])                     # = 2 * n_u

        # Dirichlet DOFs: all velocity components on the boundary. The pressure gauge is
        # handled by projection, but the reference solve needs a pin to make K invertible.
        self.register_buffer("dirichlet_mask", layout.dof_mask("u", self.boundary_mask))
        self._pressure_pin = int(layout.dof_index("p", int(layout.node_ids("p")[0])))

        # --- pressure mass matrix (Q1 on the corner grid) --------------------------
        # The Q1 pressure space on the quad9 mesh is exactly the Q1 space of the corner
        # grid, so we can assemble M_p with the ordinary scalar machinery.
        p_mesh = structured_quad_mesh(nx=nx_p, ny=ny_p)
        self.M_p = MassElementAssembler.from_mesh(
            p_mesh, quadrature_order=quadrature_order)(p_mesh.points)
        # lumped pressure mass (row sums) — the Schur-complement surrogate and the weight
        # of the zero-mean pressure gauge.
        self.register_buffer(
            "m_p_lumped",
            self._spmm(self.M_p, torch.ones(self.n_p, dtype=self.M_p.dtype)).float())

    # -- device/dtype management ------------------------------------------------
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.K = self.K.to(*args, **kwargs)
        self.M_p = self.M_p.to(*args, **kwargs)
        return self

    # ------------------------------------------------------------------ packing
    @property
    def grid_size(self):
        """Velocity (fine) grid ``(nx, ny)`` — what the FNO consumes and emits."""
        return (self.nx_u, self.ny_u)

    @property
    def pgrid_size(self):
        """Pressure (corner) grid ``(nx, ny)``."""
        return (self.nx_p, self.ny_p)

    def from_grid(self, out_grid: torch.Tensor):
        """FNO output ``[B, 3, Ny, Nx]`` → ``(u_node [B, n_u, 2], p_node [B, n_p])``.

        Channels are ``(u_x, u_y, p)``. The pressure channel is read on the ``[::2, ::2]``
        subgrid — the Q1 corner nodes — which is why the two spaces need no index table.
        """
        nx, ny = self.grid_size
        ux = grid_to_node(out_grid[:, 0], nx, ny)
        uy = grid_to_node(out_grid[:, 1], nx, ny)
        u_node = torch.stack([ux, uy], dim=-1)                    # [B, n_u, 2]
        p_grid = out_grid[:, 2, ::2, ::2]                         # [B, ny_p, nx_p]
        p_node = grid_to_node(p_grid, self.nx_p, self.ny_p)       # [B, n_p]
        return u_node, p_node

    def to_grid(self, u_node: torch.Tensor, p_node: torch.Tensor):
        """Inverse of :meth:`from_grid` for visualization: ``(u [B,2,Ny,Nx], p [B,ny_p,nx_p])``."""
        nx, ny = self.grid_size
        u_grid = torch.stack([node_to_grid(u_node[..., 0], nx, ny),
                              node_to_grid(u_node[..., 1], nx, ny)], dim=-3)
        return u_grid, node_to_grid(p_node, self.nx_p, self.ny_p)

    def pack(self, u_node: torch.Tensor, p_node: torch.Tensor) -> torch.Tensor:
        """``(u [B,n_u,2], p [B,n_p])`` → monolithic DOF vector ``[B, n_dofs]``.

        Velocity is node-major / component-minor, matching TensorMesh's
        ``dof = offset + n_local * c + comp`` block layout.
        """
        return torch.cat([u_node.reshape(*u_node.shape[:-2], 2 * self.n_u), p_node], dim=-1)

    def unpack(self, c: torch.Tensor):
        """Monolithic DOF vector ``[B, n_dofs]`` → ``(u [B, n_u, 2], p [B, n_p])``."""
        u = c[..., :self.off_p].reshape(*c.shape[:-1], self.n_u, 2)
        return u, c[..., self.off_p:]

    # ------------------------------------------------------ boundary & gauge
    def project_velocity_bc(self, u_node: torch.Tensor) -> torch.Tensor:
        """Zero both velocity components on the Dirichlet boundary. ``[..., n_u, 2]``."""
        mask = self.boundary_mask.to(u_node.device)
        return u_node.masked_fill(mask.reshape(*([1] * (u_node.dim() - 2)), -1, 1), 0.0)

    def project_pressure_gauge(self, p_node: torch.Tensor) -> torch.Tensor:
        r"""Remove the constant mode: :math:`p \leftarrow p - \frac{\mathbf 1^\top M_p p}
        {\mathbf 1^\top M_p \mathbf 1}`.

        The saddle-point residual is blind to this mode (:math:`B^\top\mathbf 1 = 0`), so
        it is never determined by a label-free loss. Fixing the gauge the same way for the
        prediction and the reference is what makes the pressure error meaningful.
        """
        w = self.m_p_lumped.to(dtype=p_node.dtype, device=p_node.device)
        mean = (p_node * w).sum(dim=-1, keepdim=True) / w.sum()
        return p_node - mean

    def _mask_dirichlet(self, r: torch.Tensor) -> torch.Tensor:
        """Zero the residual on constrained (boundary velocity) DOFs. ``[..., n_dofs]``."""
        mask = self.dirichlet_mask.to(r.device)
        return r.masked_fill(mask, 0.0)

    # ------------------------------------------------------------------ operators
    def load_vector(self, f_node: torch.Tensor) -> torch.Tensor:
        r"""Consistent monolithic load :math:`b = (M_u f,\ 0)`. ``f_node``: ``[B, n_u, 2]``.

        ``M_u`` is the Q2 vector mass matrix, which is the scalar ``M`` applied to each
        component — the same ``b = Mf`` convention :class:`PoissonProblem` uses, so the
        reference solve and the physics residual share one right-hand side by construction.
        """
        b_u = torch.stack([self._spmm(self.M, f_node[..., 0]),
                           self._spmm(self.M, f_node[..., 1])], dim=-1)      # [B, n_u, 2]
        b_p = f_node.new_zeros(*f_node.shape[:-2], self.n_p)
        return self.pack(b_u, b_p)

    def residual(self, u_node: torch.Tensor, p_node: torch.Tensor,
                 f_node: torch.Tensor) -> torch.Tensor:
        r"""Boundary-masked saddle-point residual :math:`R = \mathcal K c - b`.

        ``u_node``: ``[B, n_u, 2]``, ``p_node``: ``[B, n_p]``, ``f_node``: ``[B, n_u, 2]``.
        The velocity is projected to zero on the Dirichlet boundary first and the returned
        residual is zeroed on those DOFs. Fully differentiable in ``u`` and ``p``.
        Returns ``[B, n_dofs]``: momentum rows then continuity rows.
        """
        c = self.pack(self.project_velocity_bc(u_node), p_node)
        r = self._spmm(self.K, c) - self.load_vector(f_node)
        return self._mask_dirichlet(r)

    # ------------------------------------------------------------ reference solve
    @torch.no_grad()
    def fem_reference(self, f_node: torch.Tensor, chunk: int = 64):
        r"""Discrete Taylor-Hood solution of :math:`\mathcal K c = (M_u f, 0)`.

        This is the build-time ground truth: it zeroes exactly the residual that
        :meth:`residual` computes, so a perfectly trained label-free model reproduces it.
        The system is solved once per chunk with a **batched right-hand side** (one sparse
        factorization, many RHS) in float64 — the pressure pin makes ``K`` invertible and
        the pin is removed afterwards by the zero-mean gauge.

        ``f_node``: ``[B, n_u, 2]``. Returns ``(u [B, n_u, 2], p [B, n_p])`` in the
        caller's dtype, with zero velocity boundary and zero-mean pressure.
        """
        single = (f_node.dim() == 2)
        if single:
            f_node = f_node.unsqueeze(0)
        out_dtype, device = f_node.dtype, f_node.device
        B = f_node.shape[0]

        K = self.K.to(dtype=torch.float64, device=device)
        mask = self.dirichlet_mask.clone().to(device)
        mask[self._pressure_pin] = True
        condenser = Condenser(mask, torch.zeros(int(mask.sum()), dtype=torch.float64,
                                                device=device))

        u_out = torch.zeros(B, self.n_u, 2, dtype=torch.float64, device=device)
        p_out = torch.zeros(B, self.n_p, dtype=torch.float64, device=device)
        for lo in range(0, B, chunk):
            hi = min(lo + chunk, B)
            rhs = self.load_vector(f_node[lo:hi].to(torch.float64))                 # [b, n_dofs]
            K_i, f_i = condenser(K, rhs.transpose(0, 1).contiguous())               # [n_i, b]
            sol = condenser.recover(K_i.solve(f_i))
            if sol.dim() == 1:                       # a single RHS may come back squeezed
                sol = sol.unsqueeze(-1)
            u_out[lo:hi], p_out[lo:hi] = self.unpack(sol.transpose(0, 1))           # [b, n_dofs]

        p_out = self.project_pressure_gauge(p_out)
        u_out = self.project_velocity_bc(u_out)
        if single:
            u_out, p_out = u_out[0], p_out[0]
        return u_out.to(out_dtype), p_out.to(out_dtype)

    # ------------------------------------------------------------------- FE norms
    def velocity_l2(self, u_node: torch.Tensor) -> torch.Tensor:
        r"""FE :math:`L^2` norm :math:`\sqrt{u^\top M_u u}` per sample — ``[B]``."""
        acc = sum((u_node[..., k] * self._spmm(self.M, u_node[..., k])).sum(dim=-1)
                  for k in range(2))
        return acc.clamp_min(0).sqrt()

    def pressure_l2(self, p_node: torch.Tensor) -> torch.Tensor:
        r"""FE :math:`L^2` norm :math:`\sqrt{p^\top M_p p}` per sample — ``[B]``."""
        Mp = self._spmm(self.M_p, p_node)
        return (p_node * Mp).sum(dim=-1).clamp_min(0).sqrt()
