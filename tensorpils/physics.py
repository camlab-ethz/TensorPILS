"""FEM operators built on TensorMesh's assembled matrices.

Three PDE operators live here, each assembled once from a :class:`tensormesh.Mesh` and each
exposing a batched, autograd-differentiable interface used by the physics-informed losses:

* :class:`PoissonProblem` — static :math:`-\\Delta u = f`
* :class:`ACProblem`      — time-dependent Allen–Cahn :math:`u_t = a^2 \\Delta u + \\epsilon^2 u(1-u^2)`,
  discretised in time by the convex–concave (Eyre) splitting
* :class:`StokesProblem`  — static Stokes saddle point :math:`-\\mu\\Delta u + \\nabla p = f`,
  :math:`\\nabla\\cdot u = 0`, on a Taylor-Hood P2/P1 pair on triangles

They share :class:`FEMOperator`, which assembles the stiffness ``A`` and mass ``M``
matrices (via TensorMesh's ``LaplaceElementAssembler`` / ``MassElementAssembler``),
holds the Dirichlet ``boundary_mask``, and confines the ``[n_nodes, batch]`` transpose
required by sparse mat–vec products. All public methods take/return node fields in
``[batch, n_nodes]`` (or ``[n_nodes]``) convention.
"""

import torch
import torch.nn as nn

import meshio
from tensormesh import (Condenser, Field, LaplaceElementAssembler, Mesh,
                        MassElementAssembler, MixedElementAssembler)

__all__ = ["FEMOperator", "PoissonProblem", "ACProblem", "StokesProblem",
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
        The mesh. ``mesh.boundary_mask`` defines the Dirichlet nodes.
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
    r"""Discrete FEM operator for :math:`-\Delta u = f` with zero Dirichlet BC."""

    # ------------------------------------------------------------------ residual
    def residual(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        r"""Boundary-masked Galerkin residual :math:`R = A\,u_{\mathrm{bc}} - M f`.

        ``u`` is first projected to zero on the Dirichlet boundary; the returned
        residual is also zeroed there. Fully differentiable in ``u``.
        """
        u_bc = apply_zero_boundary(u, self.boundary_mask)
        r = self._spmm(self.A, u_bc) - self.load_vector(f)
        return apply_zero_boundary(r, self.boundary_mask)


class ACProblem(FEMOperator):
    r"""Discrete FEM operator for the Allen–Cahn equation

    .. math::
        \partial_t u = a^2 \Delta u + \epsilon^2 u(1-u^2)

    on the unit square with zero Dirichlet BC. ``a`` is the diffusion coefficient and
    ``\epsilon`` the reaction strength (the large factor multiplies the double-well term, not
    the Laplacian).

    Time is discretised by the convex–concave (Eyre) splitting: the convex cubic term is
    implicit and the concave linear term explicit at the old level, which gives the weak-form
    nodal step residual (group / product FEM, consistent mass ``M`` and stiffness ``A``)

    .. math::
        R = M\,\frac{u^{n+1} - u^n}{\Delta t} + a^2 A\,u^{n+1}
            + \epsilon^2 M\,\big((u^{n+1})^3 - u^n\big).

    Its Newton Jacobian :math:`M/\Delta t + a^2A + 3\epsilon^2 M\,\mathrm{diag}(u^2)` is SPD for
    every ``u`` and ``Δt``, so the scheme is unconditionally energy-stable. The label-free loss
    drives ``R`` to zero, and the FEM reference trajectory (:meth:`fem_reference`) solves
    ``R = 0`` per step by Newton, so it exactly zeroes this residual — the discrete target the
    physics loss is trained toward. ``A``/``M`` are the same matrices :class:`PoissonProblem`
    assembles.
    """

    # ------------------------------------------------------------------ residual
    def residual(self, u_curr: torch.Tensor, u_next: torch.Tensor,
                 a: float, eps: float, dt: float) -> torch.Tensor:
        r"""Boundary-masked weak-form convex–concave Allen–Cahn step residual (``/dt`` scaling).

        ``u_curr``/``u_next`` are :math:`u^n, u^{n+1}` in ``[N]`` or ``[B, N]``. Both are projected
        to zero on the Dirichlet boundary before use and the returned residual is also zeroed
        there. Fully differentiable in the inputs:

        .. math::
            M\frac{u^{n+1}-u^n}{\Delta t} + a^2Au^{n+1} + \epsilon^2 M((u^{n+1})^3-u^n).

        The ``/dt`` scaling matches :meth:`fem_reference`, so a least-squares loss built on it has
        a ``dt``-independent gradient scale, and the reference exactly zeros this residual."""
        mask = self.boundary_mask
        uc = apply_zero_boundary(u_curr, mask)
        un = apply_zero_boundary(u_next, mask)
        reaction = (eps * eps) * (uc - un ** 3)                   # linear term explicit (uⁿ)
        r = self._spmm(self.M, (un - uc) / dt) \
            + (a * a) * self._spmm(self.A, un) \
            - self._spmm(self.M, reaction)
        return apply_zero_boundary(r, mask)

    # ------------------------------------------ FEM reference solve (sparse float64)
    @torch.no_grad()
    def fem_reference(self, u0: torch.Tensor, a: float, eps: float, dt: float,
                      n_steps: int, newton_tol: float = 1e-8, newton_max: int = 20,
                      chunk: int = 64, solver_backend: str = "auto") -> torch.Tensor:
        r"""Sparse-solve Newton reference trajectory --- the build-time ground truth.

        Solves ``R(u^{n+1}; u^n) = 0`` each step by Newton on the interior DOFs (Dirichlet nodes
        held at 0). ``u0``: ``[N]`` or ``[B, N]``. Returns the trajectory ``[B, n_steps+1, N]`` (or
        ``[n_steps+1, N]`` for a single sample), processing ``chunk`` samples at a time.

        Each Newton system is solved with a **batched sparse** solve (``SparseMatrix.solve_batch``:
        shared sparsity, per-sample values). ``solve_batch`` loops per sample internally, so it is
        genuinely sparse but not GPU-parallel; SPD is auto-detected.

        The interior Jacobian is ``J = M/dt + a^2 A + M diag(w)`` with ``w = 3 eps^2 u^2``; its
        values are assembled per Newton step as ``L.values + M.values * w[:, col]`` on the shared
        ``(row,col)`` layout of ``L = M/dt + a^2 A``. ``solver_backend`` is passed through
        (``"auto"`` picks scipy on CPU, cupy/cuDSS on CUDA if installed)."""
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
                    reac = (eps * eps) * (u ** 3 - u_old)     # cubic implicit, linear explicit
                    w = 3.0 * (eps * eps) * (u * u)           # J = L + 3ε²M diag(u²)  (SPD)
                    r = matvec(M_ii, (u - u_old) / dt) \
                        + (a * a) * matvec(A_ii, u) \
                        + matvec(M_ii, reac)                  # [b, Ni]
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

    Mirrors TensorMesh's ``examples/fluid/stokes_taylor_hood``. On a ``triangle6`` mesh the
    order-2 velocity field is P2 and the order-1 pressure field is P1 on the triangle corners.
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
    r"""Discrete Taylor-Hood **P2/P1** operator for the stationary Stokes equations

    .. math::
        -\mu\Delta u + \nabla p = f,\qquad \nabla\cdot u = 0,\qquad u|_{\partial\Omega}=0,

    on a triangulation, assembled as the symmetric **saddle-point** system

    .. math::
        \mathcal K c = \begin{pmatrix} A & B^\top \\ B & 0\end{pmatrix}
        \begin{pmatrix}\mathbf u \\ \mathbf p\end{pmatrix}
        = \begin{pmatrix}\mathbf b \\ \mathbf 0\end{pmatrix}.

    The unknown is a *pair* of fields: velocity (2 components) on the P2 node set, which is the
    mesh's node set, and pressure on the P1 nodes — the triangle *corners*, an arbitrary subset
    of the P2 numbering read from the assembly layout (:attr:`p_node_ids`). :meth:`from_nodes`
    splits a model's 3-channel output accordingly, and :meth:`pack` assembles the monolithic DOF
    vector in TensorMesh's block layout (velocity first, node-major/component-minor; then
    pressure).

    Two structural facts drive the design:

    * ``A`` decouples across velocity components and equals :math:`\mu` times the P2 scalar
      stiffness that :class:`FEMOperator` already assembles — so a scalar V-cycle preconditions
      the velocity block directly.
    * :math:`B^\top \mathbf 1 = 0` on the zero-BC velocity space, i.e. the residual is
      **invariant** under a constant pressure shift. The constant mode is therefore a pure
      gauge: it is fixed by :meth:`project_pressure_gauge` (zero mean in the FE
      :math:`L^2` inner product), never by the loss.

    Parameters
    ----------
    mesh : tensormesh.Mesh
        A ``triangle6`` mesh, e.g. from :func:`~tensorpils.meshing.obstacle_mesh`.
    mu : float
        Dynamic viscosity :math:`\mu`.
    """

    def __init__(self, mesh, mu: float = 1.0, quadrature_order: int = 5):
        # FEMOperator gives us the P2 scalar stiffness A and P2 scalar mass M. A is the
        # per-component velocity block (up to mu); M builds the load vector.
        super().__init__(mesh, quadrature_order=quadrature_order)
        if "triangle6" not in list(mesh.cells.keys()):
            raise ValueError(
                "Taylor-Hood P2/P1 needs a 'triangle6' mesh (order=2); got cells "
                f"{list(mesh.cells.keys())}. meshing.obstacle_mesh(order=2) builds one.")
        self.mu = float(mu)
        self.n_u = self.n_nodes

        # --- monolithic saddle-point operator -------------------------------------
        asm = _StokesBilinearForm.from_mesh(mesh, quadrature_order=quadrature_order, mu=mu)
        layout = asm.layout
        if layout.n_nodes("u") != self.n_u:
            raise RuntimeError("velocity field is not the mesh node set")
        self.n_p = int(layout.n_nodes("p"))
        self.K = asm()
        self.n_dofs = int(layout.n_dofs)
        self.off_p = int(layout.offsets["p"])                     # = 2 * n_u

        # Dirichlet DOFs: all velocity components on the boundary. The pressure gauge is
        # handled by projection, but the reference solve needs a pin to make K invertible.
        self.register_buffer("dirichlet_mask", layout.dof_mask("u", self.boundary_mask))
        p_ids = layout.node_ids("p")
        self._pressure_pin = int(layout.dof_index("p", int(p_ids[0])))
        # Which P2 nodes carry pressure. The model emits three channels on the P2 node set and
        # the pressure one is gathered here, so the two spaces need no scatter table.
        self.register_buffer("p_node_ids", p_ids.long())

        # --- P1 pressure mass, on a sub-mesh of the corners -----------------------
        # node_ids("p") is sorted, so searchsorted maps a global node id to its pressure DOF
        # index; the sub-mesh is therefore in pressure-DOF order and M_p needs no permutation.
        corners = mesh.cells["triangle6"][:, :3].long()
        local = torch.searchsorted(p_ids.long(), corners.reshape(-1)).reshape(corners.shape)
        p_points = mesh.points[p_ids].detach().cpu().numpy()
        p_mesh = Mesh(meshio.Mesh(points=p_points,
                                  cells=[("triangle", local.detach().cpu().numpy())]),
                      reorder=False)
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
    def from_nodes(self, out_nodes: torch.Tensor):
        """Model output ``[B, n_u, 3]`` -> ``(u_node [B, n_u, 2], p_node [B, n_p])``.

        Channels are ``(u_x, u_y, p)`` on the **P2 node set**, and pressure is gathered at the
        corner nodes. The velocity is used everywhere; the pressure channel's values at the
        edge-midpoint nodes are simply never read, which costs one third of one channel and
        buys the model a single uniform output space.
        """
        u_node = out_nodes[..., :2]
        p_node = out_nodes[..., self.p_node_ids, 2]
        return u_node, p_node

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

        ``M_u`` is the P2 vector mass matrix, which is the scalar ``M`` applied to each
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
