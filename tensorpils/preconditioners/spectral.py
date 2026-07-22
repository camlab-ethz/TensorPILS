r"""Spectral preconditioner: an exact function of the interior stiffness ``A``.

Realizes the two families analysed in ``preconditioner_notes/transition_preconditioner.tex``
via a single eigendecomposition of the SPD interior block ``A = Q Λ Qᵀ``:

* ``kind="blend"``  — convex blend  ``P_t = (1-t) I + t A^{-1}``, spectral map
  ``p_t(λ) = (1-t) + t/λ``;
* ``kind="power"``  — fractional power ``P_s = A^{-s}``, spectral map ``p_s(λ) = λ^{-s}``.

The parameter ``strength`` is ``t`` (blend) or ``s`` (power) in ``[0, 1]``. Both endpoints
are exact and need no special-casing: ``strength=0`` gives ``P = I`` (the no-preconditioner /
residual-least-squares anchor) and ``strength=1`` gives ``P = A^{-1}`` (the supervised
anchor). The action on a residual is ``P r = Q ( p(Λ) ⊙ (Qᵀ r) )``, restricted to the
interior nodes and embedded back with zeros on the boundary.
"""

import math

import torch

from .base import Preconditioner

__all__ = ["SpectralPreconditioner", "SineSpectralPreconditioner"]


def _spectral_map(evals: torch.Tensor, kind: str, strength: float) -> torch.Tensor:
    """Eigenvalue map ``p(λ)`` for the chosen family. ``evals``: positive eigenvalues of A."""
    if kind == "blend":
        return (1.0 - strength) + strength / evals
    if kind == "power":
        return evals ** (-strength)
    raise ValueError(f"kind must be 'blend' or 'power', got {kind!r}")


class SpectralPreconditioner(Preconditioner):
    r"""Exact spectral preconditioner ``P = Q p(Λ) Qᵀ`` on the interior stiffness block.

    Parameters
    ----------
    A_full : torch.Tensor
        Assembled stiffness on the full grid (dense or sparse), shape ``[N_full, N_full]``.
        Only its interior block (rows/cols of non-boundary nodes) is used.
    boundary_mask : torch.Tensor
        Boolean mask of length ``N_full``; ``True`` on Dirichlet boundary nodes.
    kind : {"blend", "power"}
    strength : float in [0, 1]
        ``t`` for the blend, ``s`` for the power. ``0`` → ``P=I``; ``1`` → ``P=A^{-1}``.
    dtype : torch.dtype
        Dtype of the cached operator / applied arithmetic (default ``float32``). The
        eigendecomposition itself is always computed in ``float64`` for accuracy.
    """

    def __init__(self, A_full: torch.Tensor, boundary_mask: torch.Tensor,
                 kind: str = "blend", strength: float = 1.0,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if kind not in ("blend", "power"):
            raise ValueError(f"kind must be 'blend' or 'power', got {kind!r}")
        if not (0.0 <= float(strength) <= 1.0):
            raise ValueError(f"strength must be in [0, 1], got {strength}")
        self.kind = kind
        self.strength = float(strength)

        # A_full may be a dense tensor, a torch sparse tensor, or tensormesh's
        # torch_sla.SparseMatrix — all expose .to_dense().
        A_dense = A_full.to_dense() if hasattr(A_full, "to_dense") else A_full
        A_dense = A_dense.to(torch.float64)
        mask = boundary_mask.to(torch.bool)
        interior = (~mask).nonzero(as_tuple=False).squeeze(-1)      # [N_int]
        self.n_full = int(A_dense.shape[0])

        A_II = A_dense.index_select(0, interior).index_select(1, interior)
        A_II = 0.5 * (A_II + A_II.T)                                # symmetrize (guard)
        evals, Q = torch.linalg.eigh(A_II)                         # ascending, SPD
        p = _spectral_map(evals, kind, self.strength)

        self.register_buffer("interior", interior)
        self.register_buffer("Q", Q.to(dtype))
        self.register_buffer("p_diag", p.to(dtype))
        self.register_buffer("evals", evals.to(dtype))

    @classmethod
    def from_problem(cls, problem, kind: str = "blend", strength: float = 1.0,
                     dtype: torch.dtype = torch.float32) -> "SpectralPreconditioner":
        """Build from a :class:`~tensorpils.physics.PoissonProblem` (uses ``A`` and mask)."""
        return cls(problem.A, problem.boundary_mask, kind=kind, strength=strength, dtype=dtype)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Apply ``P``. ``r``: ``[B, N_full]`` or ``[N_full]``; boundary entries zeroed out."""
        squeeze = r.dim() == 1
        if squeeze:
            r = r.unsqueeze(0)
        r_I = r.index_select(-1, self.interior)        # [B, N_int]  (drop boundary)
        rhat = r_I @ self.Q                            # Qᵀ r_I  (project to eigenbasis)
        ehat = rhat * self.p_diag                      # scale each mode by p(λ)
        e_I = ehat @ self.Q.transpose(0, 1)           # Q ehat  (back to nodal basis)
        e = e_I.new_zeros(r.shape[0], self.n_full).index_copy(1, self.interior, e_I)
        return e.squeeze(0) if squeeze else e

    # -------------------- diagnostics (closed form from the spectrum) --------------------
    @property
    def g_spectrum(self) -> torch.Tensor:
        """Preconditioned spectrum ``g(λ) = λ p(λ)`` (eigenvalues of ``PA``)."""
        return self.evals * self.p_diag

    @property
    def cond_PA(self) -> float:
        g = self.g_spectrum
        return (g.max() / g.min()).item()

    @property
    def cond_H(self) -> float:
        """Condition number of the squared-loss Hessian ``H = A Pᵀ P A``: ``κ(PA)²``."""
        c = self.cond_PA
        return c * c

    def report(self) -> None:
        print(f"[spectral precond] kind={self.kind}  strength={self.strength:.3f}  "
              f"cond(PA)={self.cond_PA:.3e}  cond(H)={self.cond_H:.3e}")


def _dst1_matrix(n: int) -> torch.Tensor:
    r"""Orthonormal 1D discrete sine transform (DST-I) matrix ``S`` of size ``n``,
    ``S_{jk} = sqrt(2/(n+1)) sin(pi j k / (n+1))`` for ``j,k = 1..n``. It is symmetric and
    orthogonal, so ``S = Sᵀ = S^{-1}``: the same matrix transforms forward and back."""
    j = torch.arange(1, n + 1, dtype=torch.float64)
    return math.sqrt(2.0 / (n + 1)) * torch.sin(torch.outer(j, j) * math.pi / (n + 1))


def _stiffness_mass_1d_eigs(n: int, h: float):
    r"""Closed-form eigenvalues of the 1D ``Q1`` interior stiffness ``K`` and (consistent) mass
    ``M`` on ``n`` interior nodes with spacing ``h``, both in the DST-I basis
    (``preconditioner_notes/transition_preconditioner.tex``, §2):
    ``mu_k = (4/h) sin^2(theta_k/2)`` (stiffness), ``nu_k = (h/3)(2 + cos theta_k)`` (mass),
    with ``theta_k = k*pi/(n+1)``."""
    k = torch.arange(1, n + 1, dtype=torch.float64)
    theta = k * math.pi / (n + 1)
    mu = (4.0 / h) * torch.sin(theta / 2.0) ** 2
    nu = (h / 3.0) * (2.0 + torch.cos(theta))
    return mu, nu


class SineSpectralPreconditioner(Preconditioner):
    r"""Fast spectral preconditioner on a **uniform** grid via the 2D sine transform.

    Same object as :class:`SpectralPreconditioner` — ``P = (1-t)I + t A^{-1}`` (``kind="blend"``)
    or ``A^{-s}`` (``kind="power"``) — but it exploits that the bilinear-``Q1`` interior
    stiffness on a uniform rectangular grid is ``A = K_y⊗M_x + M_y⊗K_x``, diagonalized exactly by
    the tensor DST-I with closed-form eigenvalues ``Λ_{k,l} = μ^{K_y}_k ν^{M_x}_l + ν^{M_y}_k
    μ^{K_x}_l``. So it needs **no eigendecomposition and no dense eigenvector matrix**: each
    application is two small per-axis matmuls, ``O(N^{1.5})``, which scales to 128²/256² where the
    dense route (``O(N^3)`` build, ``N×N`` cache) is infeasible. Results are identical to the dense
    class up to floating point (see the equivalence test).

    Parameters
    ----------
    boundary_mask : torch.Tensor
        Boolean mask of length ``nx*ny`` (row-major ``k = i*nx + j``); ``True`` on the Dirichlet
        outer frame. Must be exactly the frame of the grid (else a ``ValueError`` is raised).
    nx, ny : int
        Node counts along x (columns) and y (rows); interior is ``(ny-2)×(nx-2)``.
    xlims, ylims : tuple(float, float)
        Domain extents; set the spacings ``h_x = (x1-x0)/(nx-1)``, ``h_y = (y1-y0)/(ny-1)``.
    kind : {"blend", "power"}
    strength : float in [0, 1]
        ``t`` (blend) or ``s`` (power). ``0`` → ``P=I``; ``1`` → ``P=A^{-1}``.
    dtype : torch.dtype
        Dtype of the cached transforms / applied arithmetic (default ``float32``); the transforms
        and eigenvalues are built in ``float64`` and cast down.
    """

    def __init__(self, boundary_mask: torch.Tensor, nx: int, ny: int,
                 xlims=(0.0, 1.0), ylims=(0.0, 1.0),
                 kind: str = "blend", strength: float = 1.0,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if kind not in ("blend", "power"):
            raise ValueError(f"kind must be 'blend' or 'power', got {kind!r}")
        if not (0.0 <= float(strength) <= 1.0):
            raise ValueError(f"strength must be in [0, 1], got {strength}")
        self.kind = kind
        self.strength = float(strength)
        self.nx, self.ny = int(nx), int(ny)
        self.n_full = self.nx * self.ny

        mask = boundary_mask.to(torch.bool)
        if mask.numel() != self.n_full:
            raise ValueError(f"boundary_mask has {mask.numel()} entries, expected nx*ny="
                             f"{self.n_full}")
        # The sine realization requires the structured uniform mesh: mask must be the outer frame.
        frame = torch.zeros(self.ny, self.nx, dtype=torch.bool, device=mask.device)
        frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = True
        if not torch.equal(mask.view(self.ny, self.nx), frame):
            raise ValueError("SineSpectralPreconditioner requires boundary_mask to be exactly the "
                             "outer frame of an nx*ny structured uniform grid.")

        n_x, n_y = self.nx - 2, self.ny - 2
        h_x = (xlims[1] - xlims[0]) / (self.nx - 1)
        h_y = (ylims[1] - ylims[0]) / (self.ny - 1)
        Sx = _dst1_matrix(n_x)
        Sy = _dst1_matrix(n_y)
        muKx, nuMx = _stiffness_mass_1d_eigs(n_x, h_x)
        muKy, nuMy = _stiffness_mass_1d_eigs(n_y, h_y)
        # Λ grid [n_y, n_x] (row-major y-outer, x-inner, matching k = i*nx + j).
        Lam = muKy[:, None] * nuMx[None, :] + nuMy[:, None] * muKx[None, :]
        p_grid = _spectral_map(Lam, kind, self.strength)

        interior = (~mask).nonzero(as_tuple=False).squeeze(-1)      # [N_int], ascending = row-major
        self.register_buffer("interior", interior)
        self.register_buffer("Sx", Sx.to(dtype))
        self.register_buffer("Sy", Sy.to(dtype))
        self.register_buffer("p_grid", p_grid.to(dtype))
        # Flat spectrum for the diagnostics (identical API to the dense class).
        self.register_buffer("evals", Lam.reshape(-1).to(dtype))
        self.register_buffer("p_diag", p_grid.reshape(-1).to(dtype))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Apply ``P``. ``r``: ``[B, N_full]`` or ``[N_full]``; boundary entries zeroed out."""
        squeeze = r.dim() == 1
        if squeeze:
            r = r.unsqueeze(0)
        B = r.shape[0]
        r_I = r.index_select(-1, self.interior)                    # [B, N_int] (drop boundary)
        rg = r_I.view(B, self.ny - 2, self.nx - 2)                 # interior grid [B, n_y, n_x]
        rhat = (self.Sy @ rg) @ self.Sx                            # forward DST on both axes
        ehat = rhat * self.p_grid                                   # scale each mode by p(Λ)
        e_I = ((self.Sy @ ehat) @ self.Sx).reshape(B, -1)          # inverse DST (S = S^{-1})
        e = e_I.new_zeros(B, self.n_full).index_copy(1, self.interior, e_I)
        return e.squeeze(0) if squeeze else e

    # -------------------- diagnostics (closed form from the spectrum) --------------------
    @property
    def g_spectrum(self) -> torch.Tensor:
        """Preconditioned spectrum ``g(λ) = λ p(λ)`` (eigenvalues of ``PA``)."""
        return self.evals * self.p_diag

    @property
    def cond_PA(self) -> float:
        g = self.g_spectrum
        return (g.max() / g.min()).item()

    @property
    def cond_H(self) -> float:
        """Condition number of the squared-loss Hessian ``H = A Pᵀ P A``: ``κ(PA)²``."""
        c = self.cond_PA
        return c * c

    def report(self) -> None:
        print(f"[sine precond] kind={self.kind}  strength={self.strength:.3f}  "
              f"grid={self.nx}x{self.ny}  cond(PA)={self.cond_PA:.3e}  cond(H)={self.cond_H:.3e}")
