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

import torch

from .base import Preconditioner

__all__ = ["SpectralPreconditioner"]


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
