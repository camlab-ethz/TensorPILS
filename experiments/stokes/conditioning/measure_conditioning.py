r"""Measure the conditioning that the theory predicts for the Stokes saddle point, in float64.

The methodology note claims three things about the Gauss--Newton matrix of a Stokes loss
(``notes``/paper Lemma "saddle point preconditioners"):

===========================  ==================================  ==================
loss                         Gauss--Newton matrix                predicted kappa
===========================  ==================================  ==================
``½‖Kc−b‖²`` (galerkin)      ``Jᵀ K²  J``                        ``O(h^-4)``
``½ rᵀPr``   (pls)           ``Jᵀ K P K J``                      ``O(h^-2)``
``½‖P(Kc−b)‖²`` (dead end)   ``Jᵀ (PK)ᵀ(PK) J``                  ``O(h^-4)``
===========================  ==================================  ==================

and, underneath all three, that the *eigenvalue* ratio ``max|λ(PK)|/min|λ(PK)|`` stays ``O(1)``
while ``kappa(P)`` itself grows like ``h^-2`` — which is exactly what makes the middle row the
only viable one. This script measures all of it directly, so the h-scaling in the paper is a
measurement rather than an assertion.

**What is diagonalized.** The loss never sees the full DOF space: the velocity is projected to
zero on the Dirichlet boundary and the residual is zeroed there, and the pressure is projected to
zero mean in the lumped-``M_p`` inner product (the residual is blind to that mode, ``Bᵀ1 = 0``).
The honest object is therefore ``K`` restricted to the *admissible* subspace

    S = {interior velocity DOFs} ⊕ {p : wᵀp = 0},        w = lumped M_p,

which is where ``K`` is invertible. We build an **orthonormal** basis ``Z`` of ``S`` (a selection
matrix for velocity, a Householder basis of ``w^⊥`` for pressure) and work with
``K_S = ZᵀKZ`` and ``P_S = ZᵀPZ``. Orthonormality is what makes those restrictions
similarity-faithful, so the reported condition numbers are basis-independent.

Everything is float64: at ``65²`` the galerkin conditioning is ``O(10^8)``, and float32
eigenvalues of such a matrix carry no usable smallest eigenvalue at all.

The ``P`` measured is the one training actually uses — ``StokesBlockPreconditioner`` with the same
``--mg_levels`` (4) and ``--schur_omega`` (0.5) as the runs — including the inexactness of the
V-cycle. Spectra of the non-symmetric ``P_S K_S`` are obtained without ever forming a
non-symmetric eigenproblem: with ``P_S = LLᵀ`` (Cholesky), ``P_S K_S`` is similar to
``Lᵀ K_S L``, which is symmetric.

Usage
-----
    python experiments/stokes/conditioning/measure_conditioning.py --grids 9 17 33 65
    python experiments/stokes/conditioning/measure_conditioning.py --grids 129 --skip_naive

Writes ``output/stokes/conditioning/cond_gr<grid>.json`` per grid and prints a LaTeX table.
Cost is dense-eigensolve bound: seconds up to ``33²``, a few minutes at ``65²``, hours at ``129²``.
"""

import argparse
import json
import os
import time

import torch

from tensorpils.meshing import structured_quad9_mesh
from tensorpils.physics import StokesProblem
from tensorpils.preconditioners.stokes import StokesBlockPreconditioner

torch.set_grad_enabled(False)


# ---------------------------------------------------------------- reduced basis
def admissible_basis(problem: StokesProblem) -> torch.Tensor:
    """Orthonormal basis ``Z`` ``[n_dofs, m]`` of the admissible subspace ``S``.

    Velocity: the non-Dirichlet DOFs (columns of the identity). Pressure: an orthonormal basis
    of ``w^⊥`` with ``w`` the lumped pressure mass, via a Householder reflector that maps
    ``w/‖w‖`` to ``e₁`` — its remaining columns span the orthogonal complement.
    """
    n_dofs, off_p, n_p = problem.n_dofs, problem.off_p, problem.n_p
    free_u = (~problem.dirichlet_mask[:off_p]).nonzero(as_tuple=True)[0]

    w = problem.m_p_lumped.double()
    w = w / w.norm()
    e1 = torch.zeros(n_p, dtype=torch.float64)
    e1[0] = 1.0
    v = w - e1
    H = torch.eye(n_p, dtype=torch.float64)
    if v.norm() > 1e-14:                      # Householder H w = e1, H symmetric orthogonal
        v = v / v.norm()
        H = H - 2.0 * torch.outer(v, v)
    Z_p = H[:, 1:]                            # [n_p, n_p-1], orthonormal, all ⟂ w

    m = free_u.numel() + Z_p.shape[1]
    Z = torch.zeros(n_dofs, m, dtype=torch.float64)
    Z[free_u, torch.arange(free_u.numel())] = 1.0
    Z[off_p:, free_u.numel():] = Z_p
    return Z


def dense_preconditioner(precond, n_dofs: int, batch: int = 256) -> torch.Tensor:
    """Materialize ``P`` ``[n_dofs, n_dofs]`` by applying it to the identity, in float64.

    Returns the symmetrized operator and the relative asymmetry, which is a real check on the
    V-cycle: ``P`` is used as a *norm* in the pls loss, so a non-symmetric ``P`` would not even
    define one.
    """
    cols = []
    for lo in range(0, n_dofs, batch):
        hi = min(lo + batch, n_dofs)
        E = torch.zeros(hi - lo, n_dofs, dtype=torch.float64)
        E[torch.arange(hi - lo), torch.arange(lo, hi)] = 1.0
        cols.append(precond(E))               # row j = (P e_j)ᵀ
    P = torch.cat(cols, dim=0)                # = Pᵀ (rows are images) — symmetric up to roundoff
    asym = (P - P.T).abs().max().item() / max(P.abs().max().item(), 1e-300)
    return 0.5 * (P + P.T), asym


# ---------------------------------------------------------------- spectra
def _cond_sym(M: torch.Tensor, indefinite: bool = False):
    """``(kappa, |λ|min, |λ|max)`` of a symmetric matrix via ``eigvalsh``."""
    ev = torch.linalg.eigvalsh(0.5 * (M + M.T))
    a = ev.abs()
    lo, hi = a.min().item(), a.max().item()
    if not indefinite and ev.min().item() <= 0.0:
        print(f"    ! expected SPD but min eigenvalue is {ev.min().item():.3e}")
    return hi / max(lo, 1e-300), lo, hi


def exact_block_spectrum(K_S: torch.Tensor, n_free_u: int) -> dict:
    r"""Validation: the spectrum of :math:`\mathcal P\mathcal K` with the **exact** blocks.

    With :math:`\mathcal P=\operatorname{diag}(A^{-1}, S^{-1})` and :math:`S=BA^{-1}B^\top`, the
    note's elementary computation gives exactly three eigenvalues
    :math:`\{1, (1\pm\sqrt5)/2\}`, so the ratio is :math:`(1+\sqrt5)/(\sqrt5-1)=\varphi^2\approx2.618`.
    Reproducing that to machine precision checks two things at once: the theory claim, and this
    script's reduction to the admissible subspace (a wrong basis would not give three eigenvalues).
    It also separates "the block idea is weak" from "our *inexact* blocks are weak".
    """
    A = K_S[:n_free_u, :n_free_u]
    B = K_S[n_free_u:, :n_free_u]
    A_inv = torch.linalg.inv(A)
    S = B @ A_inv @ B.T
    P = torch.zeros_like(K_S)
    P[:n_free_u, :n_free_u] = A_inv
    P[n_free_u:, n_free_u:] = torch.linalg.inv(S)
    P = 0.5 * (P + P.T)
    L = torch.linalg.cholesky(P)
    ev = torch.linalg.eigvalsh(L.T @ K_S @ L)
    a = ev.abs()
    return {"exact_absmin_PK": a.min().item(), "exact_absmax_PK": a.max().item(),
            "exact_ratio_PK": a.max().item() / max(a.min().item(), 1e-300),
            "exact_distinct": torch.unique(ev.round(decimals=6)).tolist()[:8],
            "exact_cond_P": _cond_sym(P)[0]}


def measure(grid: int, mg_levels: int, schur_omega: float, mu: float,
            skip_naive: bool = False, exact_blocks: bool = False) -> dict:
    n_p = (grid + 1) // 2
    h = 1.0 / (n_p - 1)                       # element size of the pressure (corner) grid
    t0 = time.time()
    print(f"[gr={grid}] mesh + assembly (velocity {grid}^2, pressure {n_p}^2, h={h:.4g})")
    mesh = structured_quad9_mesh(nx=n_p, ny=n_p)
    problem = StokesProblem(mesh, nx_p=n_p, ny_p=n_p, mu=mu)
    precond = StokesBlockPreconditioner(problem, mg_levels=mg_levels,
                                        schur_omega=schur_omega).double()

    Z = admissible_basis(problem)
    n_dofs, m = Z.shape
    print(f"[gr={grid}] n_dofs={n_dofs}  admissible dim m={m}  ({time.time()-t0:.1f}s)")

    K = problem.K.to_dense().double()
    P, asym = dense_preconditioner(precond, n_dofs)
    print(f"[gr={grid}] dense K, P built; P relative asymmetry {asym:.2e}  "
          f"({time.time()-t0:.1f}s)")

    K_S = Z.T @ K @ Z
    P_S = Z.T @ P @ Z
    del K, P
    K_S = 0.5 * (K_S + K_S.T)
    P_S = 0.5 * (P_S + P_S.T)

    out = {"grid": grid, "pgrid": n_p, "h": h, "n_dofs": n_dofs, "m": m,
           "mg_levels": mg_levels, "schur_omega": schur_omega, "mu": mu,
           "P_asymmetry": asym}

    if exact_blocks:
        n_free_u = int((~problem.dirichlet_mask[:problem.off_p]).sum())
        out.update(exact_block_spectrum(K_S, n_free_u))
        print(f"[gr={grid}] exact blocks: |lam(PK)| in "
              f"[{out['exact_absmin_PK']:.6g}, {out['exact_absmax_PK']:.6g}]  "
              f"ratio = {out['exact_ratio_PK']:.6g} (theory 2.618034)  "
              f"distinct = {[round(v, 5) for v in out['exact_distinct']]}")

    # --- kappa(K): the operator itself, symmetric indefinite -> O(h^-2) expected
    out["cond_K"], lo, hi = _cond_sym(K_S, indefinite=True)
    out["absmin_K"], out["absmax_K"] = lo, hi
    # GN of the bare least-squares loss is K², whose conditioning is kappa(K)²
    out["cond_K2"] = out["cond_K"] ** 2
    print(f"[gr={grid}] kappa(K)   = {out['cond_K']:.4e}   -> kappa(K²) = {out['cond_K2']:.4e}"
          f"  ({time.time()-t0:.1f}s)")

    # --- kappa(P): the preconditioner as a norm -> O(h^-2) expected, drives Lemma 1
    out["cond_P"], *_ = _cond_sym(P_S)
    print(f"[gr={grid}] kappa(P)   = {out['cond_P']:.4e}  ({time.time()-t0:.1f}s)")

    # --- spectrum of PK via the similar symmetric matrix Lᵀ K L, P = LLᵀ  -> O(1) expected
    L = torch.linalg.cholesky(P_S)
    ev_pk = torch.linalg.eigvalsh(L.T @ K_S @ L).abs()
    out["absmin_PK"], out["absmax_PK"] = ev_pk.min().item(), ev_pk.max().item()
    out["ratio_PK"] = out["absmax_PK"] / max(out["absmin_PK"], 1e-300)
    q = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    out["quantiles_absPK"] = torch.quantile(ev_pk, q).tolist()
    del L
    print(f"[gr={grid}] |lam(PK)| in [{out['absmin_PK']:.4g}, {out['absmax_PK']:.4g}]  "
          f"ratio = {out['ratio_PK']:.4g}  ({time.time()-t0:.1f}s)")

    # --- GN of the pls loss: K P K, SPD -> O(h^-2) expected
    KP = K_S @ P_S
    out["cond_KPK"], *_ = _cond_sym(KP @ K_S)
    print(f"[gr={grid}] kappa(KPK) = {out['cond_KPK']:.4e}  ({time.time()-t0:.1f}s)")

    # --- GN of the note's dead-end applied form ½‖P r‖²: (PK)ᵀPK = K P² K -> O(h^-4) expected
    if not skip_naive:
        out["cond_naive"], *_ = _cond_sym(KP @ KP.T)
        print(f"[gr={grid}] kappa((PK)ᵀPK) = {out['cond_naive']:.4e}  ({time.time()-t0:.1f}s)")

    out["seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------- reporting
def slope(xs, ys):
    """Least-squares slope of log(y) vs log(x) — the measured exponent in ``kappa ~ (1/h)^s``."""
    import math
    n = len(xs)
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


def latex_table(rows) -> str:
    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Measured conditioning of the Stokes Gauss--Newton matrices, float64, on the "
        r"admissible subspace (interior velocity $\oplus$ zero-mean pressure). $\mathcal P$ is the "
        r"block preconditioner \eqref{eq:stokes_precond} exactly as trained with "
        r"(\texttt{mg\_levels}=4, $\omega=1/2$). The last row is the least-squares fit of the "
        r"exponent $s$ in $\kappa\sim h^{-s}$.}",
        r"\label{tab:stokes_cond}",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"grid & $\kappa(\mathcal K)$ & $\kappa(\mathcal K^2)$ & $\kappa(\mathcal P)$ & "
        r"$\frac{\max|\lambda(\mathcal{PK})|}{\min|\lambda(\mathcal{PK})|}$ & "
        r"$\kappa(\mathcal{KPK})$ & $\kappa((\mathcal{PK})^\top\mathcal{PK})$ \\",
        r"     &                     & $G$ of $L_{\textup{LS}}$ &  & & "
        r"$G$ of $L_{\textup{PLS}}$ & $G$ of $L_{\text{naive}}$ \\",
        r"\midrule",
    ]

    def f(x):
        if x is None:
            return "---"
        e = 0
        while abs(x) >= 10:
            x /= 10.0
            e += 1
        return f"${x:.2f}\\times 10^{{{e}}}$" if e else f"${x:.2f}$"

    for r in rows:
        lines.append(f"${r['grid']}^2$ & {f(r['cond_K'])} & {f(r['cond_K2'])} & "
                     f"{f(r['cond_P'])} & {f(r['ratio_PK'])} & {f(r['cond_KPK'])} & "
                     f"{f(r.get('cond_naive'))} \\\\")
    lines.append(r"\midrule")
    inv_h = [1.0 / r["h"] for r in rows]
    cells = []
    for key in ("cond_K", "cond_K2", "cond_P", "ratio_PK", "cond_KPK", "cond_naive"):
        ys = [r.get(key) for r in rows]
        cells.append("---" if any(y is None for y in ys) or len(rows) < 2
                     else f"${slope(inv_h, ys):.2f}$")
    lines.append("fitted $s$ & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grids", type=int, nargs="+", default=[9, 17, 33, 65],
                    help="Velocity grid resolutions (odd). Dense eigensolves: 129 costs hours.")
    ap.add_argument("--mg_levels", type=int, default=4)
    ap.add_argument("--schur_omega", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--skip_naive", action="store_true",
                    help="Skip kappa((PK)^T PK) (one extra eigensolve).")
    ap.add_argument("--exact_blocks", action="store_true",
                    help="Also compute the spectrum of PK with the EXACT blocks diag(A^-1,S^-1), "
                         "which must reproduce the note's {1,(1+-sqrt5)/2}. Needs a dense inverse.")
    ap.add_argument("--out_dir", default="output/stokes/conditioning")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for grid in sorted(args.grids):
        if grid % 2 == 0:
            raise SystemExit(f"velocity grid must be odd, got {grid}")
        rec = measure(grid, args.mg_levels, args.schur_omega, args.mu, args.skip_naive,
                      args.exact_blocks)
        path = f"{args.out_dir}/cond_gr{grid}.json"
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=2)
        print(f"[gr={grid}] -> {path}\n")
        rows.append(rec)

    print("\n" + latex_table(rows) + "\n")
    with open(f"{args.out_dir}/table.tex", "w") as fh:
        fh.write(latex_table(rows) + "\n")
    print(f"table -> {args.out_dir}/table.tex")


if __name__ == "__main__":
    main()
