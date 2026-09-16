#!/usr/bin/env python
r"""Operator-level comparison of the geometric and algebraic V-cycles, before any training.

Everything the training result depends on is a property of ``P`` alone, so it can be measured in
seconds and used as a *prediction*. Four quantities per grid:

* **symmetry defect** --- ``max |<Px,y> - <x,Py>| / |<Px,y>|``. Not a curiosity: the AMG
  preconditioner's backward pass *is* a forward cycle, which is only the vjp while ``P`` is
  symmetric. A large value here means every gradient computed with that ``P`` is wrong.
* **contraction** --- ``||r - A P r|| / ||r||`` for one cycle. The classical multigrid health
  check, and the thing that must stay flat under refinement.
* **kappa(PA)** --- and with it ``kappa(H) = kappa(PA)^2``, the condition number of the ``pls``
  loss Hessian, which is the x-axis of the paper's collapse figure. Dense, so it is computed only
  where that is cheap; the contraction factor covers the larger grids.
* **cost** --- ms per batch of 32 applies. AmgX takes one right-hand side at a time, so a batch is
  a Python loop and the cost is dominated by launch overhead rather than by ``N``.

Only one AmgX solver may be alive at a time (two abort the process at exit), so the AMG
preconditioner is built and dropped per grid.

Usage
-----
    python experiments/poisson/amg_dropin/measure_precond.py --grids 33 64 65 129
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

from tensorpils.meshing import structured_quad_mesh
from tensorpils.physics import PoissonProblem
from tensorpils.preconditioners import GeometricMultigrid, build_preconditioner

# On a small enough problem AMG coarsens straight down to a direct solve, which reports a perfect
# kappa(PA) = 1 and a zero contraction -- a degenerate operator, not a good preconditioner. The
# check below is on the measured symptom rather than a guessed DOF count, because the crossover
# depends on the coarsening: measured here, 17^2 (225 interior DOFs) is degenerate while 33^2
# (961) is not.
DEGENERATE_KAPPA = 1.02
DEGENERATE_CONTRACTION = 1e-3


def apply_batched(precond, V, chunk_is_loop):
    return (torch.stack([precond(V[i]) for i in range(V.shape[0])])
            if chunk_is_loop else precond(V))


@torch.no_grad()
def symmetry_defect(apply, N, device, k=8, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(k, N, generator=g).to(device)
    y = torch.randn(k, N, generator=g).to(device)
    a = (apply(x) * y).sum(-1)
    b = (x * apply(y)).sum(-1)
    return float(((a - b).abs() / a.abs().clamp_min(1e-30)).max())


@torch.no_grad()
def contraction(apply, matvec, N, mask, device, k=16, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(k, N, generator=g).to(device)
    r[:, mask] = 0
    # Mask the boundary: `matvec` is the RAW assembled A, whose boundary rows couple to the
    # interior, so A(Pr) carries entries there that the BC-eliminated operator the preconditioner
    # actually inverts does not. Leaving them in inflates the ratio several-fold.
    res = r - matvec(apply(r))
    res[:, mask] = 0
    return float((res.norm(dim=1) / r.norm(dim=1)).mean())


@torch.no_grad()
def _interior_dense_A(matvec, N, mask, device, chunk=512):
    """Dense interior block of ``A`` in float64 on the device."""
    idx = (~mask).nonzero().squeeze(1)
    ni = len(idx)
    cols = []
    for lo in range(0, ni, chunk):
        hi = min(lo + chunk, ni)
        E = torch.zeros(hi - lo, N, device=device)
        E[torch.arange(hi - lo, device=device), idx[lo:hi]] = 1.0
        cols.append(matvec(E)[:, idx].double())
    return torch.cat(cols, 0).T.contiguous(), idx      # [ni, ni] (column j = A e_j)


@torch.no_grad()
def cond_PA(apply, matvec, N, mask, device, chunk=512):
    r"""``kappa(PA)`` for SPD ``A`` and SPD ``P``, via a symmetric similarity.

    ``PA`` is not symmetric, but with the Cholesky factor ``A = L L^T`` the matrix
    ``S = L^T P L`` *is*, and ``eig(S) = eig(P L L^T) = eig(PA)``. So the spectrum comes from
    ``eigvalsh`` rather than a general ``eigvals``: real by construction, better conditioned,
    and roughly an order of magnitude faster at a few thousand DOFs -- which is the difference
    between this measurement taking seconds and taking tens of minutes.
    """
    A_ii, idx = _interior_dense_A(matvec, N, mask, device, chunk)
    ni = len(idx)
    L = torch.linalg.cholesky(0.5 * (A_ii + A_ii.T))
    PL = torch.empty_like(L)
    for lo in range(0, ni, chunk):                      # apply P to the columns of L
        hi = min(lo + chunk, ni)
        E = torch.zeros(hi - lo, N, device=device, dtype=torch.float64)
        E[:, idx] = L[:, lo:hi].T
        PL[:, lo:hi] = apply(E.float()).double()[:, idx].T
    S = L.T @ PL
    ev = torch.linalg.eigvalsh(0.5 * (S + S.T))
    ev = ev[ev > 1e-12 * ev.max()]
    return float(ev.max() / ev.min())


@torch.no_grad()
def cond_A(matvec, N, mask, device, chunk=512):
    A_ii, _ = _interior_dense_A(matvec, N, mask, device, chunk)
    ev = torch.linalg.eigvalsh(0.5 * (A_ii + A_ii.T))
    return float(ev.max() / ev.min())


def timed(fn, n_rep=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_rep * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grids", type=int, nargs="+", default=[33, 64, 65, 129])
    ap.add_argument("--sweeps", type=int, default=2, help="pre = post smoothing sweeps (both)")
    ap.add_argument("--mg_levels", type=int, default=4)
    ap.add_argument("--mg_omega", type=float, default=2.0 / 3.0)
    ap.add_argument("--cond_max_dofs", type=int, default=5000,
                    help="skip the dense kappa(PA) above this many interior DOFs")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--precond", choices=["both", "gmg", "amg"], default="both",
                    help="which preconditioners to measure. The geometric V-cycle densifies its "
                         "level operators (multigrid.py: A.to_dense()), so it cannot be built "
                         "much past 129^2 -- 32.5 GiB for the fine level alone at 257^2. Use "
                         "'amg' to characterise the algebraic cycle beyond that.")
    ap.add_argument("--out", type=str,
                    default="output/poisson/amg_dropin/precond_comparison.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("AmgX is CUDA-only; this measurement needs a GPU.")
    device = "cuda"
    rows = []

    hdr = (f"{'grid':>6} {'N':>7} {'precond':>5} {'sym defect':>11} {'contract':>9} "
           f"{'kappa(PA)':>10} {'kappa(H)':>10} {'ms/batch':>9}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for n in args.grids:
        prob = PoissonProblem(structured_quad_mesh(nx=n, ny=n)).to(device)
        N = n * n
        mask = prob.boundary_mask.to(device)
        matvec = lambda V: prob._spmm(prob.A, V)                      # noqa: E731
        ni = int((~mask).sum())
        do_cond = ni <= args.cond_max_dofs

        V = torch.randn(args.batch, N, device=device)
        V[:, mask] = 0

        # --- geometric ------------------------------------------------------
        if args.precond in ("both", "gmg"):
          gmg = GeometricMultigrid(nx_fine=n, ny_fine=n, n_levels=args.mg_levels,
                                 pre_smooth=args.sweeps, post_smooth=args.sweeps,
                                 omega=args.mg_omega).to(device)
          rec = dict(grid=n, n_dofs=N, n_interior=ni, precond="gmg",
                     sym=symmetry_defect(gmg, N, device),
                     contraction=contraction(gmg, matvec, N, mask, device),
                     kappa_PA=cond_PA(gmg, matvec, N, mask, device) if do_cond else None,
                     ms_per_batch=timed(lambda: gmg(V)))
          rec["kappa_H"] = rec["kappa_PA"] ** 2 if rec["kappa_PA"] else None
          rec["kappa_A"] = cond_A(matvec, N, mask, device) if do_cond else None
          rows.append(rec)
          _print(rec)
          del gmg
          gc.collect()
          torch.cuda.empty_cache()

        if args.precond == "gmg":
            continue

        # --- algebraic ------------------------------------------------------
        amg = build_preconditioner(kind="amg", problem=prob, grid_size=(n, n),
                                   amg_sweeps=args.sweeps, device=device)
        loop = lambda W: torch.stack([amg(W[i]) for i in range(W.shape[0])])   # noqa: E731
        rec = dict(grid=n, n_dofs=N, n_interior=ni, precond="amg",
                   sym=symmetry_defect(loop, N, device),
                   contraction=contraction(loop, matvec, N, mask, device),
                   kappa_PA=cond_PA(loop, matvec, N, mask, device) if do_cond else None,
                   ms_per_batch=timed(lambda: loop(V)))
        rec["kappa_H"] = rec["kappa_PA"] ** 2 if rec["kappa_PA"] else None
        rec["kappa_A"] = rows[-1]["kappa_A"] if rows else None
        if (rec["contraction"] < DEGENERATE_CONTRACTION
                and (rec["kappa_PA"] or 1.0) < DEGENERATE_KAPPA):
            rec["warning"] = (f"contraction {rec['contraction']:.1e} and kappa(PA) "
                              f"{rec['kappa_PA']} on {ni} interior DOFs: AMG has coarsened this "
                              f"straight to a direct solve, so these numbers describe a "
                              f"degenerate operator, not preconditioner quality. Use a finer "
                              f"grid to characterise the preconditioner.")
        rows.append(rec)
        _print(rec)
        # One live AmgX solver at a time: two abort the process at exit.
        del amg, loop
        gc.collect()
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"sweeps": args.sweeps, "mg_levels": args.mg_levels,
                   "mg_omega": args.mg_omega, "rows": rows}, fh, indent=2)
    print(f"\nwritten -> {args.out}")
    for r in rows:
        if "warning" in r:
            print(f"[warning] grid {r['grid']}: {r['warning']}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)      # AmgX teardown is fragile; skip interpreter cleanup (flush first)


def _print(r):
    def f(x, spec):
        return format(x, spec) if x is not None else "-".rjust(int(spec.split('.')[0].strip('>')))
    print(f"{r['grid']:>6} {r['n_dofs']:>7} {r['precond']:>5} {r['sym']:>11.2e} "
          f"{r['contraction']:>9.4f} {f(r['kappa_PA'], '>10.3f')} {f(r['kappa_H'], '>10.2f')} "
          f"{r['ms_per_batch']:>9.2f}", flush=True)


if __name__ == "__main__":
    main()
