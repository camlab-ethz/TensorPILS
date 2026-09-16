# Experiment: swapping the geometric V-cycle for an algebraic one

**The question.** The method's preconditioner is currently a *geometric* multigrid V-cycle, and
that is the one component of the whole pipeline that cannot leave a structured grid: it
re-discretizes every level with `structured_quad_mesh`, and its prolongation
(`_build_2d_prolongation`) is bilinear interpolation that assumes the row-major node order.
An *algebraic* V-cycle builds its hierarchy from the matrix alone, so it carries over to an
unstructured mesh unchanged. Before trusting it there, we check it here: **on the same
structured grid, with the same operator, the same data and the same budget, does `P_AMG` train
as well as `P_GMG`?**

This is deliberately a *controlled* swap. Only the preconditioner changes, so any difference is
attributable to AMG-instead-of-GMG and not to unstructured-instead-of-structured. Doing both at
once would make a negative result uninterpretable.

## Why the reference arms are free

The comparison arms already exist in this checkout, from
[`experiments/baselines/poisson`](../../baselines/poisson/README.md) — same grid, same data,
same budget, three seeds:

| arm | test rel-L² (mean over seeds 42/43/44) |
|---|---|
| `data` — supervised gold standard | **0.59 %** (0.54–0.65) |
| `pls` + **GMG** — what we are replacing | **0.66 %** (0.61–0.72) |
| `galerkin` — bare residual, negative control | **29.13 %** (27.5–32.0) |

A 44× gap between the preconditioned and unpreconditioned arms makes this an unambiguous
pass/fail: landing near 0.7 % is success, landing near 29 % means the preconditioner is not
being applied — or its *gradient* is not (see "the trap" below).

## Facts

| | |
|---|---|
| Arm | `--loss pls --precond_kind amg`, everything else byte-identical to the `pls` row of `experiments/baselines/poisson/sweep.sbatch` |
| Grid | `64²`, `Q1`, zero Dirichlet |
| Dataset | `K=4` multi-frequency, `n_train/val/test = 1024/128/256` |
| Budget | 500 epochs, batch 32, Adam, `lr 1e-3 → 1e-6` cosine, seeds 42/43/44 |
| Preconditioner | one AmgX V(2,2) cycle, `CLASSICAL` coarsening, `BLOCK_JACOBI` smoother, relaxation 0.8 |
| Learning rate | **not re-tuned** — reused from the GMG arm on purpose, see below |
| Output | `output/poisson/amg_dropin/seed<N>/` (git-ignored) |

**The learning rate is reused, not re-tuned, and that is a claim rather than laziness.** The
paper's collapse figure (`experiments/poisson/sweep_blend`) says the final error is a function of
the loss Hessian's condition number `κ(H) = κ(PA)²`, with diminishing returns below `κ(H) ≈ 10³`.
Measured here (fp32, V(2,2)) at `65²`: `κ(PA) = 1.18` for GMG and `2.47` for AMG, i.e.
`κ(H) = 1.4` and `6.1`. **Both are three orders of magnitude inside the flat part of that curve**,
so the collapse curve *predicts* the two should train alike at the same rate. Reusing the rate is
what makes that prediction falsifiable; if AMG needs its own rate, the prediction was wrong and
that is the more interesting outcome. Run `measure_precond.py` for the conditioning table.

## The trap this experiment exists to catch

`torch-amgx` is a binding with **no autograd**, so `AMGXPreconditioner` supplies the vjp itself.
It does so by exploiting that a V-cycle with a symmetric smoother and **equal pre/post sweeps** is
a symmetric operator, hence `Pᵀ = P` and the backward pass is another forward cycle. That identity
is load-bearing and it is fragile: at `presweeps != postsweeps` the measured asymmetry jumps from
`1e-16` to `1e-1`, and a wrong vjp does not raise — it silently degrades training and would read
as "AMG is worse than GMG". The constructor therefore refuses mismatched sweeps, and
`tests/test_amg_precond.py` checks both the symmetry and the gradient of the assembled `pls` loss
against its closed form `(1/B)·Mask·A·P·P·r`.

Two more AmgX-specific hazards are handled in `preconditioners/algebraic.py` and documented there:
`tolerance` must be 0 (the default `1e-8` ABSOLUTE makes a *successfully converging* model get
`Pr = 0` and a vanishing gradient), and `store_res_history` must stay 1 (the binding reads the
iteration residual unconditionally and throws a 3.5 KB stack trace per apply without it).

## Run

```bash
# operator-level comparison first -- seconds, and it predicts the training result
python experiments/poisson/amg_dropin/measure_precond.py --grids 33 64 65 129

# the drop-in arm, 3 seeds
bash experiments/poisson/amg_dropin/run_local.sh          # inside an allocation: 3 concurrent
clsubmit experiments/poisson/amg_dropin/sweep.txt         # from a login node: one job per seed

# read it against the existing GMG/data/galerkin arms
python experiments/poisson/amg_dropin/compare.py
```

## Result (2026-09-16)

**The swap is clean: the two preconditioners are indistinguishable within the seed noise.**

| arm | test rel-L² (seeds 42/43/44) | range |
|---|---|---|
| `data` — supervised gold standard | 0.59 % | 0.54–0.65 |
| `pls` + **GMG** | **0.66 %** | 0.61–0.72 |
| `pls` + **AMG** | **0.70 %** | 0.65–0.74 |
| `galerkin` — bare residual | 29.13 % | 27.5–32.0 |

Ratio 1.06x, against a geometric-arm seed spread of 17 % of its own mean, with overlapping
ranges. **The learning rate was not re-tuned**, and the prediction stated above — that it would
not need to be, because `κ(H)` is 6.1 against 1.4 and both sit deep in the flat part of the
collapse curve — held. That is the collapse figure used as a predictor rather than as an
explanation.

Operator-level comparison behind that prediction (fp32, V(2,2), one RTX 4090, uncontended):

| grid | contraction GMG / AMG | κ(PA) GMG / AMG | κ(H) GMG / AMG | κ(A) | ms per batch of 32, GMG / AMG |
|---|---|---|---|---|---|
| 33² | 0.029 / 0.060 | 1.17 / 1.63 | 1.4 / 2.7 | 207 | 3.9 / 7.2 |
| 64² | 0.049 / 0.075 | 1.31 / 2.38 | 1.7 / 5.7 | 795 | 4.1 / 8.0 |
| 65² | 0.030 / 0.074 | 1.18 / 2.47 | 1.4 / 6.1 | 830 | 4.1 / 8.3 |
| 129² | 0.031 / 0.074 | — | — | — | 4.5 / 9.3 |

AMG costs ~2x per apply and its contraction constant is ~2.4x worse, both of which are the price
of not being told the grid. Neither matters here, because `κ(H)` is what the error depends on and
6.1 is as good as 1.4.

### The secondary finding: only the algebraic cycle scales

`GeometricMultigrid` densifies its level operators (`multigrid.py:117`, `A.to_dense()`), so the
fine level alone needs 2.1 GiB at 129², **32.5 GiB at 257² and 516 GiB at 513²** — it cannot be
*built* past ~129². The algebraic path hands AmgX a sparse CSR and has no such wall:

| grid | DOFs | contraction | symmetry defect | ms per batch of 32 |
|---|---|---|---|---|
| 129² | 16 641 | 0.0735 | 4.5e-07 | 9.7 |
| 257² | 66 049 | 0.0790 | 2.1e-06 | 13.6 |
| 513² | 263 169 | 0.0812 | 6.6e-07 | 19.5 |

Flat contraction across a 16x increase in DOFs — genuine mesh independence — for 2x the time.
So the algebraic cycle is not only the route to unstructured meshes, it is also the only one of
the two that survives refinement. (`measure_precond.py --precond amg` skips the geometric arm,
which is what makes those rows measurable at all.)

## Reading the result

Check the **training curve**, not only the final number: the bare `galerkin` arm is known to still
be descending at 500 epochs, so "did not converge" and "converged somewhere worse" are different
claims and only the curve separates them. If AMG lands between 0.6 % and 0.8 % the swap is clean
and the unstructured step is unblocked. If it lands materially worse while `κ(PA)` is still O(1),
suspect the gradient before suspecting the preconditioner.

## Environment

AmgX is **CUDA-only**. Running through the project env file needs one line appended to
`~/cluster-kit/env/projects/TensorPILS.euler.sh`, because `module load gcc/8.5.0` puts a
libstdc++ with `GLIBCXX ≤ 3.4.25` ahead of the system one while the bundled `libamgxsh.so`
needs `3.4.29`:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
```

Without it the failure is **silent, not loud**: `import torch_amgx` still succeeds, because the
native extension is loaded lazily and the error is swallowed into `is_available() -> False`.
Measured here, that is exactly what happens — CUDA is fine, and only `from torch_amgx import _C`
reveals `GLIBCXX_3.4.29 not found`. `AMGXPreconditioner` therefore checks `is_available()` at
construction and re-raises the real cause. Check it reaches the compute nodes before spending a
sweep:

```bash
clrun -p gpu1 -t 5 -- python -c "import torch_amgx; print(torch_amgx.is_available())"   # must print True
```

Also: **two live AmgX solver objects abort the process at exit** (exit code 134, which Slurm
reports as FAILED even though every number is correct). One preconditioner per run is the safe
case and is what the CLI does; do not build a second solver, and do not call
`torch_sla.solve(backend="amgx")` in the same process.
