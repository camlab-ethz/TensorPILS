# Experiments: published physics-informed baselines

**The question.** The paper claims preconditioned physics-informed training "vastly outperforms
previous approaches", but every label-free arm it currently compares against is our own: the bare
FEM least-squares loss (`galerkin`, the `t=0` end of the blend sweep) is a negative control *we*
built. No published method appears in any table. These experiments fix that.

Two baselines, both label-free, both scored by the repo's usual FEM relative-`L2` metric with the
eval-time boundary projection — so their numbers drop straight into the existing tables:

| baseline | architecture | residual | differentiation |
|---|---|---|---|
| **PINO** | FNO | strong form | central finite differences |
| **PI-DeepONet** | DeepONet | strong form | autodiff through the trunk coordinate |
| *(ours)* | FNO | **weak** form (FEM) | assembled `A`, `M` + multigrid `P` |

## Why finite differences for PINO, and not FFT

The reference implementation (`neuraloperator/physics_informed`) uses **both**, and the choice
tracks the boundary condition rather than taste:

| example | BC | differentiation |
|---|---|---|
| Burgers (`FDM_Burgers`) | periodic in `x` | FFT |
| Navier–Stokes (`FDM_NS_vorticity`) | periodic | FFT |
| **Darcy** (`FDM_Darcy`) | **Dirichlet** | **central differences** |

Every PDE in this repo is homogeneous Dirichlet on the unit square, so Darcy is the analogue and FD
is the faithful port. Three further details are copied from `FDM_Darcy` / `train_2d.py`, each of
which *helps* the baseline:

1. residual on **interior points only**;
2. output multiplied by a **mollifier** `sin(pi x) sin(pi y)` — the zero Dirichlet BC imposed hard,
   an advantage our bare `galerkin` arm does not have;
3. reduction is the **relative** `Lp` ratio `||Lu - f|| / ||f||`, not our sum-over-nodes convention.

The Laplacian itself is `neuralop.losses.differentiation.FiniteDiff` — the reference library's own
utility — so "the baseline was coded wrong" is not available as a reading of any result.

## The comparison is at fixed architecture

Each block below varies only the loss, so the loss effect is not confounded with the architecture:

| arch | loss | labels | derivative |
|---|---|---|---|
| FNO | `data` | yes | — |
| FNO | `galerkin` (bare FEM LS) | no | FEM |
| FNO | **`pino`** | no | FD |
| FNO | `pls` (ours: FEM + multigrid `P`) | no | FEM |
| DeepONet | `data` | yes | — |
| DeepONet | **`pideeponet`** | no | autodiff |
| DeepONet | `pls` (ours) | no | FEM |

The supervised DeepONet row is **not optional**: without it, a poor PI-DeepONet number cannot be
told apart from "a DeepONet cannot fit this problem at all". The `pls`-on-DeepONet row is nearly
free (the model is a drop-in for the FNO) and is the architecture-agnostic evidence for the
paper's second contribution.

## Load-bearing check

`tests/test_baselines.py::test_fd_and_fem_residuals_have_the_same_discretization_error` —
measured on the analytical Poisson pair:

| grid | `-lap_FD(u*)` vs `f` | `A u*` vs `M f` |
|---|---|---|
| `33²` | `1.049e-2` | `1.056e-2` |
| `65²` | `2.566e-3` | `2.570e-3` |

The two residuals are the same PDE at the same `O(h²)` accuracy (they agree to 0.7 %, and both
converge at rate 2.03). This is what licenses reading any training difference as a property of the
loss rather than of the discretisation.

## Fairness protocol

Every **new** arm gets its own learning-rate sweep `{3e-4, 1e-3, 3e-3, 1e-2, 3e-2}` at a reduced
budget, then the best `lr` is run at the full budget with **3 seeds**. Existing arms keep their
published settings. This is not a formality: PINO's relative-`Lp` reduction alone shifts the useful
learning rate by orders of magnitude relative to our sum-over-nodes losses, so a shared `lr` would
silently handicap one side.

Two initialisation details were measured and fixed for the same reason — see
`tensorpils/baselines/deeponet.py`: the DeepONet output is scaled by `1/sqrt(p)` (without it the
initial output is `5.5x` the target at `p=32` and `12.4x` at `p=128`), and the trunk gets random
Fourier features (a plain MLP trunk underfits multi-frequency data for reasons unrelated to the
objective).

## Sub-experiments

| Experiment | Question |
|---|---|
| [`poisson/`](poisson/README.md) | The headline table: PINO and PI-DeepONet against supervised and preconditioned FEM training at `64²`, `K=4`. |
| [`allen_cahn/`](allen_cahn/README.md) | The same comparison for a nonlinear, time-dependent problem, as an autoregressive one-step stepper. |
| [`stokes/`](stokes/README.md) | The strong-form port of both baselines to the saddle point (stacked momentum + continuity residual, velocity-only BC, full-grid pressure), with tests pinning the FD and autodiff operators. The runs are in [`experiments/stokes/stokes_paper/stokes_benchmark/`](../stokes/stokes_paper/stokes_benchmark/README.md) (the Table 1 rows). |

## Scope

- **Stokes now has both baseline arms** — strong-form momentum + continuity on the velocity grid,
  stacked in one relative residual with a tunable continuity weight (`--pi_div_weight`), pressure
  taken on the full fine grid, BC on the velocity pair only. See [`stokes/`](stokes/README.md) for
  the port and its three design decisions; the Table 1 runs are in
  `experiments/stokes/stokes_paper/stokes_benchmark/`.
- **Collocation points are the grid nodes**, optionally subsampled (`--pi_n_colloc`). This matches
  every other arm's discretisation budget, and for Allen–Cahn it is forced: the reference comes
  from a FEM Newton solve and exists only at nodes, so off-node collocation would require
  interpolating the reference and would measure that interpolation.
- This branch is **not merged to `main`** — it is a baseline study, not a feature.
