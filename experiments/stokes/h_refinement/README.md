# Experiment: Stokes mesh refinement (does the preconditioned loss survive `h → 0`?)

The theory claim for the Stokes saddle point is a statement about **mesh dependence**, not about
one grid: the bare least-squares loss has Gauss--Newton conditioning `κ(K²) = O(h⁻⁴)` and the
norm-weighted preconditioned loss `κ(KPK) = O(h⁻²)` (paper Lemma 1). A single-resolution
head-to-head cannot distinguish "the preconditioner helps" from "the preconditioner helps *here*",
so this sweep refines the mesh with **everything else held fixed** and asks whether the gap widens.

| `--loss` | objective | labels |
|---|---|---|
| `data` | supervised FE-norm error vs the discrete Taylor-Hood solution, each field normalised by its own scale | yes |
| `galerkin` | bare least squares `½‖Kc−b‖²` — the negative control | no |
| `pls` | `½ rᵀPr` with `P = diag(mg/μ, ω·μ/diag(M_p))` used as a **norm weight** | no |

The companion measurement of the actual condition numbers is
[`../conditioning/`](../conditioning/README.md) — that is where the `O(h⁻⁴)` / `O(h⁻²)` exponents
are verified numerically. This experiment is the *learning* side of the same claim.

The `65²` column reproduces the headline head-to-head reported in the paper
(`\Cref{tab:stokes_head_to_head}`) with identical settings, so it doubles as a reproducibility
check on those numbers.

## Facts

| | |
|---|---|
| Losses (3) | `pls`, `galerkin`, `data` |
| Velocity grids (3) | `33²`, `65²`, `129²` (pressure `17²`, `33²`, `65²`; DOFs 2467 / 9539 / 37507) |
| Dataset | Stokes, `μ=1`, force decay `r=−0.5`, `K=4`, `n_train=512`, `n_val=64`, `n_test=128`, `seed=42` |
| Reference | discrete Taylor-Hood solve of `K c = (M_u f, 0)`, batched sparse float64 |
| Optimizer | `adam`, cosine `lr 1e-3 → 1e-6` |
| Epochs / batch | `300` / `32` |
| FNO | modes `16×16`, hidden `64`, `5` layers, `2 → 3` channels (3,008,803 params) — **identical at every grid** |
| Preconditioner | `mg_levels=4`, pre/post `2/2`, `schur_omega=0.5` |
| Output dir | `output/stokes/h_refinement/` (git-ignored; prefix carries grid + loss) |
| Recorded | per-epoch `stats.val_rel_l2_u` / `val_rel_l2_p`, test `test_rel_l2_u` / `test_rel_l2_p` |

Note that the FNO is *not* re-tuned per grid — that is deliberate. The network parameterisation is
resolution-independent, so any systematic trend across the three columns is attributable to the
loss geometry rather than to model capacity.

## Run

One cell:

```bash
python -m tensorpils.cli --pde stokes --loss pls --grid_resolution 65 \
    --n_train 512 --n_val 64 --n_test 128 -k 4 --epochs 300 --batch_size 32 --seed 42 \
    --device cuda --output_dir output/stokes/h_refinement
```

All nine on Euler (submit from `~/TensorPILS`; on a `eu-a*`/`eu-g*` compute node run the loop
directly instead):

```bash
mkdir -p logs
sbatch experiments/stokes/h_refinement/sweep.sbatch      # array 0-8
```

`129²` needs `--stokes_ref_chunk 32` (the sbatch sets it) — the batched reference solve is the
memory peak, not training.

## Plot

```bash
python experiments/stokes/h_refinement/plot_h_refinement.py \
    --results_dir output/stokes/h_refinement/results
```

Produces, in `output/stokes/h_refinement/`:

- `h_refinement.png` — test relative FE-`L²` vs grid, velocity and pressure panels, one line per
  loss, with the `100%` "predict zero" level marked;
- `h_refinement_curves.png` — the validation curves behind those numbers, one panel per grid. Worth
  looking at before quoting a `galerkin` number: that arm never converges, so its final error is a
  sample from a wandering curve rather than a converged value, and it is **not** monotone in `h`;
- `table.tex` — the LaTeX table used in the paper.
