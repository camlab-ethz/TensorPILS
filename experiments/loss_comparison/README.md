# Experiment: loss comparison (supervised metrics vs Deep Ritz)

Compare several training objectives on the same problem, all evaluated by the same
finite-element relative-L2 error on the validation set:

- **`data`** — supervised **MSE**, a flat average of squared nodal errors on the grid (no
  geometry / mass weighting).
- **`data_l2`** — supervised **true-L2** loss `½‖u−u★‖²_{L²} = ½ eᵀ M e` (mass-weighted). This
  trains in *exactly* the metric used for evaluation.
- **`data_h1`** — supervised **H¹₀ norm** loss `½‖u−u★‖²_{H¹₀} = ½ eᵀ A e` (stiffness-weighted;
  same as `data_l2` with `A` in place of `M`). The prediction is projected to zero on the
  boundary first, so the error lies in `H₀¹` where the seminorm is a genuine norm (otherwise
  the constant/boundary mode is unconstrained); the boundary is likewise projected at eval.
- **`deepritz` (penalty BC)** — Deep Ritz energy, **label-free**, soft boundary penalty
  (`λ_bc=100`), no preconditioner.
- **`deepritz` (hard BC)** — Deep Ritz energy, **label-free**, hard boundary projection, no
  preconditioner.

The questions: does training in the correct L² metric (`data_l2`) beat plain MSE (`data`)?
How does the H¹₀ metric (`data_h1`) compare? And how do the label-free Deep Ritz variants
compare to the supervised losses — including the effect of **hard vs penalty boundary
conditions** (hard BC is the fair analogue of the `t=0` preconditioned Deep Ritz)? MSE, L²,
and H¹₀ differ by their weighting matrix (identity, mass, stiffness), so they are genuinely
different objectives.

## Facts

| | |
|---|---|
| Losses | `data`, `data_l2`, `data_h1`, `deepritz` (penalty BC), `deepritz --bc_mode hard` (all Deep Ritz without preconditioner) |
| Dataset | 2D Poisson, `K=4`, grid `64²`, `n_train=1024`, `n_val=128`, `n_test=256`, `seed=42` |
| Optimizer | `adam`, default cosine lr `1e-3 → 1e-4` (all defaults) |
| Epochs / batch | `1000` / `32` |
| FNO modes | `16×16` |
| Output dir | `output/loss_comparison/` (git-ignored; files keyed by loss tag) |
| Recorded | per-epoch validation relative-L2 (`stats.val_rel_l2_errors`) |

## Run

Single loss:

```bash
python -m tensorpils.cli --loss data_l2 --n_train 1024 --n_val 128 --n_test 256 \
    -k 4 --epochs 1000 --batch_size 32 --device cuda --output_dir output/loss_comparison
```

All five on Euler (SLURM array over the losses; submit from `~/TensorPILS`):

```bash
mkdir -p logs
sbatch experiments/loss_comparison/sweep.sbatch
```

## Plot

```bash
python experiments/loss_comparison/plot_loss_comparison.py \
    --results_dir output/loss_comparison/results
```

Produces `output/loss_comparison/loss_comparison.png` — validation relative-L2 vs epoch,
one curve per loss.
