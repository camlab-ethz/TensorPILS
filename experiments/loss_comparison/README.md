# Experiment: loss comparison (true-L2 vs MSE vs Deep Ritz)

Compare three training objectives on the same problem, all evaluated by the same
finite-element relative-L2 error on the validation set:

- **`data_l2`** — supervised **true-L2** loss `½‖u−u★‖²_{L²} = ½ eᵀ M e` (mass-weighted). This
  trains in *exactly* the metric used for evaluation.
- **`data`** — supervised **MSE**, a flat average of squared nodal errors on the grid (no
  geometry / mass weighting).
- **`deepritz`** — the Deep Ritz energy, **label-free**, no preconditioner.

The question: does training in the correct L² metric (`data_l2`) beat plain MSE (`data`),
and how do both supervised losses compare to the label-free Deep Ritz energy? MSE and the
true-L2 loss differ by the mass matrix (area weighting + off-diagonal coupling), so they are
genuinely different objectives, not a rescale.

## Facts

| | |
|---|---|
| Losses | `data_l2`, `data`, `deepritz` (Deep Ritz uses default penalty BC, no preconditioner) |
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

All three on Euler (SLURM array over the losses; submit from `~/TensorPILS`):

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
