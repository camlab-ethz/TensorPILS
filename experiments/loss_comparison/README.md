# Experiment: loss comparison (supervised metrics vs Deep Ritz)

Compare several training objectives on the same problem, all evaluated by the same
finite-element relative-L2 error on the validation set. **Eval enforces the known zero
Dirichlet BC for every loss** (the prediction's boundary is projected to zero before scoring;
see `Trainer.eval_project_bc`), so the comparison reflects the losses' *interior* behaviour
and no curve is penalised for a boundary the BC already fixes.

- **`data`** — supervised **MSE**, a flat average of squared nodal errors on the grid (no
  geometry / mass weighting).
- **`data_l2`** — supervised **true-L2** loss `½‖u−u★‖²_{L²} = ½ eᵀ M e` (mass-weighted). This
  trains in *exactly* the metric used for evaluation.
- **`data_h1`** — supervised **H¹₀ norm** loss `½‖u−u★‖²_{H¹₀} = ½ eᵀ A e` (stiffness-weighted;
  same as `data_l2` with `A` in place of `M`). The prediction is projected to zero on the
  boundary inside the loss, so the error lies in `H₀¹` where the seminorm is a genuine norm.
  *[plot flag `h1`]*
- **`deepritz` (penalty BC)** — Deep Ritz energy, **label-free**, soft boundary penalty
  (`λ_bc=100`) *during training*, no preconditioner.  *[plot flag `dr_penalty`]*
- **`deepritz` (hard BC)** — Deep Ritz energy, **label-free**, hard boundary projection during
  training, no preconditioner. Mathematically the `t=0` preconditioned Deep Ritz.  *[flag `dr_t0`]*
- **preconditioned Deep Ritz, `t=1`** — Deep Ritz gradient preconditioned by the convex
  blend `P=(1−t)I+tA⁻¹` at `t=1` (`P=A⁻¹`).  *[flag `dr_t1`]*
- **preconditioned least-squares (PLS), `t=1`** — `½‖P(Au−b)‖²` with the blend at `t=1`
  (`P=A⁻¹`, the supervised anchor).  *[flag `pls_t1`]*
- **preconditioned least-squares (PLS), `t=0.5`** — same, blend at `t=0.5`.  *[flag `pls_t0.5`]*

The data losses (`data`/`data_l2`/`data_h1`, plot flags `mse`/`l2`/`h1`) differ by their
weighting matrix (identity / mass / stiffness). The questions: does training in the correct
L² metric beat plain MSE? How does H¹₀ compare? How do the label-free Deep Ritz variants
compare to the supervised losses? And where do the preconditioned physics losses (PLS / Deep
Ritz) at `t=0.5, 1` land relative to all of these?

Because eval now enforces the BC uniformly, the `dr_penalty` vs `dr_t0` difference is purely
the *training* boundary treatment (soft penalty vs hard projection), and `mse` is expected to
track `dr_t1` closely — both are identity-metric supervised objectives toward (nearly) the
same target, once the boundary artifact is removed from the score.

## Facts

| | |
|---|---|
| Losses (8) | `data`, `data_l2`, `data_h1`, `deepritz` (penalty), `deepritz --bc_mode hard`, `deepritz --precondition` blend `t=1`, `pls` blend `t=1`, `pls` blend `t=0.5` |
| Dataset | 2D Poisson, `K=4`, grid `64²`, `n_train=1024`, `n_val=128`, `n_test=256`, `seed=42` |
| Optimizer | `adam`, default cosine lr `1e-3 → 1e-4` (all defaults) |
| Epochs / batch | `1000` / `32` |
| FNO modes | `16×16` |
| Output dir | `output/loss_comparison/` (git-ignored; files keyed by loss tag) |
| Recorded | per-epoch validation relative-L2 (`stats.val_rel_l2_errors`), BC enforced at eval |

## Run

Single loss:

```bash
python -m tensorpils.cli --loss data_l2 --n_train 1024 --n_val 128 --n_test 256 \
    -k 4 --epochs 1000 --batch_size 32 --device cuda --output_dir output/loss_comparison
```

All eight on Euler (SLURM array over the configs; submit from `~/TensorPILS`):

```bash
mkdir -p logs
sbatch experiments/loss_comparison/sweep.sbatch      # array 0-7
```

## Plot

```bash
# all present curves:
python experiments/loss_comparison/plot_loss_comparison.py \
    --results_dir output/loss_comparison/results
```

By default every present curve is drawn. To declutter, pass per-curve flags — only the
listed ones are shown:

```bash
python experiments/loss_comparison/plot_loss_comparison.py \
    --results_dir output/loss_comparison/results --mse --h1 --pls_t1 --pls_t0.5
```

Flags: `--mse --l2 --h1 --dr_penalty --dr_t0 --dr_t1 --pls_t1 --pls_t0.5`.
Produces `output/loss_comparison/loss_comparison.png` — validation relative-L2 vs epoch.
