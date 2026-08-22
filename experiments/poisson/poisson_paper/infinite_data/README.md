# Dataset-size scaling: preconditioned LS vs supervised training

**How much labelled data does supervised training need to match label-free preconditioned
training?** Every run gets the same **32,000 optimizer steps**, so the x-axis is dataset size and
nothing else.

A new edition of `experiments/poisson/data_scaling`, differing in four ways: `65²` instead of `64²`
(nested dyadic, so the multigrid hierarchy is exact), `K=10` instead of `K=4`, the correct Jacobi
damping `ω = 8/9` instead of the `2/3` default, and a supervised arm to compare against — the
original swept PLS alone.

## The runs — 15 tasks

| tasks | what |
|---|---|
| 0–13 | seven sizes (64 … 4096) × two arms (`data`, `pls`) |
| 14 | PLS on an infinite stream — fresh samples every step, none seen twice |

| | |
|---|---|
| grid / dataset | `65²`, `K=10`, `n_val/n_test = 128/256`, `--fixed_eval` (all runs share eval sets) |
| labels | **`--dataset_solution fem`** — the $Q_1$ FEM solution of $Au = Mf$, for train, val *and* test |
| budget | 1000 virtual epochs × 32 steps = **32,000 steps**, batch 32 |
| early stopping | **off** — every run spends the full budget |
| seeds | one (42) |
| learning rates | **3e-4 for both arms**; `lr_min = 3e-5` (= lr/10) for both |

Both rates are the per-arm optima from the 500-epoch `poisson_benchmark` sweep, and both arms
independently select the same one (`data` 0.0457, `pls` 0.0495 at 3e-4). So the optimizer setup is
identical across arms and the only difference is the loss — a stronger controlled comparison than
per-arm tuning would have given. `lr_min` is set explicitly because the CLI default is a fixed
`1e-4`, which at `lr=3e-4` would be `lr/3` rather than `lr/10`.

## No streaming run for the supervised arm

Deliberate. PLS's infinite-data limit is *reachable*: fresh source terms cost nothing because the
loss needs no labels. Infinite **labelled** data would require a PDE solve per sample and does not
exist in practice — it is free in this repo only because the dataset is manufactured
(`PoissonMultiFrequency` has closed-form solutions). The missing curve is the figure's point, not
a gap in it, and if PLS merely *matches* supervised training in that limit, that is already the
result: the limit is unobtainable for the arm it would help.

## Running it

```bash
# on Euler
cd ~/TensorPILS && git pull && mkdir -p logs
sbatch --array=0-14%4 experiments/poisson/poisson_paper/infinite_data/sweep.sbatch
squeue --me

# locally
rsync -avz --info=progress2 --exclude='checkpoints/' --exclude='visualization/' \
  mzeinhofer@euler.ethz.ch:TensorPILS/output/poisson/poisson_paper/infinite_data/ \
  output/poisson/poisson_paper/infinite_data/
.venv/bin/python experiments/poisson/poisson_paper/infinite_data/plot_scaling.py
```

`%4` throttles to four concurrent tasks. Writes `infinite_data.{png,pdf}` and prints a table of
both arms per size with their ratio, plus the size at which supervised training catches PLS.

## Things to know

- **Reported numbers are best-checkpoint, selected on validation.** "No early stopping" means every
  run spends the full budget, not that the final epoch is reported. The small-`n` supervised runs
  will overfit hard over 32,000 steps; checkpoint selection absorbs that.
- **Fixed budget across sizes** is what `--steps_per_epoch 32` buys: finite datasets are sampled
  i.i.d. with replacement, so `n_train=64` sees its 64 samples ~16 times per virtual epoch while
  `n_train=4096` sees a fraction of its set. Optimizer steps are equal; passes over the data are not.
- **The budget is 2× `poisson_benchmark`** (32,000 steps against 500 × 32 = 16,000), so numbers
  here are not directly comparable to the table there.
- **Labels are FEM solutions, so there is no discretisation floor.** With the closed-form labels
  the arms were scored against different targets: PLS converges to the discrete solution and could
  never cross the discretisation error (0.68% relative here), while supervised training had no such
  floor — a plateau would have been unreadable. Under `--dataset_solution fem` the label *is* what
  the residual losses target, so any plateau is a property of the method. The solve runs once at
  dataset construction, in scipy, outside autograd; nothing backpropagates through it.
- **Numbers here are not comparable to `poisson_benchmark`**, which uses analytic labels.
- **One seed.** On the `K=10` benchmark the supervised arm's spread across seeds was 0.037–0.121 at
  `n=1024`. Treat small differences between neighbouring points as noise.
