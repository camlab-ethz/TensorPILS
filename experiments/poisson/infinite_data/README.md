# Training at the infinite-data limit

Reproduces **Figure 3**: test error against training-set size at a fixed optimisation budget, for
the preconditioned least-squares loss `L_PLS` and supervised training `L_data`, and `L_PLS` on an
infinite stream of fresh samples, none of which is ever seen twice.

## Setup

- Poisson with `K = 10` on the 65 × 65 grid, FNO, multigrid V(2,2) cycle with Jacobi damping 8/9.
- **Fixed budget.** Every run takes 32,000 optimizer steps at batch size 32 (1000 virtual epochs of
  32 steps). Finite training sets are sampled with replacement, so only the size of the set varies.
- **Labels are FEM solutions** (`--dataset_solution fem`), the discrete solution the residual
  losses target, for training, validation and test alike; the closed-form solution would put a
  discretisation floor under `L_PLS` alone.
- **Shared evaluation sets.** Validation and test sets are drawn from dedicated seeds
  (`--fixed_eval`), so every run is scored on the same samples.
- Adam with learning rate `3e-4` cosine-annealed to `3e-5` for both losses (the rate both select in
  the Poisson benchmark), seed 42. Reported errors are at the best-validation checkpoint.

There is no streaming run for `L_data`: infinite labelled data would need one PDE solve per
sample, while fresh sources cost `L_PLS` nothing.

## Run

From the repository root, on a machine with a CUDA GPU:

```bash
bash experiments/poisson/infinite_data/run.sh             # 13 runs, ~30-40 minutes each
python experiments/poisson/infinite_data/plot_scaling.py  # -> output/poisson/infinite_data/
```

## Expected results

At every finite size up to 2048 samples the two losses are indistinguishable. The infinite stream
reaches 2.1 % relative error, against 2.8 % for supervised training on 2048 samples.
