# Variant: PLS — blend-strength sweep on the **256² grid** (fast DST preconditioner)

Same experiment as [`../pls/`](../pls/README.md) and [`../pls128/`](../pls128/README.md), on the
**256²** grid. At this size the dense eigendecomposition of `A` (`N≈64516`) needs a ~33 GB
eigenvector matrix and **will not run** — the exact **fast sine transform** (`--precond_method sine`)
is the only viable route; it applies `P` in `O(N^{1.5})` with `O(N)` memory. See
`../../../notes/preconditioner_notes/` (§Fast realization on the regular grid).

Preset to the sanity config (`EPOCHS=20`, coarse `t` grid); **raise `EPOCHS`** at the top of
`sweep.sbatch` once the 128 sanity looks right. Note the 256² dataset generation (FEM reference
solves for `n_train+n_val+n_test` samples) and training are markedly heavier than 128² — hence the
longer wall-time and `gpumem:20g`.

## Facts

| | |
|---|---|
| Loss | `pls` (`½‖P(Au−b)‖²`) |
| Preconditioner | `blend` + `--precond_method sine` (dense infeasible here); `t ∈ {1.0, 0.9, 0.5, 0.25, 0.01, 0.0}` (+ optional `multigrid`, task 6) |
| Dataset | 2D Poisson, `K=4`, grid `256²`, `n_train=1024`, `n_val=128`, `n_test=256`, `seed=42` |
| Optimizer | `adam`, default cosine lr `1e-3 → 1e-4` |
| Epochs / batch | `20` (sanity; raise later) / `32` |
| Output dir | `output/poisson/sweep_blend/pls256/` (git-ignored) |
| Recorded | per-epoch val relative-L2, and `κ(PA)`/`κ(H)` per run (in the results JSON) |

## Run

```bash
mkdir -p logs
sbatch experiments/poisson/sweep_blend/pls256/sweep.sbatch          # array 0-5 (six t values)
sbatch --array=6 experiments/poisson/sweep_blend/pls256/sweep.sbatch # + multigrid baseline
```

## Plot

```bash
python experiments/poisson/sweep_blend/plot_sweep.py --results_dir output/poisson/sweep_blend/pls256/results
```
