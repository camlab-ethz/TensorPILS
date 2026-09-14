# Poisson: the paper table

Final Poisson experiments. Seven arms, one dataset, one optimisation budget — the arms differ
only in the training objective (and, for PI-DeepONet, the architecture it requires).

| # | arm | flags | labels? |
|---|---|---|---|
| 0 | data-driven | `--loss data` | supervised |
| 1 | least squares | `--loss galerkin` | label-free |
| 2 | LS + multigrid | `--loss pls --precond_kind multigrid --mg_omega 8/9` | label-free |
| 3 | PINO | `--loss pino` | label-free |
| 4 | PI-DeepONet | `--model deeponet --loss pideeponet` | label-free |

Deep Ritz and its preconditioned variant were part of this sweep and have been removed; they are
reported separately. Their results remain in the output directories and are simply not matched by
the plotting scripts any more.

Shared setting: `K=10`, grid `65²`, `n_train/val/test = 1024/128/256`, 500 epochs, batch 32,
Adam with cosine `lr → lr/10`, `seed=42` (final runs also 43, 44).

## Two stages

**Stage 1 — learning rate.** `lr_sweep.sbatch`: 7 arms × 6 learning rates
(`1e-4 … 3e-2`) = 30 tasks at a **reduced 100-epoch budget**, a fifth of the full runs.
Selection is on the **validation** relative L2, never the test set.

The cosine schedule still anneals `lr → lr/10` over whatever budget the run has, so the schedule
has the same shape in the sweep as in the finals and the comparison stays self-consistent.

Caveat worth one cheap check: the optimal lr depends on the budget — tuning the bare least-squares
arm at 1e4 steps picks ~`3e-2`, while at 1e5 steps a 30× smaller rate is 25× better. Short sweeps
systematically prefer rates that are too large. After picking, re-run the winner and runner-up at
500 epochs for one or two arms (2–4 runs) to confirm the ranking holds before committing.

The grid is wide because the arms do **not** share a reduction: ours is a summed square, PINO's is
a relative norm ratio. Their gradient scales
differ by orders of magnitude, so a narrow grid centred on `1e-3` would be fair to nobody.
`lr_min = lr/10` throughout, so the cosine schedule keeps its shape across the sweep — a fixed
`lr_min` would mean no decay at the small end and a 300× decay at the large end.

**Stage 2 — the table.** `sweep.sbatch`: 5 arms × 3 seeds = 15 tasks at the selected rates.
**Edit `ARMS_LR` first** — `plot_lr_sweep.py` prints the line to paste.

## Run it (Euler)

```bash
cd ~/TensorPILS && mkdir -p logs
git pull

# stage 1
sbatch experiments/poisson/poisson_paper/poisson_benchmark/lr_sweep.sbatch     # 30 tasks
squeue --me

# pull results back and choose the rates
rsync -avz --exclude 'checkpoints' \
    euler:~/TensorPILS/output/poisson/poisson_paper ~/Documents/TensorPILS/output/poisson/
python experiments/poisson/poisson_paper/poisson_benchmark/plot_lr_sweep.py \
    --root output/poisson/poisson_paper/poisson_benchmark/lr_sweep

# edit ARMS_LR in sweep.sbatch with the printed line, commit, push, pull on Euler, then:
sbatch experiments/poisson/poisson_paper/poisson_benchmark/sweep.sbatch        # 15 tasks
```

Logs land in `logs/pp_lr_<jobid>_<task>.out` and `logs/pp_final_<jobid>_<task>.out`.

## Conventions worth knowing

- **Run prefixes encode neither the learning rate nor the seed**, so each lr (stage 1) and each
  seed (stage 2) writes to its own output directory. The five arms *are* distinguished by prefix
  (`fno_data_…`, `fno_galerkin_…`, `fno_pls_mg-L4-s22_…`, `fno_pino-rel_…`, `deeponet_pi-rel_…`),
  so they can share one.
- **`omega = 8/9`** is the optimal Jacobi damping for `Q1` in 2D (the high-frequency spectrum of
  `D⁻¹A` is `[3/4, 3/2]`; smoothing factor 1/3 rather than 1/2 at the usual `2/3`). At `65²` the
  hierarchy is `65 → 33 → 17 → 9`, exactly dyadic and nested.
- **PINO zeroes its boundary nodes** (`--pino_bc zero`, the default), the same operation
  `losses.py` applies to our own arms — so it differs from the `ls` arm only in the residual.
  **The table's PI-DeepONet arm (`--loss pideeponet` with defaults) still carries the
  `sin(πx)sin(πy)` mollifier**, a hard BC that the `galerkin`/`pls` arms do not get. This is the
  reference implementation's own choice and it helps it; it is worth stating in the paper.
- **`--pideeponet_bc zero` is the PI-DeepONet counterpart of `--pino_bc zero`**: the grid
  output's boundary nodes are zeroed as our arms do, and — because the autodiff Laplacian at
  interior points cannot see the boundary values or any mask on them (unlike PINO's FD stencil,
  which touches the zeroed ring) — the residual gets the original PI-DeepONet soft boundary
  penalty, weighted by `--pi_lambda_bc`. The penalty is expressed in the residual's own units
  (boundary RMS relative to the solution's own RMS, denominator detached; see
  `PIDeepONetPoissonLoss`), so `lambda=1` is a meaningful default; at `K=10` the solutions have
  RMS `~1.6e-3` and a plain MSE penalty at unit weight would be five to six orders of magnitude
  too weak.
  Without the penalty the objective is invariant under adding harmonic functions, i.e. it has no
  BC at all. The study that produced the table's PI-DeepONet number with this treatment lives in
  [`pideeponet_zero/`](pideeponet_zero/README.md) (its own two-stage sbatch pair, a compute-node
  driver, and `summarize.py`); it writes to `output/.../poisson_benchmark/pideeponet_zero/`. Runs
  are tagged `deeponet_pi-rel-zbc<lambda>_…` so they never collide with the mollified arm's files.
