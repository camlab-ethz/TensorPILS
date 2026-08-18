# Poisson: the paper table

Final Poisson experiments. Seven arms, one dataset, one optimisation budget — the arms differ
only in the training objective (and, for PI-DeepONet, the architecture it requires).

| # | arm | flags | labels? |
|---|---|---|---|
| 0 | data-driven | `--loss data` | supervised |
| 1 | least squares | `--loss galerkin` | label-free |
| 2 | LS + multigrid | `--loss pls --precond_kind multigrid --mg_omega 8/9` | label-free |
| 3 | Deep Ritz | `--loss deepritz --bc_mode hard` | label-free |
| 4 | Deep Ritz + precond | `--loss deepritz --bc_mode hard --precondition ...` | label-free |
| 5 | PINO | `--loss pino` | label-free |
| 6 | PI-DeepONet | `--model deeponet --loss pideeponet` | label-free |

Shared setting: `K=10`, grid `65²`, `n_train/val/test = 1024/128/256`, 500 epochs, batch 32,
Adam with cosine `lr → lr/10`, `seed=42` (final runs also 43, 44).

## Two stages

**Stage 1 — learning rate.** `lr_sweep.sbatch`: 7 arms × 6 learning rates
(`1e-4 … 3e-2`) = 42 tasks at a **reduced 100-epoch budget**, a fifth of the full runs.
Selection is on the **validation** relative L2, never the test set.

The cosine schedule still anneals `lr → lr/10` over whatever budget the run has, so the schedule
has the same shape in the sweep as in the finals and the comparison stays self-consistent.

Caveat worth one cheap check: the optimal lr depends on the budget — tuning the bare least-squares
arm at 1e4 steps picks ~`3e-2`, while at 1e5 steps a 30× smaller rate is 25× better. Short sweeps
systematically prefer rates that are too large. After picking, re-run the winner and runner-up at
500 epochs for one or two arms (2–4 runs) to confirm the ranking holds before committing.

The grid is wide because the arms do **not** share a reduction: ours is a summed square, PINO's is
a relative norm ratio, Deep Ritz's is an energy (and is routinely negative). Their gradient scales
differ by orders of magnitude, so a narrow grid centred on `1e-3` would be fair to nobody.
`lr_min = lr/10` throughout, so the cosine schedule keeps its shape across the sweep — a fixed
`lr_min` would mean no decay at the small end and a 300× decay at the large end.

**Stage 2 — the table.** `sweep.sbatch`: 7 arms × 3 seeds = 21 tasks at the selected rates.
**Edit `ARMS_LR` first** — `plot_lr_sweep.py` prints the line to paste.

## Run it (Euler)

```bash
cd ~/TensorPILS && mkdir -p logs
git pull

# stage 1
sbatch experiments/poisson/poisson_paper/lr_sweep.sbatch     # 42 tasks
squeue --me

# pull results back and choose the rates
rsync -avz --exclude 'checkpoints' \
    euler:~/TensorPILS/output/poisson/poisson_paper ~/Documents/TensorPILS/output/poisson/
python experiments/poisson/poisson_paper/plot_lr_sweep.py \
    --root output/poisson/poisson_paper/lr_sweep

# edit ARMS_LR in sweep.sbatch with the printed line, commit, push, pull on Euler, then:
sbatch experiments/poisson/poisson_paper/sweep.sbatch        # 21 tasks
```

Logs land in `logs/pp_lr_<jobid>_<task>.out` and `logs/pp_final_<jobid>_<task>.out`.

## Conventions worth knowing

- **Run prefixes encode neither the learning rate nor the seed**, so each lr (stage 1) and each
  seed (stage 2) writes to its own output directory. The seven arms *are* distinguished by prefix
  (`fno_data_…`, `fno_galerkin_…`, `fno_pls_mg-L4-s22_…`, `fno_deepritz_bc-hard_…`,
  `fno_deepritz_precond-mg-L4-s22_…`, `fno_pino-rel_…`, `deeponet_pi-rel_…`), so they can share one.
- **`--bc_mode hard` on both Deep Ritz arms.** The preconditioned variant implies hard BC anyway;
  matching it on the bare arm keeps the boundary treatment identical and avoids carrying a tuned
  `lambda_bc = 100` into one row of the table.
- **`omega = 8/9`** is the optimal Jacobi damping for `Q1` in 2D (the high-frequency spectrum of
  `D⁻¹A` is `[3/4, 3/2]`; smoothing factor 1/3 rather than 1/2 at the usual `2/3`). At `65²` the
  hierarchy is `65 → 33 → 17 → 9`, exactly dyadic and nested.
- **PINO and PI-DeepONet carry the `sin(πx)sin(πy)` mollifier** (`--mollify auto`), a hard BC that
  the `galerkin`/`pls` arms do not get. This is the reference implementations' own choice and it
  helps them; removing it would be strawmanning. It is worth stating in the paper.
