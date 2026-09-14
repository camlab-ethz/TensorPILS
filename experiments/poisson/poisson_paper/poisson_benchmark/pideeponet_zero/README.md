# PI-DeepONet with the zero-BC treatment (the Table 1 row)

**The question.** Table 1's PINO arm imposes the zero Dirichlet BC by zeroing the boundary nodes
of its output (`--pino_bc zero`), exactly as our own arms do — but the PI-DeepONet arm it sat
next to still carried the reference implementation's `sin(πx)sin(πy)` mollifier. On this
truncated-sine dataset that ansatz is close to the solution's own structure and does not
generalise to other geometries, so the two baselines were not treated alike. This study reruns
PI-DeepONet with the matching treatment, `--pideeponet_bc zero`, under the table's own protocol.

**What `--pideeponet_bc zero` is.** The grid output's boundary ring is set to zero (the
operation `losses.py` applies to our arms, and what `--pino_bc zero` does), and the residual
carries the soft boundary penalty of the original PI-DeepONet paper, weighted by
`--pi_lambda_bc`. The penalty is not optional: PINO's finite-difference stencil touches the
zeroed boundary ring, so masking alone couples its residual to the BC, whereas an autodiff
Laplacian at interior points is blind to the boundary values and to any mask on them — without
the penalty the objective is invariant under adding harmonic functions (measured: `λ=0` gives
162–184 % error). The penalty is written in the residual's own units — per sample, boundary RMS
divided by the detached interior RMS, dimensionless like the relative residual `‖Lu−f‖/‖f‖` — so
`λ=1` is a meaningful starting point rather than a number that depends on the solution scale
(at `K=10` the solutions have RMS `~1.6e-3`; a plain MSE penalty at unit weight would be five
orders of magnitude too weak). Implementation: `DeepONetModel(zero_boundary=True)`,
`PIDeepONetPoissonLoss.boundary_term`, `PIDeepONetPoissonTrainer(pi_bc="zero")`.

## Facts

| | |
|---|---|
| Dataset / grid | Poisson, `K=10`, `65²`, `n_train/val/test = 1024/128/256`, analytic labels — identical to the table |
| Model | DeepONet `p=128`, width 256, depth 4, 64 random Fourier trunk features (scale 2) — identical to the table |
| Budget | 500 epochs, batch 32, Adam, cosine `lr → lr/10` — identical to the table |
| BC | `--pideeponet_bc zero`: boundary nodes zeroed + relative boundary penalty `--pi_lambda_bc` |
| Stage 1 | seed 42, **full** 500-epoch budget: `lr ∈ {3e-5, 1e-4, 3e-4, 1e-3} × λ ∈ {0.3, 1, 3}`, then `λ ∈ {0.1, 0.03, 0}` at the two best `lr`, then `λ ∈ {0.01, 0.003, 0.001}` at `lr 1e-4` plus `lr 3e-4 × λ 0.03` — each extension because the optimum sat on the grid's lower `λ` edge; `λ=0` is the no-BC control. 22 cells. Selection on **validation** relative L² |
| Stage 2 | seeds 42, 43, 44 at the selected `(lr, λ)`, in `final/lr<lr>_lam<λ>/` (config-named, so a re-selection never mixes configs) |
| Control | the reference's mollifier (`--pideeponet_bc mollifier`, `lr 1e-4` from the main sweep), seeds 42–44 |
| Metric | FEM relative L² on the test set at the best-validation checkpoint, mean ± half-range over seeds (Table 1's convention) |
| Output | `output/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/{lr_sweep,final/lr*_lam*,control_mollified}` (git-ignored) |
| Cost | ~30 min per cell alone on an RTX 4090; ~7 GB GPU memory per process, so at most 3 concurrently on a 24 GB card |

## Run

From a **login** node (the route the rest of the table uses):

```bash
cd ~/TensorPILS && mkdir -p logs
sbatch experiments/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/lr_sweep.sbatch   # 22 cells, seed 42
python experiments/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/summarize.py        # prints "selected: lr=… lambda_bc=…"
# edit ZERO_LR / ZERO_LAM in sweep.sbatch if the selection differs, then
sbatch experiments/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/sweep.sbatch       # 3 seeds x {zero, mollified control}
python experiments/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/summarize.py        # the Table 1 cell
```

On a **compute** node (already inside an allocation), the same protocol end to end, three cells
at a time, restartable:

```bash
PAR=3 bash experiments/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/run_local.sh
DRY_RUN=1 bash experiments/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/run_local.sh   # list what would run
```

One cell directly:

```bash
python -m tensorpils.cli --pde poisson --model deeponet --loss pideeponet \
    --pideeponet_bc zero --pi_lambda_bc 0.3 \
    -k 10 --grid_resolution 65 --n_train 1024 --n_val 128 --n_test 256 \
    --epochs 500 --batch_size 32 --optimizer adam --lr 1e-4 --lr_min 1e-5 \
    --seed 42 --device cuda --output_dir output/poisson/poisson_paper/poisson_benchmark/pideeponet_zero/final/lr1e-4_lam0.3/seed42
```

Runs are tagged `deeponet_pi-rel-zbc<λ>_K10_…` (mollified: `deeponet_pi-rel_K10_…`), so the two
BC treatments never land on each other's files; `plot_final.py --root …/pideeponet_zero/final/lr<lr>_lam<λ>`
also picks the arm up (prefix `deeponet_pi-`).

## Results

Run on an Euler `eu-g6` node (RTX 4090), 2026-09-08/10. `summarize.py` reproduces every number below
from the results JSONs.

**Stage 1 — the penalty weight has a sharp interior optimum.** Best validation relative L² on
seed 42, 500 epochs. At `lr 1e-4` the curve in `λ` is U-shaped: too large and the penalty
dominates the residual, too small and the boundary condition is lost (`λ=0` is the no-BC control).

| `λ_bc` | 3 | 1 | 0.3 | 0.1 | **0.03** | 0.01 | 0.003 | 0.001 | 0 |
|---|---|---|---|---|---|---|---|---|---|
| best-val, `lr 1e-4` | 88.1 % | 49.2 % | 34.9 % | 29.9 % | **22.1 %** | 24.0 % | 38.9 % | 59.2 % | 179.7 % |

The learning rate is an interior optimum as well: at `λ=0.3`, `3e-5 / 1e-4 / 3e-4 / 1e-3` give
`50.2 / 34.9 / 62.2 / 83.1 %`, and at `λ=0.03`, `3e-5 / 1e-4 / 3e-4` give `37.5 / 22.1 / 62.5 %`.
**Selected: `lr 1e-4`, `λ_bc 0.03`.**

**Stage 2 — the Table 1 row.** Test relative L² at the best-validation checkpoint, mean ± half-range
over seeds 42/43/44 (the table's convention).

| arm | seed 42 | seed 43 | seed 44 | mean ± half-range |
|---|---|---|---|---|
| **PI-DeepONet, zero-BC (`lr 1e-4`, `λ 0.03`)** | 22.04 % | 25.83 % | 16.33 % | **21.4 ± 4.7 %** |
| PI-DeepONet, zero-BC (`lr 1e-4`, `λ 0.3`, the first grid's edge) | 34.95 % | 40.58 % | 34.99 % | 36.8 ± 2.8 % |
| PI-DeepONet, mollifier (`lr 1e-4`, the reference's BC; Table 1 before this rerun) | 19.74 % | 18.85 % | 18.72 % | 19.1 ± 0.5 % |
| PINO, zero-BC (`--pino_bc zero`, Table 1) | | | | 33.1 ± 1.1 % |

Three things to carry into the text:

- **The mollified control reproduces the main benchmark to the digit** (`0.191 ± 0.005` here,
  `0.191 ± 0.005` in the table; seed 42's best-val 20.35 % matches the `0.2035` recorded in
  `../sweep.sbatch`), so the two BC treatments are compared on one pipeline and the difference is
  attributable to the BC mechanism alone.
- **With the BC treated like our arms, PI-DeepONet lands at 21.4 %, within its own seed spread of
  the mollified 19.1 %** — once the penalty weight is tuned. The weight is not a detail: the first
  grid's edge value `λ=0.3` gives 36.8 %, and `λ=0` gives >160 %. The seed spread (16–26 %) is the
  largest of any Table 1 arm.
- **Every run is still descending at epoch 500** (best epoch ≥ 496 for all seeds), as for the bare
  least-squares arm: these are fixed-budget numbers, not converged ones, and should be described as
  such.

