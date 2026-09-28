# Allen–Cahn time stepping

Reproduces the Allen–Cahn rows of **Table 1** and **Figure 4** (validation curves and free
roll-out to `T = 1`).

## Setup

- **Problem.** `∂_t u = Δu + ε² u (1 − u²)` on the unit square with zero Dirichlet data,
  `ε = 32`. The learned operator is one step `τ = 0.01` of the convex–concave (Eyre) scheme,
  cubic term implicit and linear term explicit. Initial conditions are sine series with `K = 4`
  modes per direction; the reference trajectories come from a finite-element Newton solve.
- **Discretisation.** Bilinear finite elements on the uniform 129 × 129 grid, whose nodes are the
  FNO's grid (`h ε = 1/4`).
- **Training.** Autoregressive roll-out over the first 10 steps (to `t = 0.1`) with detached
  inputs (pushforward), seeded with the exact initial condition. `L_PLS` is the step residual
  preconditioned by one geometric multigrid V-cycle for the frozen Jacobian
  `J₀ = (τ⁻¹ + 3ε²) M + A`; `L_LS` is the unpreconditioned residual; `L_data` matches the reference
  trajectory. Baselines: PINO (finite-difference strong-form residual) and PI-DeepONet (autodiff
  residual, 1024 random collocation points per step, output multiplied by `sin(πx) sin(πy)`).
- **Optimisation.** Adam, 500 epochs, batch 16, cosine schedule from `lr` to `lr / 10`,
  1024/128/256 samples; learning rate selected on validation with seed 42 (`lr_sweep.sh`), then
  seeds 42, 43 and 44 (`run.sh`).
- **Metric.** Test space-time relative L² over steps 0–10, per sample and averaged over the test
  set, at the best-validation checkpoint (`evaluate_test.py`).

## Run

From the repository root, on a machine with a CUDA GPU:

```bash
bash experiments/allen_cahn/benchmark/lr_sweep.sh      # 30 runs, seed 42 (optional)
python experiments/allen_cahn/benchmark/plot_lr_sweep.py
bash experiments/allen_cahn/benchmark/run.sh           # 15 runs
python experiments/allen_cahn/benchmark/evaluate_test.py   # test errors and roll-outs to T = 1
python experiments/allen_cahn/benchmark/plot_paper.py      # Figure 4 and the Table 1 rows
```

Each launcher runs all its tasks in order; `bash <script> list` shows them and `bash <script> <i>`
runs one, which is also what a SLURM array job does. `plot_final.py` draws per-arm diagnostic
curves of the final runs.

## Expected results

Selected learning rates: `3e-4` for `L_PLS` and `L_data`, `1e-3` for `L_LS`, PINO and PI-DeepONet.

Table 1 (test relative L² in %, mean ± half-range over three seeds; seconds per epoch on an
RTX 4090):

| method | test error | s / epoch |
|---|---|---|
| `L_PLS` | 3.2 ± 0.3 | 46 |
| `L_data` | 2.0 ± 0.2 | 40 |
| PINO | 5.3 ± 1.6 | 40 |
| PI-DeepONet | 90.0 ± 1.7 | 8.2 |

In the free roll-out to `T = 1` (Figure 4, right) every physics-informed arm except PI-DeepONet
stays accurate far beyond the training horizon.

## Compute

This is the most expensive benchmark: a run of 500 epochs takes about 6 hours on an RTX 4090
(PI-DeepONet about 1 hour), so `run.sh` is roughly 75 GPU-hours and `lr_sweep.sh` roughly 140.
