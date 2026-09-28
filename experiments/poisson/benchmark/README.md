# Poisson benchmark

Reproduces the Poisson rows of **Table 1** and the Poisson **optimization hyperparameters** of the
appendix. The checkpoints of these runs are also the models of the appendix resolution figure.

## Setup

- **Problem.** `-Δu = ρ` on the unit square with zero Dirichlet data. Sources and solutions are
  sine series with `K = 10` modes per direction (closed form, no PDE solve).
- **Discretisation.** Bilinear finite elements on the uniform 65 × 65 grid, whose nodes are the
  FNO's grid.
- **Methods.** `L_PLS` (least squares preconditioned by one geometric multigrid V(2,2) cycle,
  weighted Jacobi with ω = 8/9), `L_data` (supervised), `L_LS` (unpreconditioned least squares),
  PINO (finite-difference strong-form residual) and PI-DeepONet (autodiff strong-form residual,
  boundary penalty `λ_bc`). PINO and PI-DeepONet set the boundary nodes of the output to zero, as
  our own losses do.
- **Model.** FNO (16 × 16 modes, width 64, 5 layers, 3.0 M parameters); DeepONet for PI-DeepONet.
- **Optimisation.** Adam, 500 epochs, batch 32, cosine schedule from `lr` to `lr / 10`,
  1024/128/256 samples. The learning rate (and PI-DeepONet's `λ_bc`) is selected on validation with
  seed 42 at the full budget (`lr_sweep.sh`); the final runs use seeds 42, 43 and 44 (`run.sh`).
- **Metric.** Relative L² error on the test set at the checkpoint with the lowest validation error.

## Run

From the repository root, on a machine with a CUDA GPU:

```bash
bash experiments/poisson/benchmark/lr_sweep.sh      # 46 runs, seed 42
bash experiments/poisson/benchmark/run.sh           # 15 runs
python experiments/poisson/benchmark/summarize.py
```

Each launcher runs all its tasks in order; `bash <script> list` shows them and `bash <script> <i>`
runs one, which is also what a SLURM array job does:

```bash
sbatch --array=0-14 --gpus=1 --time=01:00:00 --wrap "bash experiments/poisson/benchmark/run.sh"
```

Results go to `output/poisson/benchmark/`. The reference implementation's PINO boundary treatment,
multiplying the output by `sin(πx) sin(πy)`, is available as a control with `--pino_bc mollifier`.

## Expected results

Selected on validation: learning rate `3e-4` for `L_PLS` and `L_data`, `3e-3` for `L_LS` and PINO,
`1e-4` with `λ_bc = 0.03` for PI-DeepONet.

Table 1 (test relative L² in %, mean ± half-range over three seeds; seconds per epoch on an
RTX 4090):

| method | test error | s / epoch |
|---|---|---|
| `L_PLS` | 5.1 ± 0.8 | 1.2 |
| `L_data` | 4.7 ± 0.8 | 1.1 |
| PINO | 33.1 ± 1.1 | 1.1 |
| PI-DeepONet | 21.4 ± 4.7 | 3.1 |

Individual runs are not bitwise reproducible on a GPU; expect agreement within the seed spread.

## Compute

A run takes 10–30 minutes on an RTX 4090. `run.sh` is about 4 GPU-hours and `lr_sweep.sh` about 14.
