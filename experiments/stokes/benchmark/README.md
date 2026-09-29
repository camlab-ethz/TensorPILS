# Stokes flow past an obstacle

Reproduces the Stokes part of the paper: the Stokes rows of **Table 1**, **Figure 5** (domain and
validation curves), the appendix **sample figure**, the Stokes **hyperparameter table** and the
**preconditioner ablation** table (including its condition-number row).

## Setup

- **Problem.** Stationary Stokes, `-Δu + ∇p = f`, `∇·u = 0`, zero velocity on both boundary
  components, zero-mean pressure, on the unit square with a circular hole of radius 0.14 centred
  at (0.40, 0.50). The body force is a random sine series with `K = 10` modes per direction.
- **Discretisation.** Taylor–Hood P2/P1 on an unstructured Gmsh triangulation with target edge
  length 0.035: 1928 elements, 3998 velocity and 1035 pressure nodes. The reference solution is the
  discrete solve, so the label-free losses and the labels target the same discrete state.
- **Model.** GAOT, queried at the P2 nodes, with outputs `(u_x, u_y, p)`.
- **Losses.** `L_PLS`: least squares weighted by the block preconditioner
  `P = diag(A⁻¹, ω_S · diag(M_p)⁻¹)`, with `A⁻¹` one algebraic multigrid V(2,2) cycle (AmgX) per
  velocity component and `ω_S = 16`. `L_data`: supervised, in the finite-element norms. `L_LS`: the
  unpreconditioned least-squares residual, as a control. Baseline: PI-DeepONet with the
  strong-form residual by automatic differentiation at all interior nodes, a boundary penalty
  `λ_bc = 0.1` and continuity weight `w = 100`.
- **Optimisation.** Adam, 500 epochs, batch 32, cosine schedule to `lr_min = 1e-6`,
  1024/128/256 samples. Learning rate `1e-3` (GAOT) and `1e-4` (PI-DeepONet), selected on
  validation with seed 42 (`grid_search.sh`), then seeds 42, 43 and 44 (`run.sh`).
- **Metric.** Relative finite-element L² error of velocity and pressure on the test set, at the
  checkpoint with the lowest validation mean of the two.

## Run

From the repository root, on a machine with a CUDA GPU (AmgX is CUDA-only):

```bash
bash experiments/stokes/benchmark/grid_search.sh     # 12 runs, seed 42 (hyperparameters, ablation)
bash experiments/stokes/benchmark/run.sh             # 15 runs (Table 1, figures)
python experiments/stokes/benchmark/summarize.py     # tables
python experiments/stokes/benchmark/conditioning.py  # condition numbers of the ablation table
python experiments/stokes/benchmark/plot_paper.py    # Figure 5
python experiments/stokes/benchmark/plot_samples.py  # appendix sample figure
```

Each launcher runs all its tasks in order; `bash <script> list` shows them and `bash <script> <i>`
runs one. On a SLURM cluster, one GPU per task:

```bash
sbatch --array=0-14 --gpus=1 --time=02:00:00 --wrap "bash experiments/stokes/benchmark/run.sh"
```

Results go to `output/stokes/benchmark/` and the generated mesh to `output/meshes/`.

## Expected results

Table 1 (test relative L² in %, mean ± half-range over three seeds; seconds per epoch on an
RTX 4090):

| method | velocity | pressure | s / epoch |
|---|---|---|---|
| `L_PLS` | 1.6 ± 0.2 | 0.85 ± 0.04 | 3.8 |
| `L_data` | 3.3 ± 0.2 | 1.79 ± 0.16 | 2.9 |
| PI-DeepONet | 39.7 ± 2.1 | 36.7 ± 0.6 | 6.0 |

The unpreconditioned `L_LS` ends at 30.6 ± 1.6 % velocity and 11.4 ± 0.9 % pressure. PI-DeepONet
with 1024 random collocation points per step takes 1.5 s per epoch at 44.4 ± 4.3 % / 35.1 ± 1.2 %.

Preconditioner ablation (seed 42, test relative L² in %):

| | `L_LS` | ω_S = 0.5 | 4 | **16** | 32 | 64 | 256 |
|---|---|---|---|---|---|---|---|
| velocity | 30.51 | 2.58 | 1.63 | **1.69** | 3.35 | 4.25 | 22.94 |
| pressure | 10.22 | 1.25 | 0.95 | **0.79** | 1.40 | 1.62 | 3.93 |
| κ | 1.2e11 | 3.7e6 | 1.0e6 | 1.9e6 | 3.2e6 | 5.8e6 | 2.1e7 |

Individual runs are not bitwise reproducible on a GPU; expect agreement within the seed spread.
Epoch times depend on the hardware and its load.

At the learning rate `1e-3`, `L_PLS` training occasionally shows a loss spike, after which the
validation error recovers over the following epochs. Of nine runs with `ω_S = 16` (the paper's
and later reruns of seeds 42 and 43), five spiked; eight ended between 1.31 and 1.76 % velocity
and 0.79 and 0.90 % pressure error, and one, which spiked early (epochs 72 and 138), ended at
3.2 % and 1.3 %. A run far above the table is therefore most likely such a spike; its validation
curve in `results/*.json` shows it.

## Compute

One run of 500 epochs takes 25–50 minutes on an RTX 4090 and a few GB of GPU memory. `run.sh` is
about 7 GPU-hours and `grid_search.sh` about 5.
