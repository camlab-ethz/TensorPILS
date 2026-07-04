# TensorPILS

**Physics-Informed Learning System powered by [TensorMesh](https://github.com/camlab-ethz/TensorMesh).**

TensorPILS trains neural operators (currently a Fourier Neural Operator, FNO) to
solve PDEs with loss functions built from TensorMesh's finite-element machinery — so
most of them need **no labelled solutions**. Three PDEs are supported, selected with
`--pde` (all on the unit square with homogeneous Dirichlet BC):

| `--pde` | Equation | Kind | Operator |
|---------|----------|------|----------|
| `poisson` | −Δu = f | static | source → solution |
| `wave` | uₜₜ = c²Δu | time-dependent (2nd order) | autoregressive stepper `[uⁿ⁻¹, uⁿ] → uⁿ⁺¹` |
| `ac` | uₜ = a²Δu + ε²u(1−u²) | time-dependent, nonlinear (1st order) | autoregressive stepper `uⁿ → uⁿ⁺¹` |

The FEM stiffness `A`, mass `M`, the analytical fields / reference solvers, and the
boundary handling all come from TensorMesh; TensorPILS contributes the structured-grid
bridge the FNO needs, the model, the losses, the multigrid preconditioner, and the
training loop.

### Losses

**Poisson** offers four interchangeable losses — three of them label-free:

| Loss | Idea | Labels? |
|------|------|---------|
| `data` | supervised MSE against the analytical solution | yes |
| `galerkin` | weak-form residual ‖A u − b‖² on node values | no |
| `deepritz` | energy functional ∫(½\|∇u\|² − f u) dx (+ BC) | no |
| `pls` | preconditioned least-squares ½‖P(A u − b)‖², P ≈ A⁻¹ from a geometric-multigrid V-cycle | no |

**Wave / Allen–Cahn** are trained as autoregressive time-steppers with a weighted mix
`λ_gal·galerkin + λ_data·data` (`--loss` presets the mix; `--lambda_galerkin` /
`--lambda_data` override):

| Loss | Idea | Labels? |
|------|------|---------|
| `galerkin` | label-free weak-form residual of the discrete time scheme, summed over the rollout (central-difference for wave, backward-Euler for Allen–Cahn) | no |
| `data` | supervised MSE against the reference trajectory | yes |

Wave has an analytical multi-frequency solution; Allen–Cahn has **none**, so its
reference/labels come from a FEM implicit-Euler + Newton solver assembled from TensorMesh
(`ACProblem.fem_reference`, run at dataset-build time).

## Install

TensorPILS depends on `tensormesh-fem` (import name `tensormesh`) and
`neuraloperator` (import name `neuralop`).

```bash
# On the cluster, activate the shared environment first:
source ~/venvs/tensorgalerkin/bin/activate
module load stack/.2024-04-silent gcc/8.5.0
module load mesa-glu/9.0.2
module load ffmpeg

# If tensormesh-fem is not already installed, install the local checkout:
pip install -e ../TensorMesh

# Install TensorPILS in development mode:
pip install -e ".[test]"
```

## Usage

After installation a `tensorpils` console command is available (equivalent to
`python -m tensorpils.cli`):

```bash
# --- Poisson (default --pde poisson) ---
tensorpils --loss galerkin --n_train 800 -k 4 --epochs 500     # weak-form residual (label-free)
tensorpils --loss deepritz --bc_mode hard --precondition       # multigrid-preconditioned Deep Ritz
tensorpils --loss pls --mg_levels 4 --mg_pre_smooth 2          # preconditioned least-squares
tensorpils --loss data --n_train 800 -k 4 --epochs 500         # supervised (analytical solution)

# --- Wave (autoregressive time-stepper) ---
tensorpils --pde wave --loss galerkin --dt 0.005 --n_steps 20 --rollout_steps 4 --epochs 300
tensorpils --pde wave --loss data --wave_c 1.0 -k 4            # supervised variant

# --- Allen–Cahn (nonlinear; builds a FEM Newton reference at startup) ---
tensorpils --pde ac --loss galerkin --ac_a 1 --ac_eps 2 --dt 1e-3 --n_steps 20 --rollout_steps 4 --epochs 300
tensorpils --pde ac --loss data -k 4                          # supervised variant
```

Useful flags: `--pde {poisson,wave,ac}`, `--bc_mode {penalty,hard}` (Poisson
`data`/`deepritz`), `--precondition` (multigrid-preconditioned Deep Ritz),
`--grid_resolution`, `--n_modes`, `--hidden_dim`, `--num_layers`, `--optimizer`,
`--device`, `--eval_only --checkpoint <path>`. Time-dependent PDEs add `--dt`,
`--n_steps`, `--rollout_steps`, `--discount_factor`, `--lambda_galerkin`,
`--lambda_data`; Allen–Cahn adds `--ac_a`, `--ac_eps`, `--ac_r`. Run
`tensorpils --help` for the full list.

## Example results

One **label-free** training case per PDE, all on a `64×64` grid with `K=4` and the same
FNO (`n_modes=(16,16)`, `hidden=64`, 5 layers, ~3.0 M params) on a single GPU. The FNO
sees no labelled solutions during training — the reference is used only for evaluation.
The reported error is the **median relative L² error** on the held-out test set
(grid-space vs. the analytical/FEM reference — the same metric across every loss and PDE);
for the time-dependent PDEs it is the error of the autoregressive rollout.

| PDE | `--loss` | test rel-L² (median) | val MSE | run |
|-----|----------|----------------------|---------|-----|
| Poisson | `pls` | **2.52 %** | 3.9 × 10⁻⁸ | `--loss pls --n_train 1024 --epochs 500 --mg_levels 4` |
| Wave | `galerkin` | **1.47 %** | 1.0 × 10⁻⁶ | `--pde wave --n_train 512 --dt 0.005 --n_steps 20 --rollout_steps 4 --epochs 300 --lr 2e-3` |
| Allen–Cahn | `galerkin` | **1.26 %** | 5.9 × 10⁻⁷ | `--pde ac --n_train 512 --ac_eps 2 --dt 1e-3 --n_steps 20 --rollout_steps 4 --epochs 300 --lr 2e-3` |

All three reach ~1–3 % relative error **without labels**, matching supervised training
(e.g. Poisson `--loss data` reaches 1.76 % on the same setup).

For Poisson, the loss choice illustrates the library's central point: plain `galerkin`
(the raw residual ‖Au − b‖²) is badly conditioned and stalls at **~64 %** under this
budget, while `pls` — the *same* residual preconditioned by one geometric-multigrid
V-cycle (P ≈ A⁻¹) — recovers supervised-like accuracy (2.52 %) with no labels.

## Outputs

Each run writes under `--output_dir` (default `output/`):

```
output/
├── checkpoints/     # best model (.pth), keyed by loss/config
├── curves/          # training-loss + validation-error curves (.png)
├── visualization/   # source / ground-truth / prediction / error panels (.png)
└── error/           # test-set relative-L2 error distribution (.png)
```

## Ground truth

Evaluation is **always** grid-space MSE against the reference solution, regardless of
the training loss, so results are comparable across losses and PDEs:

- **Poisson** — analytical multi-mode solution via `tensormesh.dataset.PoissonMultiFrequency`
  (`r = -0.5`). Label-free losses drive the FNO toward the discrete FEM solution `A⁻¹b`.
- **Wave** — analytical trajectory `u(t_k)` via `tensormesh.dataset.WaveMultiFrequency`.
- **Allen–Cahn** — **no analytical solution**; the reference trajectory is produced by a
  batched FEM implicit-Euler + Newton solver (`ACProblem.fem_reference`, run once at
  dataset-build time, on the GPU when available). It solves the *same* discrete residual
  the `galerkin` loss minimizes, so a perfectly trained label-free model matches it.

## Package layout

```
tensorpils/
├── meshing.py     # structured quad grid -> tensormesh.Mesh; grid<->node reshapes
├── physics.py     # FEMOperator base + PoissonProblem / WaveProblem / ACProblem (A, M, residuals)
├── multigrid.py   # geometric-multigrid V-cycle preconditioner (Poisson pls / precond Deep Ritz)
├── data.py        # create_datasets / create_wave_datasets / create_ac_datasets
├── models.py      # FNOModel (wraps neuralop.models.FNO)
├── losses.py      # Poisson (build_loss) + WaveGalerkinLoss / ACGalerkinLoss (build_*_loss)
├── optim.py       # build_optimizer
├── trainer.py     # BaseTrainer; PoissonTrainer; RolloutTrainer -> WaveTrainer / ACTrainer
├── viz.py         # loss curves / sample panels / (rollout) error distribution
└── cli.py         # argparse entry point (the `tensorpils` command), dispatch on --pde
```

Extend by: adding a loss (`nn.Module` + `build_*_loss` / `build_loss` branch); an optimizer
(`optim.build_optimizer`); or a **new PDE** — subclass `physics.FEMOperator` with a `residual`,
add a `create_*_datasets`, and either mirror `PoissonTrainer` (static) or subclass
`RolloutTrainer` (time-dependent, as wave and Allen–Cahn do).
