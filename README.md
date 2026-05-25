# TensorPILS

**Physics-Informed Learning System powered by [TensorMesh](https://github.com/camlab-ethz/TensorMesh).**

TensorPILS trains neural operators (currently a Fourier Neural Operator, FNO) to
solve PDEs, comparing four interchangeable loss functions — three of which need
**no labelled solutions** because they are built from TensorMesh's finite-element
machinery:

| Loss | Idea | Labels? |
|------|------|---------|
| `data` | supervised MSE against the analytical solution | yes |
| `galerkin` | weak-form residual ‖A u − b‖² on node values | no |
| `deepritz` | energy functional ∫(½\|∇u\|² − f u) dx (+ BC) | no |
| `pls` | preconditioned least-squares ½‖P(A u − b)‖², P ≈ A⁻¹ from a geometric-multigrid V-cycle | no |

The reference problem is 2D Poisson −Δu = f on the unit square with homogeneous
Dirichlet boundary conditions. The stiffness matrix `A`, mass matrix `M`, the
analytical source/solution, and the boundary handling all come from TensorMesh;
TensorPILS contributes the structured-grid bridge the FNO needs, the model, the
losses, the multigrid preconditioner, and the training loop.

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
# Galerkin weak-form residual loss (label-free)
tensorpils --loss galerkin --n_train 800 -k 4 --epochs 500

# Deep Ritz energy loss (label-free)
tensorpils --loss deepritz --n_train 800 -k 4 --epochs 500

# Supervised data loss (uses the analytical solution as ground truth)
tensorpils --loss data --n_train 800 -k 4 --epochs 500

# Preconditioned least-squares (geometric-multigrid-preconditioned Galerkin)
tensorpils --loss pls --n_train 800 -k 4 --epochs 500
```

Useful flags: `--bc_mode {penalty,hard}` (boundary handling for `data`/`deepritz`),
`--precondition` (multigrid-preconditioned Deep Ritz), `--grid_resolution`,
`--n_modes`, `--hidden_dim`, `--num_layers`, `--optimizer`, `--device`,
`--eval_only --checkpoint <path>`. Run `tensorpils --help` for the full list.

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

The supervised `data` loss and the reported test error use the **analytical**
multi-mode solution, generated through `tensormesh.dataset.PoissonMultiFrequency`
(with `r = -0.5`, matching the source `(i²+j²)^0.5` / solution `(i²+j²)^-0.5`
spectrum). The label-free losses (`galerkin`, `deepritz`, `pls`) instead drive
the FNO toward the discrete FEM solution `A⁻¹b` assembled by TensorMesh.

## Package layout

```
tensorpils/
├── meshing.py     # structured quad grid -> tensormesh.Mesh; grid<->node reshapes
├── physics.py     # PoissonProblem: wraps TensorMesh A & M; load/residual/energy
├── multigrid.py   # geometric-multigrid V-cycle preconditioner
├── data.py        # PoissonDataset + create_datasets (analytical via TensorMesh)
├── models.py      # FNOModel (wraps neuralop.models.FNO)
├── losses.py      # Data / Galerkin / DeepRitz / PLS losses + build_loss
├── optim.py       # build_optimizer
├── trainer.py     # Trainer + TrainingStats
├── viz.py         # loss curves / sample panels / error distribution
└── cli.py         # argparse entry point (the `tensorpils` command)
```

Add a new loss or optimizer by writing the class and registering it in
`build_loss` / `build_optimizer`; add a new PDE by mirroring `PoissonProblem`.
