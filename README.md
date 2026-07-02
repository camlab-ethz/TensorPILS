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

## Marius' Euler Install

Cluster-specific setup for the ETH **Euler** cluster, verified working (GPU run on an
RTX 2080 Ti). This deviates from the generic Install above: `pyproject.toml` requires
Python `>=3.10`, but Euler's `stack/.2024-04-silent` (used in the generic notes) only
ships Python 3.9.18. Use the `2024-05` stack, which provides a CUDA-enabled Python 3.11.

### One-time setup

```bash
# 1. Clone both repos as siblings in $HOME.
#    TensorMesh from the maintained camlab repo (public); TensorPILS is private (use a PAT).
cd ~
git clone https://github.com/camlab-ethz/TensorMesh.git
git clone https://github.com/Shizheng-Wen/TensorPILS.git
cd ~/TensorPILS && git checkout marius

# 2. Load the module stack that provides a CUDA-enabled Python >= 3.10.
conda deactivate                        # leave conda (base) if it is active
module purge
module load stack/2024-05 gcc/13.2.0    # deprecated/frozen but fine; exposes python/3.11.6_cuda
module load python/3.11.6_cuda
module load ffmpeg/6.0                  # NB: mesa-glu (generic README) is absent here and not needed
                                        #     (viz.py renders via matplotlib/Agg, no OpenGL)

# 3. Create and activate the virtual environment.
python -m venv ~/venvs/tensorgalerkin
source ~/venvs/tensorgalerkin/bin/activate
pip install --upgrade pip

# 4. Install TensorMesh (editable, from the clone) then TensorPILS.
pip install -e ~/TensorMesh
pip install -e ".[test]"                # run from ~/TensorPILS
```

Verify the install:

```bash
python -c "import torch, tensormesh, neuralop, tensorpils; print('torch', torch.__version__)"
tensorpils --help
```

### Reusable session script

Module loads and venv activation are **per-session** — they reset on every new login and
inside every batch job. The venv and the clones themselves are persistent. Save the setup
once so you never retype it:

```bash
cat > ~/env_tensorpils.sh << 'EOF'
#!/bin/bash
# TensorPILS session setup — `source ~/env_tensorpils.sh` each login / in job scripts
module purge
module load stack/2024-05 gcc/13.2.0
module load python/3.11.6_cuda
module load ffmpeg/6.0
source ~/venvs/tensorgalerkin/bin/activate
EOF
```

### Resume workflow (after reconnecting to Euler)

Each time you reconnect, the shell environment is empty; reload it and work on a **compute
node** (never train on the login node):

```bash
# 1. Log in and re-establish the environment.
ssh euler
source ~/env_tensorpils.sh              # re-loads modules + activates the venv

# 2. Request a compute node. Interactive GPU session for tests/debugging:
srun --time=00:20:00 --gpus=1 --mem-per-cpu=4G --pty bash
source ~/env_tensorpils.sh              # fresh shell on the node -> reload env

# 3. Run.
cd ~/TensorPILS
tensorpils --loss galerkin --n_train 16 --n_val 8 --n_test 8 -k 2 --epochs 2 --device cuda

# 4. Release the GPU when finished.
exit
```

For full unattended runs, submit a batch job with `sbatch` (a job script is still TODO)
rather than holding an interactive `srun` session.

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
├── meshing.py         # structured quad grid -> tensormesh.Mesh; grid<->node reshapes
├── physics.py         # PoissonProblem: wraps TensorMesh A & M; load/residual/energy
├── preconditioners/   # P ≈ A⁻¹ (shared Preconditioner interface + factory)
│   ├── base.py        #   Preconditioner ABC: forward(r)->Pr, report()
│   ├── multigrid.py   #   GeometricMultigrid V-cycle (computational path)
│   ├── spectral.py    #   SpectralPreconditioner: convex blend / fractional power
│   └── factory.py     #   build_preconditioner(kind, ...)
├── data.py            # PoissonDataset + create_datasets (analytical via TensorMesh)
├── models.py          # FNOModel (wraps neuralop.models.FNO)
├── losses.py          # Data / Galerkin / DeepRitz / PLS losses + build_loss
├── optim.py           # build_optimizer
├── trainer.py         # Trainer + TrainingStats
├── viz.py             # loss curves / sample panels / error distribution
└── cli.py             # argparse entry point (the `tensorpils` command)
```

Add a new loss or optimizer by writing the class and registering it in
`build_loss` / `build_optimizer`; add a new preconditioner by implementing the
`Preconditioner` interface and registering it in `build_preconditioner`; add a new
PDE by mirroring `PoissonProblem`.
