# TensorPILS

Code for **Preconditioned Physics-Informed Neural Operator Training**
by Shizheng Wen\*, Marius Zeinhofer\* and Siddhartha Mishra (ETH Zürich; \*equal contribution).

Physics-informed losses let a neural operator learn from the governing equations alone, without
a dataset of solutions, but they are badly conditioned: for the finite-element residual
`L_LS(u) = ½‖Au − b‖²` the Hessian is `A²`, whose condition number grows like `h⁻⁴` under mesh
refinement. TensorPILS trains neural operators with the **preconditioned least-squares loss**

    L_PLS(u) = ½ ‖P (A u − b)‖²,     P ≈ A⁻¹  (one multigrid V-cycle),

whose conditioning is independent of the mesh for elliptic problems. The same construction covers
nonlinear time stepping (Allen–Cahn) and, as a block-diagonal norm weight
`½ rᵀ P r`, the Stokes saddle point on an unstructured mesh. It needs no labels, is agnostic to
the neural operator (FNO or GAOT), and adds no cost at inference.

The name stands for physics-informed least squares on [TensorMesh](https://github.com/camlab-ethz/TensorMesh),
the differentiable finite-element library that assembles every operator used here.

## Installation

Python ≥ 3.10 on Linux with an NVIDIA GPU:

```bash
git clone https://github.com/camlab-ethz/TensorPILS.git
cd TensorPILS
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[test]"
pip install https://github.com/sparsexlab/torch-amgx/releases/download/v0.1.0a14/torch_amgx-0.1.0a14-0_cu128_torch211-cp311-cp311-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl
```

The second command pulls in [`tensormesh-fem`](https://pypi.org/project/tensormesh-fem/) ≥ 0.2.1
(finite elements, import name `tensormesh`) and
[`neuraloperator`](https://github.com/neuraloperator/neuraloperator) ≥ 2.0 (the FNO, import name
`neuralop`).

**PyTorch and torch-amgx.** Any PyTorch ≥ 2.0 runs the Poisson and Allen–Cahn experiments. The
Stokes preconditioner applies its velocity block with NVIDIA AmgX through
[`torch-amgx`](https://github.com/sparsexlab/torch-amgx), which is not on PyPI and whose wheels are
built for particular PyTorch and CUDA versions (release v0.1.0a14: PyTorch 2.11 with CUDA 12.6 or
12.8, or PyTorch 2.6 with CUDA 12.4), hence the pinned PyTorch above. The wheel shown is for
Python 3.11; the [releases page](https://github.com/sparsexlab/torch-amgx/releases) has one for each
Python version. Without torch-amgx everything except the Stokes `L_PLS` runs, and the tests that
need it are skipped.

**Gmsh.** The obstacle mesh of the Stokes experiments is generated with Gmsh, whose Python wheel
needs the system OpenGL utility library (`libGLU.so.1`, e.g. `apt install libglu1-mesa`).

**Optional.** `torch-scatter` and `torch-cluster` speed up the graph operations of GAOT; without
them a pure-PyTorch fallback is used.

The commands above were tested in a clean environment with Python 3.11 on an NVIDIA RTX 4090, where
`pytest` passes (the four tests of the optional GAOT extensions are skipped).

## Quick start

Every run is one call of the command-line interface (`python -m tensorpils.cli`, or the
equivalent `tensorpils` command). Poisson on the 65 × 65 grid with the preconditioned loss:

```bash
python -m tensorpils.cli --pde poisson --model fno --loss pls --mg_omega 0.8888888888888888 \
    -k 10 --grid_resolution 65 --n_train 1024 --epochs 500 --lr 3e-4 --lr_min 3e-5
```

`--loss galerkin` trains the unpreconditioned residual `L_LS` and `--loss data` the supervised
baseline; `--loss pino` and `--model deeponet --loss pideeponet` are the PINO and physics-informed
DeepONet baselines, whose exact settings for each table row are in the launchers of
[`experiments/`](experiments/README.md). The other two problems of the paper:

```bash
# Allen-Cahn: one time step of the convex-concave scheme, residual preconditioned by a V-cycle
python -m tensorpils.cli --pde ac --model fno --loss galerkin --ac_precond multigrid \
    --ac_eps 32 -k 4 --grid_resolution 129 --dt 0.01 --n_steps 10 --rollout_steps 10 \
    --batch_size 16 --lr 3e-4 --lr_min 3e-5

# Stokes flow around a circular obstacle, P2/P1 elements on an unstructured mesh:
# GAOT with the block preconditioner
python -m tensorpils.cli --pde stokes --model gaot --loss pls --schur_omega 16 \
    --mesh_h 0.035 -k 10 --lr 1e-3 --lr_min 1e-6
```

Each run writes to `--output_dir` (default `output/`): `results/<run>.json` with the configuration,
the per-epoch statistics and the test errors, the best-validation checkpoint in `checkpoints/`,
and diagnostic plots. `python -m tensorpils.cli --help` lists every option.

## Reproducing the paper

[`experiments/`](experiments/README.md) holds one directory per experiment, each with a README
(setup, commands, expected numbers, compute), the launchers and the scripts that turn the runs
into the paper's tables and figures:

| Paper | Directory |
|---|---|
| Figure 1 (Adam on a linear model) | [`experiments/graphical_abstract`](experiments/graphical_abstract/README.md) |
| Table 1, Poisson | [`experiments/poisson/benchmark`](experiments/poisson/benchmark/README.md) |
| Figure 2 (conditioning sweep) | [`experiments/poisson/blend_sweep`](experiments/poisson/blend_sweep/README.md) |
| Figure 3 (infinite-data limit) | [`experiments/poisson/infinite_data`](experiments/poisson/infinite_data/README.md) |
| Table 1, Allen–Cahn; Figure 4 | [`experiments/allen_cahn/benchmark`](experiments/allen_cahn/benchmark/README.md) |
| Table 1, Stokes; Figure 5; preconditioner ablation | [`experiments/stokes/benchmark`](experiments/stokes/benchmark/README.md) |
| Backbone comparison (appendix) | [`experiments/poisson/backbone`](experiments/poisson/backbone/README.md) |

For example, the Poisson rows of Table 1:

```bash
bash experiments/poisson/benchmark/run.sh          # 15 runs: 5 methods x 3 seeds
python experiments/poisson/benchmark/summarize.py
```

Table 1 of the paper (test relative L² error in %, mean ± half-range over three seeds):

| PDE | `L_PLS` (ours) | `L_data` (supervised) | PINO | PI-DeepONet |
|---|---|---|---|---|
| Poisson | 5.1 ± 0.8 | 4.7 ± 0.8 | 33.1 ± 1.1 | 21.4 ± 4.7 |
| Allen–Cahn | 3.2 ± 0.3 | 2.0 ± 0.2 | 5.3 ± 1.6 | 90.0 ± 1.7 |
| Stokes, velocity | 1.6 ± 0.2 | 3.3 ± 0.2 | — | 39.7 ± 2.1 |
| Stokes, pressure | 0.85 ± 0.04 | 1.79 ± 0.16 | — | 36.7 ± 0.6 |

Training on a GPU is not bitwise reproducible, so a rerun agrees with these numbers within the
spread over seeds rather than digit for digit (for the occasional loss spike of Stokes `L_PLS`, see
[`experiments/stokes/benchmark`](experiments/stokes/benchmark/README.md)).

## Using the loss in your own code

The finite-element operators, the preconditioners and the losses are ordinary PyTorch modules.
A neural operator trained with `L_PLS` on the Poisson problem:

```python
import torch
from tensorpils import (structured_quad_mesh, PoissonProblem, build_preconditioner,
                        build_loss, grid_to_node)
from tensorpils.models import FNOModel

n, device = 65, "cuda"
problem = PoissonProblem(structured_quad_mesh(n, n)).to(device)     # Q1 stiffness A, mass M
P = build_preconditioner("multigrid", problem, grid_size=(n, n),   # one V-cycle, P ≈ A⁻¹
                         mg_omega=8 / 9, device=device)
loss_fn = build_loss("pls", problem, precond=P)                     # ½‖P(Au − Mf)‖²

model = FNOModel(n_modes=(16, 16), hidden_channels=64).to(device)
f = torch.randn(8, 1, n, n, device=device)                          # a batch of sources
u = model(f)                                                        # [8, 1, n, n]
loss = loss_fn(grid_to_node(u[:, 0], n, n), grid_to_node(f[:, 0], n, n))  # node values, no labels
loss.backward()
```

The losses sum over nodes and average over the batch. Any object with the
`Preconditioner` interface (`forward(r) -> P r`) can take the place of the V-cycle.

## Repository layout

```
tensorpils/
├── meshing.py          structured quad grids, the obstacle mesh, grid <-> node ordering
├── physics.py          finite-element operators: Poisson, Allen-Cahn, Stokes (via TensorMesh)
├── preconditioners/    geometric multigrid, AmgX algebraic multigrid, spectral blend, Stokes block
├── losses.py           L_data, L_LS, L_PLS for each problem
├── data.py             datasets: random sources, FEM reference solutions
├── models.py           FNO (from neuraloperator)
├── gaot/               GAOT, the geometry-aware operator transformer (vendored)
├── baselines/          PINO, DeepONet and physics-informed DeepONet
├── trainer.py          training loops for the static and the time-stepping problems
└── cli.py              command-line interface
experiments/            launchers and analysis scripts of every table and figure
tests/                  unit tests (pytest)
```

The tests run with `pytest`; those of the AmgX preconditioner are skipped when `torch-amgx` or a
GPU is unavailable.

GAOT in `tensorpils/gaot/` is transcribed from [camlab-ethz/GAOT](https://github.com/camlab-ethz/GAOT)
with the changes listed in its module docstring, so that a single environment runs every
experiment.

## Citation

```bibtex
@article{wen2026preconditioned,
  title   = {Preconditioned Physics-Informed Neural Operator Training},
  author  = {Wen, Shizheng and Zeinhofer, Marius and Mishra, Siddhartha},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

Apache License 2.0, see [LICENSE](LICENSE).
