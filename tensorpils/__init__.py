"""TensorPILS — preconditioned physics-informed neural operator training.

Code for *Preconditioned Physics-Informed Neural Operator Training*. Neural operators (FNO,
GAOT) are trained on the Poisson, Allen–Cahn and Stokes equations with loss functions built from
the finite-element machinery of TensorMesh: supervised data, the physics-informed least-squares
residual, and its preconditioned counterpart — the residual passed through a multigrid
preconditioner ``P ≈ A⁻¹`` (or, for the Stokes saddle point, weighted by a block-diagonal one).

The finite-element pieces (analytical data, stiffness/mass assembly, sparse solves) are
delegated to ``tensormesh``; this package owns the grid/mesh bridge the neural operators need,
the preconditioners, the losses, the baselines and the training loop.
"""

from ._version import __version__
from .meshing import (structured_quad_mesh, obstacle_mesh, topological_boundary_mask,
                      node_to_grid, grid_to_node)
from .physics import (FEMOperator, PoissonProblem, ACProblem, StokesProblem,
                      apply_zero_boundary)
from .preconditioners import (
    Preconditioner, GeometricMultigrid, AMGXPreconditioner, SpectralPreconditioner,
    StokesBlockPreconditioner, build_preconditioner,
)
from .data import (PoissonDataset, StreamingPoissonDataset, create_datasets,
                   create_scaling_datasets, ACDataset, create_ac_datasets, StokesDataset,
                   create_stokes_datasets)
from .losses import build_loss, build_ac_loss, build_stokes_loss
from .optim import build_optimizer
from .trainer import (BaseTrainer, PoissonTrainer, RolloutTrainer, ACTrainer, StokesTrainer,
                      TrainingStats)

# ``FNOModel`` lives in ``tensorpils.models`` and pulls in ``neuralop`` on import, and
# ``GAOTModel`` in ``tensorpils.gaot``; import them explicitly so that the FEM/training core
# stays importable without the heavy dependencies.

__all__ = [
    "__version__",
    "structured_quad_mesh",
    "obstacle_mesh",
    "topological_boundary_mask",
    "node_to_grid",
    "grid_to_node",
    "FEMOperator",
    "PoissonProblem",
    "ACProblem",
    "StokesProblem",
    "apply_zero_boundary",
    "Preconditioner",
    "GeometricMultigrid",
    "AMGXPreconditioner",
    "SpectralPreconditioner",
    "StokesBlockPreconditioner",
    "build_preconditioner",
    "PoissonDataset",
    "StreamingPoissonDataset",
    "create_datasets",
    "create_scaling_datasets",
    "ACDataset",
    "create_ac_datasets",
    "StokesDataset",
    "create_stokes_datasets",
    "build_loss",
    "build_ac_loss",
    "build_stokes_loss",
    "build_optimizer",
    "BaseTrainer",
    "PoissonTrainer",
    "RolloutTrainer",
    "ACTrainer",
    "StokesTrainer",
    "TrainingStats",
]
