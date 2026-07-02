"""TensorPILS — Physics-Informed Learning System powered by TensorMesh.

Train neural operators (FNO) on PDEs using losses built from TensorMesh's
finite-element machinery: supervised data, Galerkin weak-form residual,
Deep Ritz energy, and a multigrid-preconditioned least-squares loss.

The finite-element pieces (analytical solutions, stiffness/mass assembly,
boundary conditions) are delegated to ``tensormesh``; this package owns the
structured-grid bridge the FNO needs, the neural model, the losses, and the
training loop.
"""

from ._version import __version__
from .meshing import structured_quad_mesh, node_to_grid, grid_to_node
from .physics import PoissonProblem, apply_zero_boundary
from .preconditioners import (
    Preconditioner, GeometricMultigrid, SpectralPreconditioner, build_preconditioner,
)
from .data import PoissonDataset, create_datasets
from .losses import build_loss
from .optim import build_optimizer
from .trainer import Trainer, TrainingStats

# ``FNOModel`` lives in ``tensorpils.models`` and pulls in ``neuralop`` on
# import; import it explicitly (``from tensorpils.models import FNOModel``)
# so that the FEM/training core stays importable without the heavy dependency.

__all__ = [
    "__version__",
    "structured_quad_mesh",
    "node_to_grid",
    "grid_to_node",
    "PoissonProblem",
    "apply_zero_boundary",
    "Preconditioner",
    "GeometricMultigrid",
    "SpectralPreconditioner",
    "build_preconditioner",
    "PoissonDataset",
    "create_datasets",
    "build_loss",
    "build_optimizer",
    "Trainer",
    "TrainingStats",
]
