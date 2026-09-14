"""Published physics-informed operator-learning baselines, for comparison only.

This subpackage exists so the paper can compare against *published* label-free operator
training rather than only against our own ``t=0`` negative control. Two baselines live here:

* **PINO** (:mod:`.pino`) — FNO + a **strong-form** PDE residual differentiated by finite
  differences on the grid, following ``neuraloperator/physics_informed``. Its Dirichlet BC is
  imposed by zeroing the boundary nodes, matching our arms, rather than by the reference's
  mollifier — see :class:`~tensorpils.baselines.pino.ZeroBoundaryModel`.
* **PI-DeepONet** (:mod:`.pi_deeponet`) — DeepONet + the same strong form, differentiated by
  **autodiff** through the trunk's coordinate input.

Both are scored by the repo's usual FEM relative-``L²`` metric with the eval-time boundary
projection, so their numbers drop straight into the existing tables. Each exists for Poisson,
Allen–Cahn and (strong-form momentum + continuity on the velocity grid) Stokes.

Nothing in ``losses.py`` / ``trainer.py`` / ``physics.py`` is modified: the trainers here
subclass the production ones and the models are drop-in replacements for
:class:`~tensorpils.models.FNOModel`. Only ``cli.py`` gains dispatch.
"""

from .pino import (MollifiedModel, ZeroBoundaryModel, PINOPoissonLoss, PINOACLoss,
                   PINOStokesLoss, mollifier_grid, boundary_mask_grid, rel_lp, stokes_reduce)
from .deeponet import DeepONetModel
from .pi_deeponet import (PIDeepONetPoissonLoss, PIDeepONetACLoss, PIDeepONetStokesLoss,
                          autodiff_laplacian, autodiff_stokes)
from .trainers import (PINOPoissonTrainer, PINOACTrainer, PINOStokesTrainer,
                       PIDeepONetPoissonTrainer, PIDeepONetACTrainer, PIDeepONetStokesTrainer,
                       DeepONetPoissonTrainer, DeepONetACTrainer, DeepONetStokesTrainer)

__all__ = [
    "MollifiedModel", "ZeroBoundaryModel", "PINOPoissonLoss", "PINOACLoss", "PINOStokesLoss",
    "mollifier_grid", "boundary_mask_grid", "rel_lp", "stokes_reduce",
    "DeepONetModel",
    "PIDeepONetPoissonLoss", "PIDeepONetACLoss", "PIDeepONetStokesLoss",
    "autodiff_laplacian", "autodiff_stokes",
    "PINOPoissonTrainer", "PINOACTrainer", "PINOStokesTrainer",
    "PIDeepONetPoissonTrainer", "PIDeepONetACTrainer", "PIDeepONetStokesTrainer",
    "DeepONetPoissonTrainer", "DeepONetACTrainer", "DeepONetStokesTrainer",
]
