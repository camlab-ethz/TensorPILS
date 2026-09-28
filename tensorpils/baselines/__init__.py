"""Published physics-informed operator-learning baselines, for comparison only.

Two baselines live here:

* **PINO** (:mod:`.pino`) — FNO + a **strong-form** PDE residual differentiated by finite
  differences on the grid, following ``neuraloperator/physics_informed``. Its Dirichlet BC is
  imposed by zeroing the boundary nodes, matching our arms, rather than by the reference's
  mollifier — see :class:`~tensorpils.baselines.pino.ZeroBoundaryModel`. Poisson and
  Allen–Cahn (a finite-difference stencil needs a grid).
* **PI-DeepONet** (:mod:`.pi_deeponet`) — DeepONet + the same strong form, differentiated by
  **autodiff** through the trunk's coordinate input. Poisson, Allen–Cahn and Stokes on the
  obstacle mesh (momentum + continuity).

Both are scored by the repo's usual FEM relative-``L²`` metric with the eval-time boundary
projection: the trainers here subclass the production ones and only replace the training
objective, and the models are drop-in replacements for :class:`~tensorpils.models.FNOModel`.
"""

from .pino import (MollifiedModel, ZeroBoundaryModel, PINOPoissonLoss, PINOACLoss,
                   mollifier_grid, boundary_mask_grid, rel_lp)
from .deeponet import DeepONetModel
from .pi_deeponet import (PIDeepONetPoissonLoss, PIDeepONetACLoss, PIDeepONetStokesLoss,
                          autodiff_laplacian, autodiff_stokes, stokes_reduce)
from .trainers import (PINOPoissonTrainer, PINOACTrainer,
                       PIDeepONetPoissonTrainer, PIDeepONetACTrainer, PIDeepONetStokesTrainer)

__all__ = [
    "MollifiedModel", "ZeroBoundaryModel", "PINOPoissonLoss", "PINOACLoss",
    "mollifier_grid", "boundary_mask_grid", "rel_lp",
    "DeepONetModel",
    "PIDeepONetPoissonLoss", "PIDeepONetACLoss", "PIDeepONetStokesLoss",
    "autodiff_laplacian", "autodiff_stokes", "stokes_reduce",
    "PINOPoissonTrainer", "PINOACTrainer",
    "PIDeepONetPoissonTrainer", "PIDeepONetACTrainer", "PIDeepONetStokesTrainer",
]
