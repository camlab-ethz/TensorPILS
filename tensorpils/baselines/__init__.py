"""Published physics-informed operator-learning baselines, for comparison only.

This subpackage exists so the paper can compare against *published* label-free operator
training rather than only against our own ``t=0`` negative control. Two baselines live here:

* **PINO** (:mod:`.pino`) — FNO + a **strong-form** PDE residual differentiated by finite
  differences on the grid, following ``neuraloperator/physics_informed``.
* **PI-DeepONet** (:mod:`.pi_deeponet`) — DeepONet + the same strong form, differentiated by
  **autodiff** through the trunk's coordinate input.

Both are scored by the repo's usual FEM relative-``L²`` metric with the eval-time boundary
projection, so their numbers drop straight into the existing tables.

Nothing in ``losses.py`` / ``trainer.py`` / ``physics.py`` is modified: the trainers here
subclass the production ones and the models are drop-in replacements for
:class:`~tensorpils.models.FNOModel`. Only ``cli.py`` gains dispatch.
"""

from .pino import (MollifiedModel, PINOPoissonLoss, PINOACLoss,
                   mollifier_grid, rel_lp)
from .deeponet import DeepONetModel
from .pi_deeponet import (PIDeepONetPoissonLoss, PIDeepONetACLoss,
                          autodiff_laplacian)
from .trainers import (PINOPoissonTrainer, PINOACTrainer,
                       PIDeepONetPoissonTrainer, PIDeepONetACTrainer,
                       DeepONetPoissonTrainer, DeepONetACTrainer)

__all__ = [
    "MollifiedModel", "PINOPoissonLoss", "PINOACLoss", "mollifier_grid", "rel_lp",
    "DeepONetModel",
    "PIDeepONetPoissonLoss", "PIDeepONetACLoss", "autodiff_laplacian",
    "PINOPoissonTrainer", "PINOACTrainer",
    "PIDeepONetPoissonTrainer", "PIDeepONetACTrainer",
    "DeepONetPoissonTrainer", "DeepONetACTrainer",
]
