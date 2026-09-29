#!/usr/bin/env bash
# Figure 1: Adam on the physics-informed least-squares loss of a linear model (a plain vector of
# nodal values, no neural network), without and with a multigrid preconditioner, Poisson on the
# 65 x 65 grid: 4000 Adam steps at learning rate 1e-3, cosine-annealed to 1e-5. Two optimisation
# runs on a CPU, then the figure.
#
# Run from the repository root:  bash experiments/graphical_abstract/run.sh [task|list]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

OUT=${OUT:-output/graphical_abstract}
DIR=experiments/graphical_abstract
RECORD="--grid 65 --steps 4000 --record every --lr 1e-3 --lr_min 1e-5"

TASKS=(
    "$PY $DIR/run_landscape.py $RECORD --precond none --out $OUT/bare.npz"
    "$PY $DIR/run_landscape.py $RECORD --precond multigrid --mg_omega 0.8888888888888888 --skip_gd --out $OUT/mg.npz"
    "$PY $DIR/plot_abstract.py --bare $OUT/bare.npz --mg $OUT/mg.npz --out $OUT/graphical_abstract.pdf"
)

run_tasks "$@"
