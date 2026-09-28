# Ill-conditioning defeats first-order optimisers

Reproduces **Figure 1**. The setting is stripped to its skeleton, with no neural operator at all:
the physics-informed least-squares loss of the Poisson problem over a plain vector of nodal values,

    L(u) = ½ ‖A u − b‖²,     b = M f   (consistent finite-element load),

minimised with Adam, without and with a multigrid preconditioner (`½ ‖P(A u − b)‖²`, `P` one
geometric V-cycle). The Hessian of the bare loss is `A²`, so its condition number grows like
`h⁻⁴`; on the 65 × 65 grid this alone defeats Adam, before any network is involved.

The left and centre panels show the error in every eigenmode of `A` along the iterations, the
right panel the relative L² error.

## Setup

- Bilinear finite elements on the uniform grid, the sources of the Poisson experiments
  (`K = 4`), initial guess `u₀ = 0` (so the initial error `−u*` is entirely smooth).
- On a uniform grid the 2D discrete sine modes are exact eigenvectors of `A`, so every modal
  projection comes from one discrete sine transform; `run_landscape.py` verifies this at start-up.
- Adam runs in float32; every reported error is computed in float64 against a direct solve. The
  learning rate is tuned over nine values at a budget of `10⁴` steps, then the run takes `10⁵`.

## Run

From the repository root (CPU, a few minutes per run):

```bash
bash experiments/graphical_abstract/run.sh     # -> output/graphical_abstract/graphical_abstract.pdf
```
