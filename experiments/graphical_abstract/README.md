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
- Adam runs in float32 for 4000 steps, with learning rate `1e-3` cosine-annealed to `1e-5`, the
  same for both losses; every reported error is computed in float64 against a direct solve.
  These settings were not recorded with the paper's figure and are reconstructed from it: they
  reproduce its curves (the drop of `L_PLS` to about `2·10⁻⁶` within 200 steps, `L_LS` falling
  from 1 to about 0.55). Without `--lr`, `run_landscape.py` tunes the learning rate itself.

## Run

From the repository root (CPU, a few minutes per run):

```bash
bash experiments/graphical_abstract/run.sh     # -> output/graphical_abstract/graphical_abstract.pdf
```
