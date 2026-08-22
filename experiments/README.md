# Experiments

Experiments are grouped by PDE: **`poisson/`**, **`allen_cahn/`** and **`stokes/`**, plus
**`baselines/`** (published methods to compare against, grouped by PDE inside). Each leaf subfolder is one
self-contained experiment: a `README.md` (description, facts, run & plot commands), the SLURM
`*.sbatch` sweep script, and the aggregation plot script. Run artifacts land in
`output/<domain>/<name>/` (git-ignored); pull them back from the cluster with
`rsync -avz euler:~/TensorPILS/output/<domain>/<name> ~/Documents/TensorPILS/output/<domain>/`.

### `poisson/`

| Experiment | Question |
|---|---|
| [`sweep_blend/`](poisson/sweep_blend/README.md) | Does the blend preconditioner strength `t` (residual `t=0` → supervised `t=1`) control training conditioning? Overlay + collapse plots vs `κ(H)`. Variants by loss: `pls/`, `deepritz/`. |
| [`generalization/`](poisson/generalization/README.md) | Does that loss geometry `t` affect out-of-distribution generalization? Organized into per-regime **scenarios** (`pls_k4/`: K=4→6,8; `pls_k16/`: K=16→20; Deep Ritz variants planned). |
| [`loss_comparison/`](poisson/loss_comparison/README.md) | Supervised true-L2 (`data_l2`, mass-weighted) vs MSE (`data`) vs label-free Deep Ritz — which training objective minimizes validation relative-L2? |
| [`data_scaling/`](poisson/data_scaling/README.md) | At a fixed optimization budget, how does test error scale with dataset size — and does the infinite-data (streaming) limit, which eliminates the estimation error exactly, set the floor? |

### `allen_cahn/`

| Experiment | Question |
|---|---|
| [`ac_loss_comparison/`](allen_cahn/ac_loss_comparison/README.md) | Data-driven MSE vs backward-Euler / convex-concave Galerkin vs minimizing-movement — FEM-L2 rollout metrics for the learned Allen–Cahn stepper. |
| [`ac_stresstest/`](allen_cahn/ac_stresstest/README.md) | How far up the reaction-stiffness `ε` axis does each loss hold before the free-running rollout error blows up? |
| [`compare_bptt/`](allen_cahn/compare_bptt/README.md) | How does the BPTT autodiff mode (full BPTT / detach-prev / pushforward) affect FNO rollout training across the three AC losses over the `ε` axis? |
| [`long_rollout/`](allen_cahn/long_rollout/README.md) | Extrapolation past the 10-step training horizon: energy + error vs the convex-concave reference, plus 2D field heatmaps. |
| [`realistic_ac/`](allen_cahn/realistic_ac/README.md) | A realistic FEM regime (`256²`, `eps=64`, `dt=0.01`): with the Newton-Jacobian conditioning `κ~10⁴`, do data-driven / minimizing-movement / **bare** least-squares (which sees `κ²~10⁸`) survive a 100-step pushforward rollout? Error + energy + heatmaps, test & train. Preconditioned-LS arm slots in later. |
| [`realistic_ac_K8/`](allen_cahn/realistic_ac_K8/README.md) | Same as `realistic_ac` (128²/eps=32/dt=0.01, 4 loss arms) but with `K=8` ICs — the intermediate difficulty between the working K=4 and the non-converging K=16. Separate output tree; reuses the `realistic_ac` analysis scripts. |
| [`realistic_ac_K16/`](allen_cahn/realistic_ac_K16/README.md) | Same as `realistic_ac` (128²/eps=32/dt=0.01, 4 loss arms) but with `K=16` multi-frequency ICs — a much harder learning problem. Separate output tree; reuses the `realistic_ac` analysis scripts. |

### `stokes/`

The saddle-point case, where the Poisson recipe (`P ≈ A⁻¹` applied *inside* the square) does not
carry over, because no block-diagonal `P` approximates `K⁻¹`. The loss form is then **determined by
the preconditioner**, not chosen: a block-diagonal `P` is SPD but not an inverse, so it can only be a
**norm weight** (`½rᵀPr`); a monolithic V-cycle *is* an inverse but inherits `K`'s indefiniteness, so
it can only be **applied** (`½‖Pr‖²`). Mismatching them diverges. Velocity and pressure are always
scored separately, each in its own FE-`L²` norm.

| Experiment | Question |
|---|---|
| [`conditioning/`](stokes/conditioning/README.md) | **No training.** Measure in float64 what the theory claims: `κ(K²)=O(h⁻⁴)` vs `κ(KPK)=O(h⁻²)`, and the `O(1)` eigenvalue ratio underneath both. Includes an exact-block check against the note's `{1,(1±√5)/2}` and an `omega_scan.py` that finds `schur_omega` 30× off its optimum. |
| [`h_refinement/`](stokes/h_refinement/README.md) | The learning side of the same claim: 3 losses × 3 velocity grids (`33/65/129`) at a fixed budget — does the bare-residual gap widen under refinement? The `65²` column reproduces the paper's headline table. |
| [`blend_sweep/`](stokes/blend_sweep/README.md) | Sweep `P_t=(1−t)αI+tP` from the bare loss (`t=0`) to the block preconditioner (`t=1`) and collapse the final error onto the *measured* `κ(KP_tK)` — the Stokes analogue of `poisson/sweep_blend/`. |
| [`monolithic/`](stokes/monolithic/README.md) | The note's monolithic multigrid with a symmetric Uzawa smoother: a genuine `P ≈ K⁻¹`, `O(1)` conditioning instead of `O(h⁻²)`. Also the sharpest result here — `O(1)` conditioning alone makes training *worse*, and fixing the loss **metric** is what turns it into the best label-free arm (19.34 % → 4.33 % velocity). |

### `baselines/`

Published physics-informed operator methods, so the paper's "vastly outperforms previous
approaches" claim rests on something other than our own negative control. Both are label-free and
both are scored by the same FEM relative-`L2` metric as every other table. See
[`baselines/README.md`](baselines/README.md) for why PINO is ported with **finite differences**
(its Dirichlet example, Darcy, uses them, while its FFT examples are periodic) and for the
measurement showing the FD and FEM residuals are the same PDE at the same `O(h²)` accuracy.

| Experiment | Question |
|---|---|
| [`baselines/poisson/`](baselines/poisson/README.md) | PINO (FNO + FD strong form) and PI-DeepONet (DeepONet + autodiff strong form) against supervised, bare-FEM and preconditioned-FEM training at `64²`. Includes the supervised-DeepONet control and the "our loss, other architecture" cell. |
| [`baselines/allen_cahn/`](baselines/allen_cahn/README.md) | The same comparison for a nonlinear time-dependent problem, with the task held fixed as the autoregressive one-step stepper so wins/losses stay attributable to the objective. |

## Evaluation convention

Every experiment reports the FEM **relative-L2** error on the validation set (mass-matrix
weighted). At eval the prediction's boundary is **always projected to zero** — the
homogeneous Dirichlet BC is known data, so it is enforced for *every* loss (see
`Trainer.eval_project_bc` in `tensorpils/trainer.py`). This is eval-only (training is
unaffected) and makes losses comparable regardless of how — or whether — they constrained
the boundary during training.

> Note: results produced before this convention was made global (2026-07-07) understate
> nothing for the hard-BC/preconditioned Deep Ritz and `data_h1` curves (which already
> projected), but the `data`/`data_l2`/`deepritz`-penalty/`pls` curves must be re-run to
> pick up the boundary projection.

Theory write-up for the spectral preconditioners: `../notes/preconditioner_notes/`.

## Inspecting generated data (Wave / Allen–Cahn)

Before training a time-dependent experiment — or when changing the data generator itself (e.g.
swapping the Allen–Cahn integrator for a convex–concave splitting) — eyeball the reference
trajectories straight from the dataset, with **no model** involved:

```python
from tensorpils.data import ACDataset          # or WaveDataset
from tensorpils import viz

ds = ACDataset(num_samples=4, K=4, seed=0, n_steps=20, dt=1e-3, a=1.0, eps=2.0)
viz.visualize_data_trajectory(ds, sample_idx=0)   # -> output/data_viz/traj_sample0.png
```

`viz.visualize_data_trajectory(dataset, sample_idx=0, n_frames=5, save_path=None, title=None)`
renders a filmstrip of the trajectory (shared colour scale) plus a `max|u|` / `‖u‖₂`-vs-time
panel and a boundary-leak readout, and returns the resolved save path. It defaults to
`output/data_viz/traj_sample{idx}.png` (git-ignored). To compare two integrators, give each a
distinct `save_path=`/`title=`; the same `sample_idx`+`seed` share one initial condition, so any
difference in the filmstrip or the `max|u|`/`‖u‖₂` curves is purely the integrator.
