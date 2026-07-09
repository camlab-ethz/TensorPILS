# Experiments

Each subfolder is one self-contained experiment: a `README.md` (description, facts, run &
plot commands), the SLURM `*.sbatch` sweep script, and the aggregation plot script. Run
artifacts land in `output/<name>/` (git-ignored); pull them back from the cluster with
`rsync -avz euler:~/TensorPILS/output/<name> ~/Documents/TensorPILS/output/`.

| Experiment | Question |
|---|---|
| [`sweep_blend/`](sweep_blend/README.md) | Does the blend preconditioner strength `t` (residual `t=0` → supervised `t=1`) control training conditioning? Overlay + collapse plots vs `κ(H)`. Variants by loss: `pls/`, `deepritz/`. |
| [`generalization/`](generalization/README.md) | Does that loss geometry `t` affect out-of-distribution generalization? Organized into per-regime **scenarios** (`pls_k4/`: K=4→6,8; `pls_k16/`: K=16→20; Deep Ritz variants planned). |
| [`loss_comparison/`](loss_comparison/README.md) | Supervised true-L2 (`data_l2`, mass-weighted) vs MSE (`data`) vs label-free Deep Ritz — which training objective minimizes validation relative-L2? |

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
