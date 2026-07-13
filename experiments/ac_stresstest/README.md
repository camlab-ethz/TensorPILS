# Experiment: Allen-Cahn loss stress-test (sweep in reaction strength eps)

**Story: physics-informed learning with the *correct* physics loss helps for Allen-Cahn.** In
`ac_loss_comparison/` all losses tied in the easy (diffusion-dominated) regime. Here we increase
the reaction strength `eps` (reaction coefficient `eps^2`), so the double-well engages, interfaces
sharpen (`delta ~ a/eps`), and the problem stiffens. As it does, the **minimizing-movement** loss
(the well-conditioned convex-concave objective) stays accurate while the **data-driven** and
**least-squares residual** losses fall apart.

Three losses (backward-Euler Galerkin dropped — it targets the BE operator, which drifts from the
CC reference), on the **same convex-concave reference** at each `eps`:

| # | training loss | flags |
|---|---|---|
| 0 | data-driven | `--loss data` |
| 1 | Galerkin (convex-concave) | `--loss galerkin --ac_loss_integrator convex_concave` |
| 2 | minimizing-movement | `--loss galerkin --ac_loss_form min_movement` |

Sweep: **`eps = 4, 8, 12, 16, 20, 24`** (6, equally spaced) x 3 losses = **18 runs** (2D SLURM
array), **100 epochs** for fast turnover. The `eps=4, 8, 16` runs can be reused from a prior
100-epoch sweep; only `eps=12, 20, 24` are new (see the submit note in `sweep.sbatch`).

## Facts

| | |
|---|---|
| PDE | Allen-Cahn, `a=1`, zero Dirichlet; reaction `eps^2 (u - u^3)` |
| Reference data | convex-concave (Eyre) FEM + Newton (sparse, float64 solve), `--ac_ref_chunk 1` |
| Fixed resolution | grid `64^2`, `dt=0.0025`, `n_steps=rollout_steps=10` (deliberately **not** scaled with `eps`) |
| Dataset | `K=4`, `n_train=1024`, `n_val=128`, `n_test=256`, `seed=42` |
| Optimizer / epochs | `adam`, cosine `1e-3->1e-4`, `100` epochs (fast turnover), batch `32` |
| Metric | best-model test space-time / final-time relative FEM-`L^2` (+ MSE), vs `eps` |
| Output dir | `output/ac_stresstest/` (git-ignored) |

Runs are tagged `fno_ac_{data|galerkin}_{ls|mm}_cc_..._eps{E}_...` so all `eps` x loss coexist.

## What is (and isn't) meaningful at large eps

The metric is always **NN vs the CC-FEM reference**, and CC is unconditionally energy-stable, so
the reference stays well-defined even where `dt`/mesh under-resolve the physics. So "can this loss
train the NN to reproduce the CC operator" is well-posed at every `eps`. Caveats:

- At large `eps` the CC reference is **under-resolved** (stiff, near-thresholding) — still a valid
  discrete target, but not physical Allen-Cahn.
- **Watch Newton convergence** in the data-gen logs at large `eps` (stiff SPD Jacobian, fp32).
- Backward-Euler Galerkin was dropped precisely to avoid the "BE vs CC operator" confound: BE and
  CC agree only to `O(dt)`, which becomes `O(1)` once `dt*eps^2 >~ 1`.

## Choosing `dt` and `h` with eps (back-of-envelope)

Steady interface `a^2 u'' = eps^2 (u^3 - u)` => `tanh` profile of width `delta = sqrt(2)*a/eps`.

- **Mesh:** resolve the interface => `h <~ delta` => `n = 1/h >~ eps/(sqrt(2)*a)`, i.e. `n ~ eps`
  (want `n >~ 3*eps` for a few points across the interface).
- **Time:** reaction rate at the wells is `2*eps^2` => timescale `~1/(2 eps^2)` => `dt ~ 1/eps^2`
  (want `dt*eps^2 <~ 0.5` for accuracy; CC has no hard stability limit, only accuracy).

At the fixed `dt=0.0025`, `n=64`, `a=1` used here:

| `eps` | `dt*eps^2` | `delta=sqrt(2)/eps` | pts/interface (`delta*64`) | resolved? |
|----:|-------:|-------:|-----:|:--|
| 4 | 0.04 | 0.35 | 23 | yes |
| 8 | 0.16 | 0.18 | 11 | yes |
| 12 | 0.36 | 0.12 | 7.5 | yes |
| 16 | 0.64 | 0.088 | 5.7 | borderline |
| 20 | 1.0 | 0.071 | 4.5 | under (time) |
| 24 | 1.4 | 0.059 | 3.8 | under (time) |

**Well-resolved cutoff ~ `eps=16`** (time binds first: `dt*eps^2` crosses ~1 near there); the plot
marks it with a dotted line. To stay resolved you would scale `dt = 0.0025*(16/eps)^2` and
`n = 64*(eps/16)` for `eps>16` — a more expensive experiment left for later. This sweep
intentionally holds resolution fixed, so the errors rising past `eps~16` reflect the fixed
resolution as much as the loss — the *relative* ordering of losses is the point.

## Run (Euler)

```bash
mkdir -p logs
# only the NEW eps=12,20,24 (reuse eps=4,8,16 from a prior 100-epoch sweep):
sbatch --array=6-8,12-17 experiments/ac_stresstest/sweep.sbatch
# ...or the full 18-task array if starting fresh:
# sbatch experiments/ac_stresstest/sweep.sbatch
squeue --me
```

## Plot

```bash
python experiments/ac_stresstest/plot_ac_stresstest.py \
    --results_dir output/ac_stresstest/results --eps_min 4 --eps_max 24
    # --metric {st_rel_l2,final_rel_l2,mse}
```

`--eps_min/--eps_max` restrict to the equally-spaced grid (dropping any leftover `eps=2, 32` JSONs
from an earlier sweep). Produces `output/ac_stresstest/ac_stresstest_{metric}.png` — test error vs
`eps` on a **linear** axis, one curve per training loss (dotted line at the `eps~16` resolution
cutoff).
