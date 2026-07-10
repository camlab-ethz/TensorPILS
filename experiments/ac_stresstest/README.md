# Experiment: Allen-Cahn loss stress-test (sweep in reaction strength eps)

In `ac_loss_comparison/` all four losses converged together in the easy (diffusion-dominated)
regime. This experiment pushes toward a **hard** regime by increasing the reaction strength
`eps` — recall the reaction coefficient is `eps^2`, so `eps=256` means `65536`. As `eps` grows the
double-well engages, interfaces sharpen (`delta ~ a/eps`), and the problem stiffens.

Three losses (backward-Euler Galerkin dropped — it targets the BE operator, which drifts from the
CC reference), on the **same convex-concave reference** at each `eps`:

| # | training loss | flags |
|---|---|---|
| 0 | data-driven | `--loss data` |
| 1 | Galerkin (convex-concave) | `--loss galerkin --ac_loss_integrator convex_concave` |
| 2 | minimizing-movement | `--loss galerkin --ac_loss_form min_movement` |

Sweep: **`eps = 2, 4, 8, 16, 32`** (5) x 3 losses = **15 runs** (2D SLURM array), **100 epochs**
for fast turnover (run longer / larger `eps` overnight).

## Facts

| | |
|---|---|
| PDE | Allen-Cahn, `a=1`, zero Dirichlet; reaction `eps^2 (u - u^3)` |
| Reference data | convex-concave (Eyre) FEM + Newton, `--ac_newton_tol 1e-6`, `--ac_ref_chunk 1` (fp32) |
| Fixed resolution | grid `64^2`, `dt=0.0025`, `n_steps=rollout_steps=10` (deliberately **not** scaled with `eps`) |
| Dataset | `K=4`, `n_train=1024`, `n_val=128`, `n_test=256`, `seed=42` |
| Optimizer / epochs | `adam`, cosine `1e-3->1e-4`, `100` epochs (fast turnover), batch `32` |
| Metric | best-model test space-time / final-time relative FEM-`L^2` (+ MSE), vs `eps` |
| Output dir | `output/ac_stresstest/` (git-ignored) |

Runs are tagged `fno_ac_{data|galerkin}_{ls|mm}_{cc|be}_..._eps{E}_...` so all 32 coexist.

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
| 2 | 0.01 | 0.71 | 45 | yes |
| 4 | 0.04 | 0.35 | 23 | yes |
| 8 | 0.16 | 0.18 | 11 | yes |
| 16 | 0.64 | 0.088 | 5.7 | borderline |
| 32 | 2.6 | 0.044 | 2.8 | under (time) |
| 64 | 10 | 0.022 | 1.4 | under (both) |
| 128 | 41 | 0.011 | 0.7 | interface < mesh |
| 256 | 164 | 0.0055 | 0.35 | far under |

**Well-resolved cutoff ~ `eps=16`** (time binds first); the plot marks it with a dotted line. To
stay resolved you would scale `dt = 0.0025*(16/eps)^2` and `n = 64*(eps/16)` for `eps>16` — a much
more expensive experiment left for later. This sweep intentionally holds resolution fixed.

## Run (Euler)

```bash
mkdir -p logs
sbatch experiments/ac_stresstest/sweep.sbatch      # 32-task array (0-31), capped at 8 concurrent
squeue --me
```

## Plot

```bash
python experiments/ac_stresstest/plot_ac_stresstest.py \
    --results_dir output/ac_stresstest/results       # --metric {st_rel_l2,final_rel_l2,mse}
```

Produces `output/ac_stresstest/ac_stresstest_{metric}.png` — test error vs `eps`, one curve per
training loss (log-log; dotted line at the `eps~16` resolution cutoff).
