# Experiment: Allen–Cahn loss comparison

Which training objective best teaches an FNO the Allen–Cahn one-step map? Compare four losses,
all trained on and evaluated against the **same convex–concave reference dataset** (so the
rollout-MSE numbers are directly comparable):

| # | training loss | flags | labels? |
|---|---|---|---|
| 0 | data-driven | `--loss data` | supervised |
| 1 | Galerkin (backward-Euler) | `--loss galerkin --ac_loss_integrator backward_euler` | label-free |
| 2 | Galerkin (convex–concave) | `--loss galerkin --ac_loss_integrator convex_concave` | label-free |
| 3 | minimizing-movement | `--loss galerkin --ac_loss_form min_movement` | label-free |

Arms 1–3 are the three physics losses from `notes/ac_autoregressive/`: the backward-Euler and
convex–concave least-squares residuals `½‖R‖²`, and the convex–concave minimizing-movement (JKO)
objective `J` (the Deep-Ritz analogue). Arm 1 trains the **BE** residual on the **CC** data
(decoupled via `--ac_loss_integrator`), so it targets a discrete operator `O(dt)` away from the
reference — an intentional test of whether the physics-loss integrator matters.

## Facts

| | |
|---|---|
| PDE | Allen–Cahn, `a=1`, `eps=2`, zero Dirichlet |
| Reference data | convex–concave (Eyre) FEM + Newton, `--ac_newton_tol 1e-6` (fp32 solve) |
| Initial condition | multi-frequency, `K=4` |
| Grid / horizon | `64²`, `dt=0.0025`, `n_steps=rollout_steps=10` (`T=0.025`, ≈2–3 decay times) |
| Dataset | `n_train=1024`, `n_val=128`, `n_test=256`, `seed=42` (identical across all arms) |
| Optimizer / epochs | `adam`, default cosine lr `1e-3→1e-4`, `1000` epochs, batch `32` |
| Metric | per-epoch **validation rollout MSE** (mean over batch, rollout steps, and grid) |
| Output dir | `output/ac_loss_comparison/` (git-ignored) |

Each arm regenerates the reference dataset from `seed=42` (deterministic ⇒ identical data); the
`n_steps=rollout_steps` choice avoids solving frames the rollout never uses. Runs are tagged
`fno_ac_{data|galerkin}_{ls|mm}_{cc|be}_...` so they do not collide.

> Caveat: at `eps=2` and this IC amplitude (~0.15) the dynamics are diffusion-dominated (the
> double-well barely engages), so the physics losses may land close together. Larger IC amplitude
> or `eps` would sharpen the differences — a follow-up. A mass-weighted (true `L²`) metric is also
> a planned addition.

## Run (Euler)

```bash
mkdir -p logs
sbatch experiments/ac_loss_comparison/sweep.sbatch      # 4-task array (0-3)
squeue --me
```

## Plot

```bash
python experiments/ac_loss_comparison/plot_ac_loss_comparison.py \
    --results_dir output/ac_loss_comparison/results     # optional: --smooth
```

Produces `output/ac_loss_comparison/ac_loss_comparison.png` — validation rollout-MSE vs epoch,
one curve per training loss.
