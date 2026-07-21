# Experiment: realistic (but simple) Allen–Cahn — which loss survives the stiff regime?

A realistic FEM setting for the FNO one-step Allen–Cahn stepper: **256²**, `a=1`, **`eps=64`**
(stiff reaction), **`dt=0.01`**, tame `K=4` initial data, trained on a **10-step pushforward**
rollout with `n_train=1024`, then **rolled out to 100 steps** and compared against the
convex–concave (Eyre) reference. Three training losses compete:

| # | training loss | flags | conditioning it inherits |
|---|---|---|---|
| 0 | data-driven (supervised MSE) | `--loss data` | `O(1)` — no PDE operator |
| 1 | minimizing-movement (JKO / Deep-Ritz) | `--loss galerkin --ac_loss_form min_movement` | `κ(J^cc) ~ 10⁴` (linear) |
| 2 | bare least-squares `½‖R_cc‖²` | `--loss galerkin --ac_loss_integrator convex_concave` | `κ² ~ 10⁸` (squared) |

All arms train on and are evaluated against the **same** deterministic CC reference (identical
data from `--seed 42`), and all use **pushforward** (`--bptt_mode pushforward`) — best in every
prior study, and it keeps 256² activation memory to a single forward (no BPTT through the rollout).

**Hypothesis** (see `notes/ac_autoregressive/`, §"Numerical soundness of the target regime"): the
convex–concave Newton Jacobian `J^cc = M/dt + a²A + 3ε²M·diag(u²)` has `κ ~ 1.3×10⁴`, set by the
diffusion group `a²dt/h²`, *not* by `eps`. The bare least-squares loss minimizes `½‖R‖²`, whose
Hessian `≈ (J^cc)ᵀJ^cc` sees `κ² ~ 10⁸` — the squaring is expected to stall it. Minimizing-movement
(Hessian `= J^cc`, `κ ~ 10⁴`, ill-conditioned only *linearly*) and data-driven (`O(1)`) should hold.

## Why this regime is sound (and matched)

| group | value | consequence |
|---|---|---|
| interface resolution `ℓ/h = a(n−1)/ε` | `≈ 4` | interface (transition ≈ 17 cells) well resolved — the reason for 256² |
| reaction stiffness `ε²·dt` | `≈ 41` | phase separation is essentially instantaneous (within step 1); convex–concave split mandatory |
| diffusion / parabolic CFL `a²·dt/h²` | `≈ 650` | diffusion implicit; **this sets the Jacobian conditioning ~10⁴** |

Coarsening `L(t) ~ √(2a²t)`: the 10-step training horizon (`t=0.1`, `L≈0.45`) already sits in the
coarsening regime; the 100-step rollout (`t=1.0`, `L≈1.4 >` domain) runs the domain out to a
single-phase, boundary-pinned steady state. So the horizon is about how much *coarsening* to expose,
not whether separation happens.

## Facts

| | |
|---|---|
| PDE | Allen–Cahn `u_t = a²Δu + ε²u(1−u²)`, `a=1`, **`eps=64`**, zero Dirichlet |
| Reference data | convex–concave (Eyre) FEM + Newton (sparse, float64 solve) |
| Initial condition | multi-frequency, `K=4`, `r=0.5` (tame) |
| Grid / horizon | **`256²`**, **`dt=0.01`**, `n_steps=rollout_steps=10` (`T=0.1`) |
| AR mode | **pushforward** for all arms (`--bptt_mode pushforward`) |
| Dataset | `n_train=1024`, `n_val=128`, `n_test=256`, `seed=42` (identical across arms) |
| FNO | CLI defaults: modes `16²`, hidden `64`, `5` layers (bump `--n_modes 32 32` if the sharp interfaces look under-resolved) |
| Optimizer / epochs | `adam`, cosine lr `1e-3→1e-4`, `1000` epochs, batch `16` |
| Rollout eval | **100 steps**, energy + relative FEM-`L²` vs CC, on **both test and train** splits |
| Output dir | `output/allen_cahn/realistic_ac/` (git-ignored) |

## Run (Euler)

```bash
mkdir -p logs
sbatch experiments/allen_cahn/realistic_ac/sweep.sbatch      # 3-task array (0=data, 1=mm, 2=bare-LS)
squeue --me
# once training is done:
sbatch experiments/allen_cahn/realistic_ac/analyze.sbatch    # compute metrics + figures + heatmaps
```

## Plot / pull

The two-stage split keeps the large `.pth` checkpoints on Euler (`analyze.sbatch` produces the
figures there). Pull only the small metrics + PNGs, **excluding the checkpoints**:

```bash
rsync -avz --exclude 'checkpoints' \
    euler:~/TensorPILS/output/allen_cahn/realistic_ac ~/Documents/TensorPILS/output/allen_cahn/
# re-plot the scalar figures locally from metrics.json if you want to tweak them:
python experiments/allen_cahn/realistic_ac/plot_rollout.py \
    --metrics output/allen_cahn/realistic_ac/metrics.json
```

Produces (in `output/allen_cahn/realistic_ac/`):
- **`rollout_error.png`** — relative `L²` vs CC per rollout step, **test (solid) + train (dashed)**,
  one colour per loss, log-y, train-horizon marked. *(deliverable a)*
- **`rollout_energy.png`** — Ginzburg–Landau energy per step, same test/train overlay, with the CC
  reference energy. A gradient flow must not increase it; a climbing curve is a blow-up. *(deliverable c)*
- **`heatmaps/heatmap_test_sample{0,1,2}.png`** — 2D fields, rows = timesteps, columns =
  `[CC reference | data | min-movement | bare-LS]`, each panel annotated with its relative L2 vs CC.
  *(deliverable b)* Use `--split train` / `--samples …` / `--runs …` for subsets.

## Adding the preconditioned-LS arm

The point of dropping bare LS into this benchmark is to show a **preconditioned** LS loss
`½‖P·R‖²` *rescues* the squared conditioning — turning "bare LS fails" into "preconditioning fixes
it". `P ≈ J₀⁻¹` with `J₀ = a²A + cM` the frozen (`u²=1`) Newton Jacobian, `c = 1/dt + 3ε²`,
realized as a multigrid V-cycle on `a²A + cM` (grid-scalable at 256²). See
`notes/ac_autoregressive/` §"Preconditioning the least-squares loss" for the derivation.

**This loss is implemented** (`--ac_precond multigrid`, with `--ac_loss_form galerkin`). To enable
the arm: uncomment task 3 in `sweep.sbatch` and bump `--array=0-3`:

```bash
"--loss galerkin --ac_loss_integrator convex_concave --ac_precond multigrid"   # 3 preconditioned LS
```

No analysis change is needed — `compute_rollout.py` / `plot_rollout.py` / `heatmaps.py` key runs by
loss (the preconditioned arm reads back as `pls`, tagged `_ls_precmg_` in filenames so it never
collides with bare LS) and glob whatever checkpoints exist, so they pick the new arm up
automatically. At 256² consider `--mg_levels 6` (coarsens 256→…→8) for a cheaper coarse solve.
