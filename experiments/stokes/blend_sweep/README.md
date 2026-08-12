# Experiment: Stokes blend sweep (the residual → preconditioned transition)

The Stokes analogue of [`poisson/sweep_blend/`](../../poisson/sweep_blend/README.md). There the
knob interpolates `P_t = (1−t)I + tA⁻¹`, from the bare residual loss to the supervised anchor. For a
saddle point there **is no `K⁻¹` endpoint to interpolate toward** — no block-diagonal operator
approximates it, which is the reason the loss must use `P` as a norm weight in the first place — so
the family runs from the bare loss to the block preconditioner instead:

```
P_t = (1−t)·α·I + t·P_block,        α = λ_max(P_block)
```

- `t = 0` → `P = αI`, so `½ rᵀPr = ½·α·‖r‖²`: **exactly** the `galerkin` loss up to a positive
  constant (which changes no conditioning and, under Adam, essentially no step size). Measured:
  `t=0` gives 49.51 % / 13.33 % against `galerkin`'s 49.59 % / 13.31 % — a 0.1 % agreement that
  validates the whole family.
- `t = 1` → the block preconditioner exactly.
- `α` matches the *spectral radius* of the two ends, so the loss magnitude stays comparable along
  the sweep rather than jumping by orders of magnitude near `t=0`.

A convex combination of SPD operators is SPD, so `P_t` remains a legitimate norm at every `t`.

**Why the sweep and not just the two endpoints.** The endpoints already differ by an order of
magnitude in error. What the sweep adds is the *mechanism*: `κ(K P_t K)` moves continuously from
`O(h⁻⁴)` to `O(h⁻²)`, so plotting error against **measured** conditioning tests whether
conditioning is what governs the error — rather than asserting it. Unlike Poisson, `P_t` contains a
multigrid V-cycle and has no closed-form spectrum, so the axis has to be measured
(`measure_blend_cond.py`, float64 dense on the admissible subspace).

`t` is spaced logarithmically in `1−t`. The identity term dominates until `1−t ~ 1/κ(P)`, so a
uniform grid in `t` would put every interesting point in the last 5 %: at `17²`, `t=0.5` still sits
at `κ = 3.9e8` against `7.3e8` at `t=0`, and the drop to `2.2e6` all happens past `t=0.95`.

## Facts

| | |
|---|---|
| Strengths (10) | `t = 0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.997, 0.999, 1` |
| Extra arms (2) | `t=1` with `--schur_omega 4`, `16` — where the *measured* conditioning is minimal (see [`../conditioning/`](../conditioning/README.md)); output dir `output/stokes/omega/` |
| Dataset | Stokes, `μ=1`, `K=4`, velocity grid `65²`, `n_train=512`, `n_val=64`, `n_test=128`, `seed=42` |
| Optimizer | `adam`, cosine `lr 2e-3 → 1e-4`, 300 epochs, batch 32 |
| FNO | modes `16×16`, hidden `64`, `5` layers (3,008,803 params) |
| Preconditioner | `mg_levels=4`, pre/post `2/2`, `schur_omega=0.5` unless stated |
| Output dir | `output/stokes/blend_sweep/` (git-ignored; prefix carries `-t<strength>`) |

## Run

```bash
# 1. the conditioning axis (CPU, float64, a few minutes at 65^2)
python experiments/stokes/blend_sweep/measure_blend_cond.py --grid 65

# 2. the sweep itself
sbatch experiments/stokes/blend_sweep/sweep.sbatch              # array 0-9
sbatch --array=10-11 experiments/stokes/blend_sweep/sweep.sbatch  # the schur_omega arms

# or one point directly
python -m tensorpils.cli --pde stokes --loss pls --stokes_precond_strength 0.95 \
    --grid_resolution 65 --n_train 512 --n_val 64 --n_test 128 -k 4 --epochs 300 \
    --lr 2e-3 --lr_min 1e-4 --seed 42 --device cuda --output_dir output/stokes/blend_sweep
```

## Plot

```bash
python experiments/stokes/blend_sweep/plot_blend_sweep.py \
    --results_dir output/stokes/blend_sweep/results
```

Produces `blend_overlay.png` (validation error vs epoch per `t`, velocity and pressure panels,
legend annotated with the measured `κ`) and `blend_collapse.png` (final error vs `κ(K P_t K)`,
points labelled by `t`). The collapse is the figure that carries the claim.
