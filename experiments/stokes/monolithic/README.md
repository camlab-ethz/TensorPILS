# Experiment: monolithic multigrid with a symmetric Uzawa smoother

The block preconditioner is only **norm-equivalent** to `K⁻¹`, which is why the `pls` loss has to
use it as a norm weight and why the resulting conditioning is `O(h⁻²)` rather than `O(1)`. The note
(§"The Monolithic Approach") describes the way out: a monolithic Stokes V-cycle is a genuine
**approximate inverse**, and with `P ≈ K⁻¹` the *applied* form becomes viable,

```
L = ½‖P(Kc − b)‖²,     G(θ) = Jᵀ(PK)ᵀ(PK)J,     κ = κ₂(PK)² = O(1)
```

mesh-independent, exactly as for Poisson. Note this is the same algebraic form that is a documented
**dead end** for a block-diagonal `P` — the only thing that makes it work is `PK ≈ I`.

Implementation: [`tensorpils/preconditioners/stokes_monolithic.py`](../../../tensorpils/preconditioners/stokes_monolithic.py).
Two deviations from the note, both deliberate and flagged in the module docstring: the velocity
prolongation is the bilinear operator on the Q2 node set rather than a true quadratic interpolation
(the same approximation the scalar V-cycle already makes), and coarse operators are
**re-discretized** rather than formed as `RAΠ`, so that every level stays an inf-sup stable
Taylor-Hood pair.

## What was measured

`measure_monolithic.py` (float64, dense, on the admissible subspace):

| velocity grid | V-cycle contraction | `κ₂(PK)` | `κ((PK)ᵀPK)` = `G` of `½‖Pr‖²` | block `κ(KPK)` = `G` of `½rᵀPr` |
|---|---|---|---|---|
| `17²` | 0.253 | 9.73 | `9.5·10¹` | `2.2·10⁶` |
| `33²` | 0.264 | 3.29 | `1.1·10¹` | `8.7·10⁶` |

The contraction rate is flat in `h` — textbook multigrid — and the applied-form conditioning is
`O(10)` against the weighted block form's `O(10⁷)`, i.e. the regime change the note predicts, worth
**five to six orders of magnitude**.

### The form/preconditioner pairing is forced, not a preference

Neither operator can borrow the other's loss form, and the reason is structural:

- `K` restricted to the admissible subspace is a saddle point with **exactly one negative eigenvalue
  per gauge-fixed pressure DOF**. Any approximate *inverse* inherits that signature, so the monolithic
  `P` is **indefinite** — measured 288 negative eigenvalues out of 2210 at `33²`, exactly `n_p − 1`,
  with `λ_min = −1.0·10⁴`. So `½ rᵀPr` with it is unbounded below; it is not a norm and no amount of
  tuning makes it one. Measured consequence: that arm reaches **406 % / 160 %** — divergence, as
  required.
- The block `P` is SPD by construction (`λ_min = +0.25`) so it *can* be a norm weight, but it is not
  an inverse, so applying it is the dead end.

Pinned by `test_monolithic_P_is_indefinite_so_it_cannot_be_a_norm`. This is why
`--stokes_pls_form auto` exists: the pairing is determined by the operator, and getting it wrong
fails loudly in one direction and quietly in the other.

### Choosing the smoother

The smoother is what costs, so it is worth knowing the trade-off (`--scan`, at `33²`, `ω=1`):

| Uzawa sweeps `ν` | Chebyshev degree | A-applies / level | contraction | `κ((PK)ᵀPK)` |
|---|---|---|---|---|
| 3 | 4 | 48 | 0.410 | `3.7·10³` |
| 3 | 6 | 72 | 0.378 | `2.2·10²` |
| 4 | 4 | 64 | 0.310 | `8.3·10²` |
| 4 | 6 | 96 | 0.287 | `6.2·10¹` |
| 2 | 8 | 64 | 0.458 | `3.9·10²` |
| **4** | **8** | **128** | **0.264** | **`1.1·10¹`** |

Hence the defaults `--uzawa_pre 4 --uzawa_post 4 --cheb_degree 8`. Note the conditioning improves
far faster than the contraction rate does — the rate is the spectral radius, while the loss sees the
2-norm, and the last few stubborn directions are what the extra smoothing removes.

**`schur_omega` behaves oppositely here.** Inside the Uzawa *iteration* the note's restriction
`ω ∈ (0,1]` is real: `ω=1` is the optimum and `ω=2` diverges outright. That is the exact opposite of
its role as a *norm weight* in the block preconditioner, where it is unconstrained above and wants
`ω≈16` (see [`../conditioning/`](../conditioning/README.md)). Same symbol, same formula, two
regimes — worth stating explicitly, because carrying `ω=0.5` over from one to the other costs a
factor 15 in conditioning in one direction and divergence in the other.

## Training result: O(1) conditioning is necessary, not sufficient

At `33²`, 300 epochs, against the baselines from [`../h_refinement/`](../h_refinement/README.md) at
identical settings:

| `P` | loss form | κ of `G(θ)` | velocity | pressure |
|---|---|---|---|---|
| — | `data` (supervised) | — | 4.49 % | 4.23 % |
| block | weighted `½rᵀPr` | 8.7·10⁶ | 5.82 % | 3.87 % |
| monolithic | weighted `½rᵀPr` | *not a norm* | **406 %** | **160 %** |
| block | applied `½‖Pr‖²` | 2.7·10⁶ | 8.07 % | 6.37 % |
| monolithic | applied `½‖Pr‖²` | **1.1·10¹** | 19.34 % | 4.63 % |
| **monolithic** | **applied, FE metric** | **1.1·10¹** | **4.33 %** | **4.10 %** |

The fifth row is the interesting one: the best-conditioned loss in the table, by six orders of
magnitude, trains **3× worse** than the block weighted form. That is not a refutation of the
conditioning story but its boundary — [`../blend_sweep/`](../blend_sweep/README.md) already showed the
error saturating below `κ ≈ 10⁹`, so there was nothing left for conditioning to buy. What went wrong
was the **metric**.

Since `Pr ≈ c − c*`, the applied form measures error in **nodal** units, where every velocity and
pressure DOF counts equally — but this dataset has `‖p‖ ~ 50‖u‖` at unit velocity. Perturbing both
fields by the same 10 % relative error, the pressure block contributes **357×** more to `½‖Pr‖²` than
velocity, so velocity is barely optimised. That is exactly the 19.34 % / 4.63 % signature, and it is
the *same* failure the hand-written supervised loss had (see the Stokes field-scaling section of
`CLAUDE.md`), arriving by a different route. The block applied form shows a milder version (2.1×
domination). The weighted form never had it, because `KPK` is not the nodal metric.

`--stokes_pls_form applied_fe` divides each block by its own dataset FE norm. Changing **only** that —
same preconditioner, same conditioning, same hyperparameters — moves velocity from 19.34 % to
**4.33 %**, making it the best label-free arm and better than supervised training on velocity. Worth
naming for what it is: since `Pr ≈ c − c*`, that loss **is** the supervised objective, computed from
the residual with no solutions in it.

> **Honesty note.** The FE metric needs `σ_u, σ_p` in the loss, and `σ_p` is measured from reference
> solutions — one scalar, obtainable from a handful of solves or estimated from the forcing
> (`p ~ f/k`), so not supervision in any meaningful sense, but one constant more than the weighted
> form needs. The choice is therefore not settled by accuracy alone: monolithic is more accurate and
> mesh-independent but ~15× more expensive per step and wants a calibration constant; block is
> cheaper, needs nothing, and is `O(h⁻²)`.

## Facts

| | |
|---|---|
| Arms (3) | monolithic `P` + applied form; monolithic `P` + weighted form; **block** `P` + applied form (the note's dead end, as a control) |
| Grid | velocity `33²` / pressure `17²`. The monolithic V-cycle costs ~15× a block one per step, so the sweep runs at `33²` where [`../h_refinement/`](../h_refinement/README.md) already provides `galerkin`/`pls`/`data` baselines at identical settings |
| Smoother | Uzawa `4/4`, Chebyshev-Jacobi degree `8` (`ratio=30`), `schur_omega=1.0` for the monolithic arms |
| Levels | `mg_levels=4` → pressure grids `17, 9, 5, 3`; coarsest solved exactly (dense, pressure-pinned) |
| Dataset | `μ=1`, `K=4`, `n_train=512`, `n_val=64`, `n_test=128`, `seed=42` |
| Optimizer | `adam`, cosine `lr 2e-3 → 1e-4`, 300 epochs, batch 32 |
| Output dir | `output/stokes/monolithic/` (git-ignored; prefix carries `mono-L4-u44-c8`) |

## Run

```bash
# conditioning (CPU, float64) -- the O(1) claim
python experiments/stokes/monolithic/measure_monolithic.py --grids 17 33 65
python experiments/stokes/monolithic/measure_monolithic.py --grids 33 --scan   # smoother choice

# training
sbatch experiments/stokes/monolithic/sweep.sbatch        # array 0-2

# or one arm directly
python -m tensorpils.cli --pde stokes --loss pls --stokes_precond monolithic \
    --schur_omega 1.0 --uzawa_pre 4 --uzawa_post 4 --cheb_degree 8 \
    --grid_resolution 33 --n_train 512 --n_val 64 --n_test 128 -k 4 --epochs 300 \
    --lr 2e-3 --lr_min 1e-4 --seed 42 --device cuda --output_dir output/stokes/monolithic
```

`--stokes_pls_form` defaults to `auto`: the applied form for a monolithic `P`, the weighted form for
a block `P`. Forcing `applied` with `block` prints a warning and runs anyway — that pairing exists
only as the control arm above.
