# Experiment: graphical abstract — ill-conditioning defeats first-order optimisers

The claim the paper opens with, stripped to its skeleton: **no neural operator at all**. Take the
`Q1` stiffness matrix `A` of the Dirichlet Poisson problem on a `64²` grid, form the
physics-informed least-squares loss over a plain coefficient vector,

```
L(u) = ½‖Au − b‖²,     b = M f  (consistent FE load)
```

and minimise it with Adam. The Hessian is `AᵀA = A²`, so the conditioning is `κ(A)² ~ h⁻⁴`. At
`64²` that is `6.5e5`, and it is already enough to defeat Adam — before any network is involved.

Same right-hand-side distribution (`K=4` multi-frequency), same `A`, same consistent load `b = Mf`
as every other Poisson experiment in the paper.

## Facts

| | |
|---|---|
| Grid | `64²`, `Q1`, 3844 interior dofs, zero Dirichlet |
| Spectrum | `λ_max = 3.9967` (mode 1,62), `λ_min = 4.970e-3` (mode 1,1) |
| Conditioning | `κ(A) = 804`, `κ(∇²L) = κ(A)² = 6.47e5` |
| Initialisation | `u₀ = 0` (default) → `e₀ = −u*`, i.e. **entirely smooth** |
| Precision | Adam in **float32**; every reported error in **float64** vs a float64 direct solve |
| Budget | `1e5` iterations, learning rate tuned over 9 values at a `1e4` budget |

## Results

| | rel. `L²` error |
|---|---|
| Adam, best lr (`3.5e-2`), 1e4 steps | 7.7% |
| Adam, best lr, 1e5 steps | **5.6%** |
| gradient descent, optimal step, 1e5 steps | 46.7% |
| trivial predictor `u ≡ 0` | 100% |

Ten times the compute buys well under a factor of two, and the descent is **non-monotone** — Adam
first overshoots to ≈360% relative error before coming back. A direct solve of the same system
takes milliseconds.

## Two facts that decide the figures

**1. The initialisation is not a detail — it is the mechanism.** `u₀ = 0` gives `e₀ = −u*`, and
`u* = A⁻¹b` decays like `λ⁻¹`, so the entire initial error sits in the *smooth* modes — exactly the
ones a first-order method cannot reduce. A random `u₀` at the same 100% initial error is far
easier (≈3% at 1e4 steps), because white noise puts most of its energy in stiff modes that Adam
crushes in a few hundred steps. `--init random` reproduces that; `zero` is the honest default.

**2. Adam and GD fail for opposite reasons, and only GD matches the classical intuition.**
Fraction of the squared error by eigenvalue band (`landscape_bands`):

| | λ<0.1 (26 modes) | λ≥1 (3511 modes) | final rel. error |
|---|---|---|---|
| gradient descent, all `k` from 1 to 1e5 | **100%** | 0% | 46.7% |
| Adam, `k ≲ 1e3` | ~100% | 0% | — |
| Adam, `k ≳ 1e4` | 0% | **100%** | 5.6% |

GD is the textbook smoother: it annihilates everything above `λ≈0.1` within one iteration and is
then permanently stuck on the smooth error. **Adam does the reverse** — it *solves* the smooth
content (those coefficients end at ~5e-5) and is left rattling in the stiff modes, crossing over
around `k ≈ 3–7e3`. That follows from per-coordinate normalisation: Adam advances every coordinate
by ~`lr` regardless of curvature, so flat directions are easy and steep ones are impossible to
settle in. Cf. `toy2d/`, where Adam's asymptotic radius is `≈0.011·lr` independent of `κ`.

**Beware the dilution trap in the heatmap.** Its colour is a *per-mode* amplitude, and the bands
hold wildly different mode counts (3511 vs 26), so a broad dim band can carry far more error than a
narrow bright one — which is exactly what happens after `k ≈ 1e4`. Quote `landscape_bands`, not the
heatmap, for any statement about where the error lives.

**3. The contour panel is abandoned.** The contours in the plane of the extreme eigenvectors are
ellipses of axis ratio `κ(A) = 804` (the *square root* of the Hessian's condition number, since
eigenvalues enter the quadratic form and hence the axis lengths under a root). It can be drawn by
forcing `--render_aspect`, but it only ever showed a GD trajectory creeping along the valley floor,
which the band figure says better. Going coarse enough to draw it at true aspect also destroys the
result: at `16²` (`κ = 45`) **Adam simply solves the problem**, to 0.06%.

## Implementation notes

* On a uniform grid `A = K ⊗ M̃ + M̃ ⊗ K`, and the 1D sine vectors diagonalise both factors, so the
  **2D discrete sine modes are exact eigenvectors of `A`** with closed-form eigenvalues. No
  eigensolver is needed and every modal projection comes from one DST. `run_landscape.py` verifies
  this numerically at startup (rel. residual `~1e-13`) rather than assuming it. Same fact
  `preconditioners/spectral.py` relies on.
* The learning rate is tuned in good faith over `[1e-3, 3e-1]` and the sweep warns if the optimum
  lands on an edge — "we tuned the baseline" has to be defensible.

## Run

```bash
python experiments/graphical_abstract/run_landscape.py \
    --out output/graphical_abstract/landscape.npz            # ~2 min, CPU
python experiments/graphical_abstract/plot_landscape.py \
    --npz output/graphical_abstract/landscape.npz --out_dir paper/figures
```

Useful flags: `--grid`, `--steps`, `--init {zero,random}`, and `--render_aspect` on the plot script
(forces a target drawn contour ratio instead of fitting the trajectory).

## Figures

* `landscape_error` — relative error vs iteration, Adam and GD, log–log.
* `landscape_bands` — fraction of the squared error per eigenvalue band, Adam vs GD. **The
  quantitative figure**; the clean statement of the mechanism, and the one to cite.
* `landscape_spectrum`, `landscape_spectrum_gd` — modal error amplitude over
  (iteration, eigenvalue), per optimizer. Detail behind the band figure; read with the dilution
  caveat above. The GD panel is the classical picture (high modes gone in one step, smooth band lit
  throughout); the Adam panel is its opposite.
* `landscape_contours` — abandoned, see above. Still produced, not maintained.

## Later

Multigrid: add `P ≈ A⁻¹` (one V-cycle, `preconditioners/multigrid.py`) as a second arm — the same
three figures with a preconditioned curve/front should show the front reaching the bottom.
