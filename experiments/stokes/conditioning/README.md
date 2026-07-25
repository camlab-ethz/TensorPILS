# Experiment: measured conditioning of the Stokes losses (the theory side)

Everything the paper claims about Stokes is a claim about condition numbers:

| loss | Gauss--Newton matrix | predicted |
|---|---|---|
| `galerkin`  `½‖Kc−b‖²` | `Jᵀ K² J` | `O(h⁻⁴)` |
| `pls`  `½ rᵀPr` | `Jᵀ K P K J` | `O(h⁻²)` (Lemma 1, via `κ(P)`) |
| the note's dead end  `½‖P(Kc−b)‖²` | `Jᵀ (PK)ᵀ(PK) J` | `O(h⁻⁴)` (Lemma 1 bound) |

and, underneath all three, on `max|λ(PK)| / min|λ(PK)| = O(1)`. This experiment measures all of
them directly in float64 rather than asserting them, so the exponents in the paper are fitted from
data. **No training is involved** — it is a pure linear-algebra measurement of the preconditioner
that training actually uses (same `mg_levels`, same `schur_omega`, V-cycle inexactness included).

## What is diagonalized

The loss never sees the full DOF space. Velocity is projected to zero on the Dirichlet boundary
and the residual is zeroed there; pressure is projected to zero mean in the lumped-`M_p` inner
product, because `Bᵀ1 = 0` makes the residual blind to that mode. `K` is singular on the full space
and invertible on the **admissible subspace**

```
S = {interior velocity DOFs} ⊕ {p : wᵀp = 0},        w = lumped M_p
```

so the script builds an **orthonormal** basis `Z` of `S` (a selection matrix for velocity, a
Householder basis of `w^⊥` for pressure) and reports condition numbers of `K_S = ZᵀKZ` and
`P_S = ZᵀPZ`. Orthonormality is what makes those restrictions similarity-faithful, hence the
numbers basis-independent.

Two implementation notes. Spectra of the non-symmetric `P_S K_S` never require a non-symmetric
eigensolver: with `P_S = LLᵀ` (Cholesky), `P_S K_S` is similar to the symmetric `Lᵀ K_S L`.
And float64 is not optional — at `65²` the `galerkin` conditioning is `2·10¹¹`, whose smallest
eigenvalue is pure noise in float32.

## Built-in validation

With the **exact** blocks `P = diag(A⁻¹, S⁻¹)`, `S = BA⁻¹Bᵀ`, the note's elementary computation
gives exactly three eigenvalues `{1, (1±√5)/2}`, ratio `φ² = 2.618034`. `--exact_blocks` computes
this, and it comes out to 6 digits at every grid — which checks the theory claim *and* the subspace
reduction at once (a wrong basis would not produce three eigenvalues). It also separates "the block
idea is weak" from "our *inexact* blocks are weak": the answer is the latter, by a factor ~33.

## Facts

| | |
|---|---|
| Grids | `9²`, `17²`, `33²`, `65²` velocity (`5²`…`33²` pressure); `m = dim S` = 122 / 530 / 2210 / 9026 |
| Preconditioner | `StokesBlockPreconditioner`, `mg_levels=4`, pre/post `2/2`, `schur_omega=0.5`, `μ=1` |
| Precision | float64 throughout; dense `eigvalsh` / `cholesky` |
| Cost | seconds to `33²`, ~2 min at `65²`, hours at `129²` (dense `36482²`) |
| Output | `output/stokes/conditioning/cond_gr<grid>.json`, `table.tex`, `conditioning.png` |

## Run

```bash
python experiments/stokes/conditioning/measure_conditioning.py --grids 9 17 33 65 --exact_blocks
python experiments/stokes/conditioning/plot_conditioning.py  --results_dir output/stokes/conditioning
```

CPU-bound and float64, so this belongs on CPU (a consumer GPU is ~1/64 rate in double precision);
it runs happily alongside a training job. `--skip_naive` drops one eigensolve.

## Result

Fitted exponent `s` in `κ ~ h⁻ˢ`, over `9² … 65²`:

| quantity | measured `s` | predicted |
|---|---|---|
| `κ(K)` | 1.99 | 2 |
| `κ(K²)` — `G` of `L_LS` | 3.98 | 4 |
| `κ(P)` | 2.02 | 2 |
| `max|λ(PK)|/min|λ(PK)|` | **−0.05** (flat at ≈86) | `O(1)` |
| `κ(KPK)` — `G` of `L_PLS` | 1.93 | 2 |
| `κ((PK)ᵀPK)` — `G` of the dead end | 3.35 | ≤ 4 |

Three things worth carrying into the paper:

1. **Lemma 1 is confirmed and nearly tight.** At `65²`, `κ(P)·ratio² = 6.0·10⁷` against a measured
   `κ(KPK) = 3.5·10⁷` — the bound is within a factor 1.7.
2. **The `O(1)` assumption survives inexactness.** The eigenvalue ratio is flat in `h` even though
   `Â⁻¹` is a single V-cycle and `Ŝ⁻¹` a lumped diagonal; only its *constant* degrades (86 vs the
   exact-block 2.618). That constant is what `--schur_omega` buys back, see below.
3. **The dead-end variant grows like `h⁻³·⁴`, not `h⁻⁴`**, so the Lemma's bound is not attained, and
   it only *crosses* the weighted form at `65²` (`4.0·10⁷` vs `3.5·10⁷`). Below that resolution the
   naive applied form is in fact the better-conditioned of the two. The note's conclusion is right
   asymptotically; the crossover is later than the bound suggests.

### `schur_omega` is worth tuning

`ω` only rescales the pressure block of the *norm*, so — unlike its role inside an Uzawa smoother,
where the note's `ω ∈ (0,1]` restriction comes from — it is not constrained to be `≤1`. It trades
`κ(P) ∝ ω` against `ratio(PK) ∝ 1/ω`, and since `κ(KPK) ≲ κ(P)·ratio²` the product improves as `ω`
grows. Smoothing count is nearly irrelevant by comparison (`s=1`, `2`, `4` differ by <7%). See
`omega_scan.md` for the numbers and where the optimum sits.
