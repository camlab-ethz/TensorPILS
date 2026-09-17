# The continuity weight: the control the structured Stokes table was missing

## The question

The strong-form baseline carries `--pi_div_weight` and tunes it. The FEM least-squares arm had
no such knob, so the structured table compared a *weighted* objective against an *unweighted*
one and attributed the whole gap to the discretisation. Measured on the archived sweeps and on
the gradients themselves:

| arm | velocity | pressure |
|---|---:|---:|
| PINO, `w=1` | 46.05 % | 13.97 % |
| FEM bare LS, no weight | 31.27 % | 10.92 % |
| PINO, `w=100` | 1.73 % | 1.37 % |
| `pls` (block P, ω=256) | 2.38 % ± 0.27 | 1.69 % ± 0.23 |

`cosine(∇ FEM-LS, ∇ PINO-mse) = 0.993` at `w=1` and `0.83` at `w=300`. At `w=1` the two are
very nearly the *same objective* — and at `w=1` PINO is **worse** than the unweighted FEM
control. So the 27× is the weight, not the finite differences.

As a norm weight this is `W = diag(I, w·I)`: the `pls` preconditioner with the velocity block's
`A⁻¹` replaced by the identity. Two questions follow, and this directory answers both.

## 1. Does it train? (`sweep.txt` → `compare.py`)

Seed 42, 500 epochs, lr 1e-3, 65² velocity grid, `--loss galerkin --stokes_div_weight w`:

| `w` | velocity | pressure |
|---|---:|---:|
| 1 | 43.97 % | 13.88 % |
| 10 | 2.30 % | 1.64 % |
| **30** | **1.77 %** | **1.15 %** |
| 100 | 2.31 % | 1.19 % |
| 300 | 3.40 % | 1.39 % |
| 1000 | 39.02 % | 3.99 % |

`w=30` then re-run at seeds 43 and 44:

| arm | velocity | pressure | seeds |
|---|---:|---:|---|
| FEM bare LS, `w=30` | **1.84 % ± 0.10** | **1.31 % ± 0.14** | 42/43/44 → 1.77 / 1.79 / 1.96 |
| `pls` (block P, ω=256) | 2.38 % ± 0.27 | 1.69 % ± 0.23 | 3 |
| PINO, `w=100` | 1.73 % | 1.37 % | 1 |

**One scalar on the continuity rows takes the negative control from 43.97 % to 1.84 %, past the
preconditioned arm and level with PINO.** The gap over `pls` is ~0.5 pp against pooled spread
~0.17, i.e. about 3σ on 3+3 seeds — real, but at the edge of what three seeds settle. What is
not marginal is the 24× over `w=1`, and that the *bare* residual and the *strong-form* baseline
land in the same place once both carry the weight.

## 2. Is it a preconditioner? (`condition_number.py`)

Asked by Marius Zeinhofer, who expected no: "tuning this number cannot correct the
ill-conditioning ... if it improves the condition number then this is our preconditioner, just a
very simple one; block diagonal, with only the pressure block having a single number and the
velocity block being identity."

The objective `½‖W(Kc−b)‖²` has Gauss–Newton matrix `K W² K`, so the quantity is
`κ(K W² K) = κ(WK)²`. Measured on the admissible subspace (Dirichlet velocity DOFs removed, the
constant-pressure gauge deflated — `Bᵀ1 = 0` makes it a gauge, not ill-conditioning), with `w`
**retuned at every resolution**, because a fixed `w` conflates a better constant with a better
rate:

| n_p | dofs | κ(GN) at `w=1` | best κ(GN) | at `w` | gain |
|---|---:|---:|---:|---:|---:|
| 5 | 123 | 4.67e7 | 7.35e4 | 40 | 635× |
| 9 | 531 | 7.35e8 | 3.53e5 | 69 | 2084× |
| 17 | 2211 | 1.19e10 | 2.48e6 | 91 | 4792× |
| 25 | 5043 | 6.04e10 | 1.06e7 | 103 | 5677× |
| 33 | 9027 | 1.91e11 | 3.23e7 | 133 | **5913×** |

Local slope `dlog κ / dlog h`:

| | | | | |
|---|---:|---:|---:|---:|
| `w = 1` | −3.98 | −4.02 | −4.01 | **−4.00** |
| `w = ` best | −2.26 | −2.82 | −3.59 | **−3.86** |
| ideal block `P` | −1.82 | −2.00 | **−2.00** | — |

**Marius is right: the scalar does not correct the ill-conditioning.** The rate stays `O(h⁻⁴)` —
the per-h-tuned exponent marches to −4.00 — and the gain saturates (increments 3.3×, 2.3×,
1.18×, 1.04×) at ~5900× on the Gauss–Newton matrix, i.e. ~77× on `κ(WK)`.

The contrast makes the division of labour explicit. The ideal block preconditioner
`P = diag(A⁻¹/μ, ωμ/diag(M_p))` — exact `A⁻¹`, i.e. what the V-cycle approximates — scored in
the same framework converges cleanly to `O(h⁻²)`. **The velocity block's `A⁻¹` buys the order
reduction; the pressure scalar buys a bounded constant.** Two further signs the scalar is not
`h`-independent: the optimal `w` drifts 40 → 133 over `h = 1/8 → 1/64` (~`h^−0.58`), against the
`h`-independent `ω ≈ 16` recorded for the full block `P`; and the *conditioning* optimum
(`w ≈ 133`) is not the *training* optimum (`w = 30`), so κ is not the whole story — over-weighting
the continuity rows biases the minimiser toward divergence-free at the momentum equation's
expense, and 500 epochs of Adam is not a solver whose rate is set by κ.

A bounded constant is still worth having, and at one resolution it is indistinguishable from a
preconditioner: at the 65² grid the paper trains on, ~5900× is the difference between an arm
that does not train (43.97 %) and one that beats the preconditioned arm (1.84 %).

## What this means for the table

The structured Stokes row `galerkin — 31.27 %, "the negative control, it does not train"` is
only true of the *unweighted* form. With the same scalar the baseline gets, it trains, and it
wins. Either weight both or weight neither; the current table does one of each.

This does not touch the unstructured results — `experiments/stokes/unstructured/` compares
`pls` against an unweighted `galerkin` on the obstacle mesh, and whether the weight rescues it
*there* is the open question. On an unstructured mesh there is no PINO arm to be fair to, so the
stakes are lower, but the answer would say whether the scalar is a structured-grid artifact.

## Running

```bash
bash .../force_submit.sh experiments/stokes/galerkin_divweight/sweep.txt   # from a compute node
python experiments/stokes/galerkin_divweight/compare.py
clrun -p cpu -t 120 -- python experiments/stokes/galerkin_divweight/condition_number.py
```
