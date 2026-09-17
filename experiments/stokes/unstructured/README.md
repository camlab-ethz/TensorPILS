# Stokes past an obstacle — the case the strong-form baselines cannot enter

## The question

The structured Stokes table (`experiments/stokes/stokes_paper/stokes_benchmark`) reads, on
velocity / pressure relative FE-L²:

| arm | u | p |
|-----|---|---|
| PINO | **2.09 ± 0.26 %** | **1.50 ± 0.24 %** |
| PLS block (ours, label-free) | 2.38 ± 0.31 % | 1.69 ± 0.27 % |
| data (supervised) | 2.85 % | 2.81 % |
| PI-DeepONet | 33.0 % | 34.2 % |
| bare FEM LS | 21.6 % | 8.6 % |

PINO is ahead by about one seed standard deviation. **That table stays.** This experiment does
not replace it; deleting the comparison where the baseline applies, because the baseline wins
there, is the one way to make a favourable result worthless. The claim being made is a
*capability* one, and it needs both halves:

> On a uniform grid, where a finite-difference residual is at its best and both methods apply,
> PINO is marginally ahead. On a mesh with a curved hole, the FD residual does not exist and the
> FEM residual is unchanged.

Two facts make that more than rhetoric, and both are measured rather than asserted.

**The structured Stokes benchmark is close to saturated.** Its velocity solution lies 2.6 %
outside the span of the `2K² = 200` sine modes its own forcing is built from — and every arm's
reported error (2.09–2.85 %) is that same 2.6 %. The easy, essentially finite-dimensional part
of the problem is fully learned by everyone, and the arms are being separated inside a band as
wide as the part nobody resolves. On this geometry the velocity is **~50 %** outside that span.
(The Poisson case is starker still: exactly 0.0000 % on the square, 9.65 % on the disc. See
`experiments/poisson/unstructured/README.md`.)

**The unstructured case needs both new pieces at once.** GAOT with the geometric V-cycle has no
hierarchy to build; AMG with an FNO has no image to consume. The `pls` arm here is the first
place they work together, which is why it is the arm to read.

## Setup

Unit square with a circular hole of radius `0.14` at `(0.40, 0.50)` — off-centre on purpose, so
the solution does not inherit the domain's symmetry. Meshed **P2/P1 Taylor-Hood on triangles**
(`triangle6`), zero velocity on **both** boundary components, driven by the same
`stokes_body_force` as the structured runs (`K=10`, `r=-0.5`).

`chara_length=0.035` gives **3998 P2 / 1035 P1** nodes against the structured benchmark's
`65² = 4225` / `33² = 1089` — matched on node count, i.e. on the dimension of the discrete
problem and the model's cost. Everything else is byte-identical to the structured protocol:
1024/128/256, 500 epochs, batch 32, Adam.

The reference is the same discrete Taylor-Hood solve, and it is as good here as there: relative
residual **4.4e-06**, against 6.7e-06 structured. There was never a manufactured solution for
Stokes, so unlike the Poisson disc nothing had to be given up to change domain.

### What is not available, and why

| | reason |
|---|---|
| `--loss pino` | finite-difference stencils need an image |
| `--loss pideeponet` | the branch is a fixed sensor grid |
| `--model fno` / `deeponet` | same |
| `--stokes_precond monolithic` | its transfer operators are geometric — both fields nesting by two on a grid |
| `--stokes_pls_form applied` | needs a genuine `P ≈ K⁻¹`, which is the monolithic one |

The block preconditioner **is** available, and it is the one the structured table's label-free
number came from: `P = diag(Â⁻¹/μ, ω μ/diag(M_p))`, with the velocity half swapped from a
geometric V-cycle to an algebraic one (measured contraction ~0.4 per V(2,2) cycle on the P2
stiffness, better than the 0.55 the Poisson disc's P1 operator gives). The pressure half is a
lumped diagonal and was always mesh-agnostic.

## Two upstream bugs this uncovered

Recorded here because anyone repeating this will hit them.

1. **`Mesh.gen_hollow_rectangle` and `gen_hollow_circle` are broken.** They call
   `gmsh.model.occ.cut`, which *consumes* both inputs, then ask for `getBoundary` of the inner
   one: `Unknown model face with tag 2`. `meshing.obstacle_mesh` takes the boundary of the cut
   *result* instead.
2. **The mixed P2/P1 assembly is silently wrong with `reorder=False`.** The constant pressure
   mode stops being in the kernel of `Bᵀ` on the free momentum rows — measured `2.1e-1`, against
   `6.6e-15` with `reorder=True`. Nothing raises. A self-consistent solve does not catch it
   (it returns a residual of 1e-17 and a plausible picture); only probing `K·(0,1)` does, which
   is `tests/test_unstructured_stokes.py::test_constant_pressure_mode_is_in_the_kernel`.
   `Mesh.gen_circle(order=2)` shows the same `1.3e-1`, so it is not safe for Taylor-Hood as it
   stands.

## Running

```bash
clsubmit experiments/stokes/unstructured/lr_sweep.txt   # stage 1: lr and the Schur weight
clsubmit experiments/stokes/unstructured/sweep.txt      # stage 2: 3 arms x 3 seeds
python   experiments/stokes/unstructured/compare.py     # table + figure, vs the structured runs
```

Stage 1 sweeps `omega` as well as the learning rate, because the structured study found it to be
the decisive knob for this arm (3.6 % → 2.4 % across it) and picked 256 for a Q2/Q1 operator on
a uniform grid. There is no reason that number transfers to a P2/P1 operator on a domain with a
hole.

## Result (2026-09-17, `gaot-model`)

Test relative FE-``L²``, mean ± sd over seeds 42/43/44. GAOT on the obstacle, FNO on the square.

| loss | square `65²` (u / p) | obstacle 3998 P2 (u / p) |
|------|---------------------:|-------------------------:|
| `data` (supervised) | 2.85 ± 0.21 / 2.81 ± 0.21 % | 3.27 ± 0.13 / 1.79 ± 0.15 % |
| `pls` (preconditioned) | 2.38 ± 0.27 / 1.69 ± 0.23 % | **1.55 ± 0.17 / 0.85 ± 0.04 %** |
| `galerkin` (bare residual) | 31.27 ± 9.64 / 10.92 ± 2.30 % | 30.56 ± 1.28 / 11.40 ± 0.83 % |
| PI-DeepONet (strong form, autodiff) | 32.99 ± 1.09 / 34.25 ± 1.37 % | 44.44 ± 4.02 / 35.09 ± 1.08 % |
| PINO (strong form, finite differences) | 2.09 ± 0.26 / 1.50 ± 0.24 % | **not applicable** |

**The unstructured column has a physics-informed competitor, and that is deliberate.** PINO's
residual is a finite-difference stencil and does not exist on this mesh, but "the baseline
cannot run here" proves nothing on its own — it invites the reading that the venue was chosen to
exclude it. PI-DeepONet differentiates the same strong form by autodiff through a coordinate
trunk, so it *can* follow, and it is 29× behind the label-free FEM arm (44.4 % against 1.55 %).
That is the capability claim with a competitor in it rather than an empty chair.

Its knobs did transfer, unlike `pls`'s: stage 1 on this mesh re-picked the structured study's
own choice (lr 1e-4, continuity weight 100) — 41.75 % at that cell against 80.89 % at `w=1`,
100.0 % at lr 3e-4 and 99.99 % at `w=1000`. The BC is `--pideeponet_bc zero`, a relative
velocity penalty at the mesh's boundary nodes; an autodiff Laplacian at an interior collocation
point cannot see a mask, which is why the structured arm needs the penalty too. The mollifier is
refused here — no closed form vanishes on both boundary components of a domain with a hole.

**On the structured PINO number.** Re-run at its selected settings to confirm from data, not
from reading the code, that it imposes the BC by boundary *zeroing* and not by a mollifier:
velocity 1.90 / 1.87 / 2.35 % against the archived 1.97 / 1.89 / 2.40 %. It does — there is no
mollifier path for Stokes at all (`cli.py` skips the hard-BC wrapper when `pde == "stokes"`, and
`PINOStokesTrainer._predict` zeroes the velocity ring itself). The *Poisson* archived runs are a
different story: those are mollified, and the mollifier is worth 24× there.

**And on what that 2.09 % actually measures.** The FEM `galerkin` control is *unweighted* while
PINO carries a tuned continuity weight (`--pi_div_weight 300`). At `w=1` PINO scores 46.05 %
velocity — worse than the unweighted FEM control's 31.27 % — and the gradient cosine between the
two objectives is 0.993 at `w=1`, falling to 0.83 at `w=300`. So the two are very nearly the
same objective and the weight, not the residual's form, separates them. `--stokes_div_weight`
now gives the FEM arm the same knob; see `experiments/stokes/galerkin_divweight/`.

**Read down a column.** The two domains are different problems with different reference
solutions, so the cross-column numbers are not a like-for-like error — what transfers, or fails
to, is the ordering within each.

1. **The label-free arm beats the supervised one by about a factor of two, on both fields.**
   1.55 / 0.85 % against 3.27 / 1.79 %, with non-overlapping seed ranges. On the square the same
   comparison was a modest win (2.38 / 1.69 against 2.85 / 2.81); here it is decisive. Both arms
   target the same discrete Taylor-Hood solution — one through labels, one through the residual
   whose zero it is — so this says the residual is the better-conditioned route to it, not that
   it is solving something easier.
2. **The bare residual transfers unchanged, and that is the control working.** 30.6 / 11.4 %
   against 31.3 / 10.9 %: the objective is as bad on one domain as the other, which is what says
   the `pls` result is about the preconditioner rather than about the geometry. One difference
   is worth noting: on the obstacle its best epoch is **64–77**, not 499 — it peaks early and
   then drifts, where on the square it was still descending at the budget's end.
3. **`omega` had to be re-selected, and where it landed is the interesting part.** See below.

### The Schur weight does not transfer, and the conditioning optimum does

Stage 1 on this mesh (seed 42, lr 1e-3):

| `omega` | 0.5 | 4 | **16** | 32 | 64 | 256 |
|---------|----:|--:|-------:|---:|---:|----:|
| velocity | 2.58 % | 1.63 % | **1.69 %** | 3.35 % | 4.25 % | 22.94 % |
| pressure | 1.25 % | 0.95 % | **0.79 %** | 1.40 % | 1.62 % | 3.93 % |

The structured study selected **256** for this arm and called the weight its decisive knob. On
this mesh 256 is catastrophic: run at it, the three stage-2 seeds gave velocity errors of
99.92 / 10.32 / 15.61 % (kept under `superseded/`). The optimum is a broad plateau over 4–16 and
falls off hard on either side.

The failure at 256 is not a training failure but a mis-weighted metric, and the sample panels
show it directly (`output/stokes/unstructured/lr/pls_om256/visualization/`): the **pressure is
nearly right** (3.96 % on sample 0) while the **velocity has lost the jets past the obstacle**
(33.07 %), with the error concentrated in a wake-shaped region. `omega` is the weight of the
pressure block in `P = diag(Â⁻¹/μ, ω μ/diag(M_p))`, so a large `omega` makes the norm the loss
measures in 256x pressure-dominated and the optimiser serves pressure at velocity's expense.
That is the same failure mode the structured Stokes dataset's field scaling exists to prevent
(an unweighted supervised loss there is ~2500x pressure-dominated), reappearing through the
preconditioner rather than through the loss.

The error *texture* separates the three arms as cleanly as the numbers do, at the same colour
scale (velocity error peak against a field range of 2.7): `pls` at `omega=16` is mesh-scale
grain at 0.07 — converged to discretisation level; `galerkin` is smooth large-scale blobs at 0.5
sitting on the velocity maxima — the low-frequency modes a `κ = O(h⁻⁴)` objective releases last;
`pls` at `omega=256` is 0.9 in a wake around the obstacle.

**16 is also the value this repo records as the *conditioning* optimum** for the block
preconditioner used as a norm weight — the minimiser of `κ(KPK)`, measured h-independent. On the
uniform grid the empirical optimum sat a factor of 16 away from the conditioning one; here the
two coincide. The natural reading is that on the structured operator something other than
conditioning was being compensated for by the large `omega`, and that on a genuinely
unstructured operator the theory's value is the one that works. It is one mesh, so this is an
observation to test at another resolution, not a result.

Cost: ~2.9 s/epoch (`data`, `galerkin`) and ~3.9 s/epoch (`pls`, one algebraic V-cycle per
step) on an RTX 4090; ~25–35 min per run.

## What would count as a problem

* **`data` far off its structured 2.85 %.** The supervised arm has labels, so a large error
  points at the plumbing — the pressure gather, the two scalings, the projections — rather than
  at the objective. Read it first.
* **`pls` not beating `galerkin` by a wide margin.** Bare least squares is 21.6 % structured.
  If the gap closes here, suspect the algebraic V-cycle's contraction before the argument, and
  check `omega` — those are the two things that changed.
* **Pressure much worse than velocity.** The pressure channel is emitted on the P2 node set and
  only read at the corners; if that gather were misaligned the velocity would still look fine.
  `test_from_nodes_splits_the_three_channels` pins it statically.
