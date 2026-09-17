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
