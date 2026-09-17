# Poisson on an unstructured disc — does any of this survive leaving the grid?

## The question

Every result in this repo was produced on a structured grid, because the FNO needs an image and
the geometric V-cycle needs a grid hierarchy. Both halves of the method were therefore untested
where it actually matters for a PDE solver: an arbitrary mesh.

Two pieces were added to make that reachable, and this experiment is the first run that uses
both at once:

* **`--model gaot`** (`tensorpils/gaot/`) — encodes a point cloud onto a fixed structured
  *latent* token grid, so the transformer never sees the physical mesh.
* **`--precond_kind amg`** (`tensorpils/preconditioners/algebraic.py`) — a V-cycle built from
  the assembled matrix rather than from the grid.

Neither alone is enough: GAOT with the geometric V-cycle has no hierarchy to build, and AMG with
an FNO has no image to consume. The `pls` arm here is the first place they have to work together.

The question is not "is GAOT good" — that was answered on the structured grid
(`experiments/poisson/gaot_arch`, where it matched the FNO to within seed spread). It is:

> Does the loss ranking `pls ≈ data ≪ galerkin` hold when the domain is a disc, the mesh is a
> Gmsh triangulation, and the preconditioner is algebraic?

## Setup

Inscribed disc, centre `(0.5, 0.5)`, radius `0.5`. `K=4` and `r=-0.5` are the structured
dataset's, so the source distribution is unchanged; only the domain and the mesh differ.

The mesh is sized to **match the structured runs' node count**, which is what fixes the model
cost and the dimension of the discrete problem:

| mesh | nodes | interior DOFs | `h = sqrt(area/N)` | `κ(A_int)` |
|------|------:|--------------:|-------------------:|-----------:|
| structured square `64²` | 4096 | 3844 | 0.01562 | 804 |
| **disc `chara_length=0.015`** | **4205** | **4032** | 0.01367 | 1409 |
| disc `chara_length=0.0175` | 3103 | 2956 | 0.01591 | 1003 |

The two matchings disagree, and the choice is on the record: at equal node count the disc mesh
is 13 % finer in spacing and `κ(A)` is 1.75× the square's; at equal spacing it has 24 % fewer
nodes. Node count wins here because the effect being measured — `pls` against `galerkin` — is
two orders of magnitude, so a 1.75× difference in `κ(A)` (or 3× in `κ(AᵀA)`, which is what the
bare-residual arm actually sees) cannot account for anything in the table. Note also that even
at *matched* spacing the disc is worse conditioned than the square (1003 vs 804): that is the
geometry and the element type, not the refinement.

Anything coarser than this is a smoke test, not a result — `tests/test_unstructured_poisson.py`
runs at `chara_length=0.09` (~300 nodes) on purpose, and those numbers mean nothing.

**Labels must come from the FEM solve, and this is not a convenience.**
`PoissonMultiFrequency` is a sum of `sin(iπx) sin(jπy)`. That vanishes on the boundary of the
unit *square*; the disc's boundary runs through the interior of the square, where the field is
at full amplitude. Measured on this mesh, the closed form reaches `2.4e-2` on the circle against
an interior peak of `2.6e-2` — about 90 %. So `--dataset_solution analytic` would not fail, it
would simply train every arm against a field that does not solve the problem. The CLI refuses
it, `meshing.circle_mesh` documents it, and
`tests/test_unstructured_poisson.py::test_analytic_solution_is_invalid_on_the_disc` pins it.

A side effect is welcome: the label is now the *discrete* solution the residual losses target,
so the discretisation floor that separates `data` from `galerkin`/`pls` on analytic labels is
gone here, exactly as in the `--dataset_solution fem` structured runs.

## How to read the disc-vs-square gap — the square task is 16-dimensional

Measured here, and it reframes the whole table:

| domain | FEM solution outside the span of the `K²=16` sine modes |
|--------|--------------------------------------------------------:|
| square `64²` | **0.0000 %** |
| disc 4205 | **9.65 %** (max 18.0 %) |

On a uniform grid the discrete sines are exact eigenvectors of the `Q1` stiffness **and** mass
matrices, so `A u = M f` with `f` in the 16-mode span returns `u` in the same span — exactly.
Fitting the coefficient map directly confirms it is not merely low-dimensional but *diagonal*:
`‖offdiag‖/‖diag‖ = 1.5e-7` and the linear fit's residual is `1.9e-7`.

So the structured Poisson benchmark is, in the right basis, **a diagonal linear map on 16
numbers**. A model only has to realise the nodal→spectral transform and back — which is what an
FNO does natively, and is a large part of why it reaches 0.59 % there. Nothing is wrong with
that benchmark, but it bounds what it can demonstrate.

On the disc that structure is gone: the sine modes do not satisfy the boundary condition, are
not eigenvectors of anything, and the solution genuinely leaves their span by ~10 %. The target
is a richer function class, not a rescaled one.

**Consequence for the table: a disc number is not a degraded square number.** The `data` arm
going 0.55 % → 1.66 % is mostly the task changing, not the architecture failing — the solution
norms are within 1.2× of each other (7.4e-3 vs 6.1e-3), so it is not a normalisation artifact
either. Compare *within* a column. The claim under test is the ordering of the losses, and that
is exactly the comparison the change of domain leaves intact.

Learning rate is `1e-3` for all three arms, not swept again. Justification, not laziness: the
structured GAOT lr sweep (`experiments/poisson/gaot_arch/lr_sweep.txt`) found `1e-3` to be an
**interior** optimum for `data`, `pls` *and* `galerkin` — same architecture, same problem size,
same source distribution. If an arm here looks anomalous, sweep it before drawing a conclusion.

The mesh is generated once into `$TENSORPILS_CACHE_PATH/meshes/disc_h0.015.msh` and read by
every job, so all nine runs are on byte-identical geometry rather than on nine separate Gmsh
invocations that merely ought to agree.

## What is *not* available here, and why

| | reason |
|---|---|
| `--model fno` / `deeponet` | no image: the FFT and the fixed sensor grid both need one |
| `--precond_kind multigrid` / `blend` / `power` | geometric hierarchy / grid eigendecomposition |
| `--loss pino` / `pideeponet` | strong-form finite differences on an image |
| `--dataset_solution analytic` | wrong on this domain (above) |
| `--stream` | every label costs a sparse solve |

All five are refused by `_validate_mesh_args` in `cli.py` with the reason, rather than failing
somewhere inside a job. The geometric V-cycle is the one that would have been worst: it
*constructs* against this problem without complaint and only dies on a shape mismatch at its
first apply.

## Running

```bash
clsubmit experiments/poisson/unstructured/sweep.txt     # 3 losses x 3 seeds
python   experiments/poisson/unstructured/compare.py    # table + figure, vs the structured runs
```

From a compute node `clsubmit` runs the sweep **in place** instead of submitting (see the note
in the sweep file).

## Result (2026-09-17, `gaot-model` @ 5006f9f)

Test relative L2 (FEM norm), mean over seeds 42/43/44, GAOT throughout:

| loss | square `64²` | disc 4205 |
|------|-------------:|----------:|
| `data` (supervised) | 0.55 % (0.51–0.61) | 1.66 % (1.61–1.68) |
| `pls` (preconditioned residual) | 0.61 % (0.56–0.71) | **1.65 %** (1.61–1.71) |
| `galerkin` (bare residual) | 37.57 % (35.64–40.16) | 73.59 % (57.71–94.37) |

**The ranking transfers, and on the disc the label-free arm is not merely competitive — it ties
the supervised one.** `pls` 1.65 % against `data` 1.66 %, with overlapping seed ranges, while
the bare residual is 45× worse. That is the claim this repo is about, now demonstrated on a
domain with no grid, a mesh with no structure and a preconditioner built from nothing but the
matrix.

Two readings that the numbers support and one that they do not:

1. **`pls` catching `data` exactly is the expected shape, not luck.** The disc's labels are the
   FEM solve, so `data` targets the discrete solution and `pls` minimises the residual whose
   zero *is* that discrete solution. With a good enough `P` the two objectives have the same
   minimiser, and here they land on it equally well — which is precisely the argument for
   preferring the one that needs no labels.
2. **The bare residual degrades more than everything else** (37.6 % → 73.6 %) and its seed
   spread explodes (57.7–94.4 % against 35.6–40.2 %). Consistent with conditioning: `κ(A)` is
   1.75× the square's here, so `κ(AᵀA)` is ~3×, and that arm is the only one exposed to it. It
   is also still descending at epoch 500 on both domains, so this is a rate, not a floor.
3. **What the numbers do NOT say is that GAOT degrades on an unstructured mesh.** See the
   section above: the square task is a diagonal map on 16 coefficients and the disc task is not,
   so the 0.55 % → 1.66 % column shift is mostly the problem getting harder. The
   architecture-level question was already settled on the square, where GAOT matched the FNO.

Cost: ~3.0 s/epoch (`data`, `galerkin`) and ~3.5 s/epoch (`pls`, one extra AMG V-cycle per
step) on an RTX 4090, so ~25–30 min per run.

## What would count as a problem

* **`data` far off its structured value (0.55 %).** The supervised arm has labels, so it tests
  the plumbing — the mesh's node ordering reaching the model, the FEM solve, the coordinates —
  rather than the objective. It is the first thing to read.
* **Latent tokens starved of nodes.** ~15 % of the latent grid falls outside the disc and sees
  nothing; that is expected and harmless (an empty neighbourhood is a zero encoding, which is
  how GAOT handles arbitrary domains in its own paper). What is *not* harmless is a mesh node
  with no latent token in range, which would decode to zero for reasons unrelated to training.
  Measured 0 of 4205 here, and guarded by `test_every_mesh_node_can_be_decoded`.
* **`pls` no better than `galerkin`.** On the structured grid the gap is two orders of
  magnitude. If it collapses here, suspect the AMG cycle before the argument: its measured
  contraction on this matrix is ~0.55 per V(2,2) cycle at the default AmgX settings, against
  ~0.03 for the geometric cycle on the structured operator. That is a tuning gap
  (`--amg_sweeps`, `--amg_smoother`, `--amg_algorithm`), and it bounds how good a norm `P` is.
