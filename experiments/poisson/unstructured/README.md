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

Inscribed disc, centre `(0.5, 0.5)`, radius `0.5`. `chara_length=0.015` gives **4205 nodes**
against the structured runs' `64² = 4096` — deliberately, so the two are comparable in problem
size. `K=4` and `r=-0.5` are the structured dataset's, so the source distribution is unchanged;
only the domain and the mesh differ.

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
