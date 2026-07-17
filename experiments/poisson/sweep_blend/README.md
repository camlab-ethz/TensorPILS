# Experiment: blend-strength sweep (residual → supervised transition)

Sweep the convex-blend preconditioner strength `t`, with `P = (1−t)·I + t·A⁻¹`, and watch
training interpolate from residual-driven (`t=0`, `P=I`) to supervised-equivalent (`t=1`,
`P=A⁻¹`). The point is that the loss *conditioning* governs training: as `t→1` the FNO
trains with supervised-like dynamics without ever using labels. See `../../notes/preconditioner_notes/`
for the theory.

Each sweep also carries a **multigrid (GMG)** run in its default setup — the preconditioner
that actually works in practice — as a reference. GMG has no single closed-form conditioning,
so it appears as a dashed black reference curve on the overlay and is omitted from the collapse.

The same sweep applies to different **losses**; each is a **variant** in its own subfolder,
sharing the one plot script here (`plot_sweep.py`: overlay of relative-L2 vs epoch per `t`,
plus a collapse of best relative-L2 vs conditioning `κ(H)`).

## Variants

| Variant | Loss | Preconditioning | Conditioning axis |
|---|---|---|---|
| [`pls/`](pls/README.md) | PLS `½‖P(Au−b)‖²` | `P` inside the squared loss | `κ(H)=κ(A)²` (`t=0`) → `1` |
| [`deepritz/`](deepritz/README.md) | preconditioned Deep Ritz (surrogate `∂L/∂u = P r`, hard BC) | `P` on the energy gradient | `κ(PA)=κ(A)` (`t=0`) → `1` |
| [`pls128/`](pls128/README.md), [`pls256/`](pls256/README.md) | PLS on `128²` / `256²` grids | fast **DST** preconditioner (`--precond_method sine`) | as `pls/`, refined grid |
| [`deepritz128/`](deepritz128/README.md), [`deepritz256/`](deepritz256/README.md) | Deep Ritz on `128²` / `256²` grids | fast **DST** preconditioner (`--precond_method sine`) | as `deepritz/`, refined grid |

The base `pls/`/`deepritz/` variants use the `64²` grid with the dense eigendecomposition, 10 blend
strengths + 1 multigrid reference, 500 epochs. The `*128`/`*256` variants refine the grid to `128²`
/ `256²`, where the dense eigendecomposition is infeasible, so they build the **same** spectral `P`
via the exact fast sine transform (`--precond_method sine`; see `../../notes/preconditioner_notes/`
§Fast realization). They start as **sanity runs** (20 epochs, a coarse `t` grid set at the top of
each `sweep.sbatch`) — retune `GRID / EPOCHS / TVALS` there. All use the same `K=4` dataset, adam.

## Conventions

- Each variant writes to `output/poisson/sweep_blend/<variant>/` (git-ignored).
- For `pls/`/`deepritz/` the sbatch array is `0-10`: tasks `0-9` are the blend strengths, task `10`
  is the multigrid reference (`sbatch --array=10 <sweep.sbatch>` adds only the GMG line). The
  `*128`/`*256` variants instead expose an editable `TVALS` array at the top of `sweep.sbatch`
  (array `0-5` by default; the GMG baseline is task `6`, added via `sbatch --array=6 …`).
- Plot a variant with the shared script pointed at its results:
  `python experiments/poisson/sweep_blend/plot_sweep.py --results_dir output/poisson/sweep_blend/<variant>/results`.
  Optional flags: `--smooth` (centered moving average, raw kept faint; `--smooth_window N` to
  tune), on by default `--annotate`/`--no-annotate` (stamp the loss/(surrogate-)gradient formulas).
- Pull a variant's artifacts from the cluster:
  `rsync -avz euler:~/TensorPILS/output/poisson/sweep_blend/<variant> ~/Documents/TensorPILS/output/poisson/sweep_blend/`.

See each variant's `README.md` for its facts table and exact run commands.
