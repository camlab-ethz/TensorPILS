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

Both use the same `K=4` dataset, 10 blend strengths + 1 multigrid reference, 500 epochs, adam.

## Conventions

- Each variant writes to `output/sweep_blend/<variant>/` (git-ignored).
- The sbatch array is `0-10`: tasks `0-9` are the blend strengths, task `10` is the multigrid
  reference. To add only the GMG line to an already-run sweep: `sbatch --array=10 <sweep.sbatch>`.
- Plot a variant with the shared script pointed at its results:
  `python experiments/sweep_blend/plot_sweep.py --results_dir output/sweep_blend/<variant>/results`.
  Optional flags: `--smooth` (centered moving average, raw kept faint; `--smooth_window N` to
  tune), on by default `--annotate`/`--no-annotate` (stamp the loss/(surrogate-)gradient formulas).
- Pull a variant's artifacts from the cluster:
  `rsync -avz euler:~/TensorPILS/output/sweep_blend/<variant> ~/Documents/TensorPILS/output/sweep_blend/`.

See each variant's `README.md` for its facts table and exact run commands.
