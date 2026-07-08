# Experiment: blend-strength sweep (residual → supervised transition)

Sweep the convex-blend preconditioner strength `t`, with `P = (1−t)·I + t·A⁻¹`, and watch
training interpolate from residual-driven (`t=0`, `P=I`) to supervised-equivalent (`t=1`,
`P=A⁻¹`). The point is that the loss *conditioning* governs training: as `t→1` the FNO
trains with supervised-like dynamics without ever using labels. See `../../preconditioner_notes/`
for the theory.

The same sweep applies to different **losses**; each is a **variant** in its own subfolder,
sharing the one plot script here (`plot_sweep.py`: overlay of relative-L2 vs epoch per `t`,
plus a collapse of best relative-L2 vs conditioning `κ(H)`).

## Variants

| Variant | Loss | Preconditioning | Conditioning axis |
|---|---|---|---|
| [`pls/`](pls/README.md) | PLS `½‖P(Au−b)‖²` | `P` inside the squared loss | `κ(H)=κ(A)²` (`t=0`) → `1` |
| [`deepritz/`](deepritz/README.md) | preconditioned Deep Ritz (surrogate `∂L/∂u = P r`, hard BC) | `P` on the energy gradient | `κ(PA)=κ(A)` (`t=0`) → `1` |

Both use the same `K=4` dataset, 10 strengths, 500 epochs, adam.

## Conventions

- Each variant writes to `output/sweep_blend/<variant>/` (git-ignored).
- Plot a variant with the shared script pointed at its results:
  `python experiments/sweep_blend/plot_sweep.py --results_dir output/sweep_blend/<variant>/results`.
- Pull a variant's artifacts from the cluster:
  `rsync -avz euler:~/TensorPILS/output/sweep_blend/<variant> ~/Documents/TensorPILS/output/sweep_blend/`.

See each variant's `README.md` for its facts table and exact run commands.
