# Variant: preconditioned Deep Ritz — blend-strength sweep

The Deep Ritz analogue of the [`pls/`](../pls/README.md) sweep. Instead of the squared PLS
loss, the **energy gradient is preconditioned** by `P = (1−t)·I + t·A⁻¹` (surrogate loss
`⟨u, (P r).detach()⟩`, so `∂L/∂u = P r`). Sweeping `t` interpolates the descent direction
from the plain Deep Ritz energy gradient (`t=0`, `P=I` → `r = Au−b`) to the
supervised/Newton direction (`t=1`, `P=A⁻¹` → `u−u★`). As with PLS, this shows the loss
*conditioning* driving training toward supervised-like dynamics — here in the surrogate-form
metric, where the relevant conditioning is `κ(PA)` (running from `κ(A)` at `t=0` to `1` at
`t=1`). Preconditioned Deep Ritz uses hard boundary conditions (implied by `--precondition`).

`plot_sweep.py` auto-detects the loss form from each run's `loss_type` and plots the
**morally correct** conditioning per case: `κ(PA)` for this Deep Ritz surrogate (vs `κ(H)=κ(PA)²`
for the squared PLS loss). So the collapse x-axis here is `κ(PA)`, not the squared `κ(H)`.

## Facts

| | |
|---|---|
| Loss | `deepritz` with `--precondition` (surrogate `∂L/∂u = P r`, hard BC) |
| Preconditioner | `blend`, `t ∈ {0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0}` |
| Dataset | 2D Poisson, `K=4`, grid `64²`, `n_train=1024`, `n_val=128`, `n_test=256`, `seed=42` |
| Optimizer | `adam`, default cosine lr `1e-3 → 1e-4` |
| Epochs / batch | `500` / `32` |
| Output dir | `output/sweep_blend/deepritz/` (git-ignored) |
| Recorded | per-epoch val relative-L2, and `κ(PA)`/`κ(H)` per run (in the results JSON) |

## Run

Single configuration (one strength):

```bash
python -m tensorpils.cli --loss deepritz --precondition --precond_kind blend \
    --precond_strength 0.5 --n_train 1024 --n_val 128 --n_test 256 -k 4 \
    --epochs 500 --batch_size 32 --device cuda --output_dir output/sweep_blend/deepritz
```

Full sweep on Euler (SLURM array over all 10 strengths; submit from `~/TensorPILS`):

```bash
mkdir -p logs
sbatch experiments/sweep_blend/deepritz/sweep.sbatch
```

## Plot

```bash
python experiments/sweep_blend/plot_sweep.py --results_dir output/sweep_blend/deepritz/results
```

Produces, in `output/sweep_blend/deepritz/`:
- `sweep_overlay.png` — validation relative-L2 vs epoch, one curve per `t`.
- `sweep_collapse.png` — best relative-L2 vs conditioning `κ(H)`.
