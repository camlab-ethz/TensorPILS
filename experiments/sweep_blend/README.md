# Experiment: blend-strength sweep (residual → supervised transition)

Sweep the convex-blend preconditioner strength `t` in the PLS loss `½‖P(Au−b)‖²`, where
`P = (1−t)·I + t·A⁻¹`. This interpolates the training problem from the **residual**
least-squares problem `½‖Au−b‖²` (`t=0`, `P=I`, Hessian conditioning `κ(H)=κ(A)²`) to the
**supervised** least-squares problem `½‖u−u★‖²` (`t=1`, `P=A⁻¹`, `κ(H)=1`). The point is to
show that the loss *conditioning* governs training: as `t→1` the FNO trains with
supervised-like dynamics, without ever using labels. See `preconditioner_notes/` for the
theory.

## Facts

| | |
|---|---|
| Loss | `pls` (`½‖P(Au−b)‖²`) |
| Preconditioner | `blend`, `t ∈ {0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0}` |
| Dataset | 2D Poisson, `K=4`, grid `64²`, `n_train=1024`, `n_val=128`, `n_test=256`, `seed=42` |
| Optimizer | `adam`, default cosine lr `1e-3 → 1e-4` |
| Epochs / batch | `500` / `32` |
| Output dir | `output/sweep_blend/` (git-ignored) |
| Recorded | per-epoch val relative-L2, and `κ(PA)`/`κ(H)` per run (in the results JSON) |

## Run

Single configuration (one strength):

```bash
python -m tensorpils.cli --loss pls --precond_kind blend --precond_strength 0.5 \
    --n_train 1024 --n_val 128 --n_test 256 -k 4 --epochs 500 --batch_size 32 \
    --device cuda --output_dir output/sweep_blend
```

Full sweep on Euler (SLURM array over all 10 strengths; submit from `~/TensorPILS`):

```bash
mkdir -p logs
sbatch experiments/sweep_blend/sweep_blend.sbatch
```

## Plot

```bash
python experiments/sweep_blend/plot_sweep.py --results_dir output/sweep_blend/results
```

Produces, in `output/sweep_blend/`:
- `sweep_overlay.png` — validation relative-L2 vs epoch, one curve per `t`.
- `sweep_collapse.png` — best relative-L2 vs conditioning `κ(H)` (the collapse plot).
