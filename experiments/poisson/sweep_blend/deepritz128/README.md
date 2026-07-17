# Variant: Deep Ritz — blend-strength sweep on the **128² grid** (fast DST preconditioner)

Same experiment as [`../deepritz/`](../deepritz/README.md) — sweep the convex-blend strength `t` in
the **preconditioned Deep Ritz** loss, whose surrogate descent direction `P·r` interpolates from the
plain energy gradient (`t=0`, `P=I` → `r=Au−b`) to the supervised/Newton direction (`t=1`, `P=A⁻¹`
→ `u−u★`) — but on the refined **128²** grid. There the dense eigendecomposition of `A` (`N≈15876`)
is heavy, so the preconditioner is built with the exact **fast sine transform**
(`--precond_method sine`), which applies `P` in `O(N^{1.5})` with no eigensolver. See
`../../../notes/preconditioner_notes/` (§Fast realization on the regular grid).

This is a **sanity run first**: `20` epochs and a coarse `t` grid to confirm the fine-grid pipeline
trains. Retune `GRID / EPOCHS / TVALS` at the top of `sweep.sbatch` once it looks right.

## Facts

| | |
|---|---|
| Loss | `deepritz --precondition` (preconditioned energy-gradient descent) |
| Preconditioner | `blend` + `--precond_method sine`; `t ∈ {1.0, 0.9, 0.5, 0.25, 0.01, 0.0}` (+ optional `multigrid` reference, task 6) |
| Dataset | 2D Poisson, `K=4`, grid `128²`, `n_train=1024`, `n_val=128`, `n_test=256`, `seed=42` |
| Optimizer | `adam`, default cosine lr `1e-3 → 1e-4` |
| Epochs / batch | `20` (sanity) / `32` |
| Output dir | `output/poisson/sweep_blend/deepritz128/` (git-ignored) |
| Recorded | per-epoch val relative-L2, and `κ(PA)`/`κ(H)` per run (in the results JSON) |

## Run

Full sanity sweep on Euler (SLURM array `0-5`: the six `t` values; submit from `~/TensorPILS`):

```bash
mkdir -p logs
sbatch experiments/poisson/sweep_blend/deepritz128/sweep.sbatch
# add the multigrid baseline too:
sbatch --array=6 experiments/poisson/sweep_blend/deepritz128/sweep.sbatch
```

Single configuration locally:

```bash
python -m tensorpils.cli --loss deepritz --precondition --precond_kind blend --precond_strength 0.5 \
    --precond_method sine --grid_resolution 128 \
    --n_train 1024 --n_val 128 --n_test 256 -k 4 --epochs 20 --batch_size 32 \
    --device cuda --output_dir output/poisson/sweep_blend/deepritz128
```

## Plot

```bash
python experiments/poisson/sweep_blend/plot_sweep.py --results_dir output/poisson/sweep_blend/deepritz128/results
```

Produces `sweep_overlay.png` (val relative-L2 vs epoch, one curve per `t`) and `sweep_collapse.png`
(best relative-L2 vs conditioning `κ(H)`) in `output/poisson/sweep_blend/deepritz128/`.
