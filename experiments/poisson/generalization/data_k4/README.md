# Scenario: data-driven losses, train K=4 → OOD K=6

Out-of-distribution generalization of **supervised** training in the K=4 regime — the
in-band companion to [`data_k16/`](../data_k16/README.md). Two data losses are trained on
`K=4` and evaluated each epoch on the in-distribution `K=4` val set and out-of-distribution
`K=6`:

- **`data`** — supervised MSE (flat grid average).
- **`data_h1`** — supervised H¹₀ norm `½ eᵀ A e` (stiffness-weighted; prediction projected to
  zero boundary). Weights gradient / higher-frequency error more heavily than MSE.

The question: does training in the H¹₀ metric help or hurt extrapolation to the
higher-frequency `K=6` sources, versus plain MSE?

Unlike `data_k16`, here `n_modes = 16` **fully represents** `K=6` (Nyquist ≈ 32), so this is
genuine OOD *within* the FNO's representable band — no truncation floor; the K=6 curves can
be read against zero.

## Facts

| | |
|---|---|
| Losses | `data` (MSE), `data_h1` (H¹₀, boundary-projected) |
| Train dataset | 2D Poisson, `K=4`, grid `64²`, `n_train=1024`, `n_val=128`, `n_test=256`, `seed=42` |
| OOD eval sets | `K=6`, `n=128`, `seed=123` |
| FNO modes | `16×16` (K=6 fully representable) |
| Optimizer | `adam`, default cosine lr `1e-3 → 1e-4` |
| Epochs / batch | `1000` / `32` |
| Output dir | `output/poisson/generalization/data_k4/` (git-ignored) |
| Recorded | per-epoch relative-L2 on `K=4` (val) and OOD `K=6` (`stats.ood_rel_l2`) |

## Run

Single loss (with OOD eval):

```bash
python -m tensorpils.cli --loss data_h1 --n_train 1024 --n_val 128 --n_test 256 \
    -k 4 --ood_k 6 --ood_n_val 128 --epochs 1000 --batch_size 32 \
    --device cuda --output_dir output/poisson/generalization/data_k4
```

Both losses on Euler (SLURM array; submit from `~/TensorPILS`):

```bash
mkdir -p logs
sbatch experiments/poisson/generalization/data_k4/sweep.sbatch
```

## Plot

```bash
python experiments/poisson/generalization/plot_generalization.py \
    --results_dir output/poisson/generalization/data_k4/results
```

Produces, in `output/poisson/generalization/data_k4/`, one figure per eval-K, each overlaying the
MSE and H¹₀ curves (relative-L2 vs epoch):
- `generalization_K4.png` (in-distribution), `generalization_K6.png` (OOD).
