# Scenario: data-driven losses, train K=16 → OOD K=20

Out-of-distribution generalization of **supervised** training in the K=16 regime: does the
*metric* of the data loss change how well the FNO extrapolates to higher-frequency sources?
Two losses are trained on `K=16` and evaluated each epoch on the in-distribution `K=16` val
set and out-of-distribution `K=20`:

- **`data`** — supervised MSE (flat grid average). In the earlier `loss_comparison`, MSE was
  as good as (slightly better than) the true-L² loss, so it stands in for the L²-type data
  losses here.
- **`data_h1`** — supervised H¹₀ norm `½ eᵀ A e` (stiffness-weighted; prediction projected to
  zero boundary). This weights gradient/high-frequency error more heavily than MSE.

The question: does training in the H¹₀ metric — which emphasizes derivatives / higher
frequencies — help or hurt extrapolation to the higher-frequency `K=20` sources, versus plain
MSE?

**Deliberate extreme OOD:** `n_modes = 16`, so the FNO is band-limited *at* K=16 and *below*
K=20; the K=20 target has content the network cannot represent (expect a spectral-truncation
floor). The `64²` grid resolves K=20 (Nyquist ≈ 32); the cap is the FNO mode truncation.
As in `../pls_k16`, read the K=20 curves *relative to each other*, not against zero.

## Facts

| | |
|---|---|
| Losses | `data` (MSE), `data_h1` (H¹₀, boundary-projected) |
| Train dataset | 2D Poisson, `K=16`, grid `64²`, `n_train=1024`, `n_val=128`, `n_test=256`, `seed=42` |
| OOD eval sets | `K=20`, `n=128`, `seed=123` |
| FNO modes | `16×16` (**below** K=20 — deliberate truncation) |
| Optimizer | `adam`, default cosine lr `1e-3 → 1e-4` |
| Epochs / batch | `1000` / `32` |
| Output dir | `output/generalization/data_k16/` (git-ignored) |
| Recorded | per-epoch relative-L2 on `K=16` (val) and OOD `K=20` (`stats.ood_rel_l2`) |

## Run

Single loss (with OOD eval):

```bash
python -m tensorpils.cli --loss data_h1 --n_train 1024 --n_val 128 --n_test 256 \
    -k 16 --ood_k 20 --ood_n_val 128 --epochs 1000 --batch_size 32 \
    --device cuda --output_dir output/generalization/data_k16
```

Both losses on Euler (SLURM array; submit from `~/TensorPILS`):

```bash
mkdir -p logs
sbatch experiments/generalization/data_k16/sweep.sbatch
```

## Plot

```bash
python experiments/generalization/plot_generalization.py \
    --results_dir output/generalization/data_k16/results
```

Produces, in `output/generalization/data_k16/`, one figure per eval-K, each overlaying the
MSE and H¹₀ curves (relative-L2 vs epoch):
- `generalization_K16.png` (in-distribution), `generalization_K20.png` (OOD).
