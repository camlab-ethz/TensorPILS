# Scenario: PLS, train K=16 → OOD K=20  (Shizheng's regime)

Same OOD-generalization question as [`pls_k4/`](../pls_k4/README.md), but in the
higher-frequency regime Shizheng used: train the PLS loss on `K=16` sources across
convex-blend strengths `t`, and track relative-L2 **each epoch** on the in-distribution
`K=16` validation set plus out-of-distribution `K=20` sources.

**Deliberate design choice — extreme OOD:** the FNO keeps `n_modes = 16` per axis, so it is
band-limited *at* the K=16 training content and *below* the K=20 target. K=20 therefore has
spectral content the network **cannot represent**, so expect a spectral-truncation floor in
the K=20 error. This is intentional — the question here is how the loss geometry behaves
when the OOD target exceeds the model's representable band (not just higher-but-representable
frequencies as in `pls_k4/`). The `64²` grid still resolves K=20 (Nyquist ≈ 32); the cap is
purely the FNO mode truncation.

**Reading the results:** as in `pls_k4/`, confirm the in-distribution (`K=16`) curves have
plateaued before comparing OOD; and note the K=20 curves sit above the truncation floor, so
compare `t`-curves *relative to each other*, not against zero.

## Facts

| | |
|---|---|
| Loss | `pls` (`½‖P(Au−b)‖²`) |
| Preconditioner | `blend`, `t ∈ {0.05, 0.1, 0.25, 1.0}` |
| Train dataset | 2D Poisson, `K=16`, grid `64²`, `n_train=1024`, `n_val=128`, `n_test=256`, `seed=42` |
| OOD eval sets | `K=20`, `n=128`, `seed=123` |
| FNO modes | `16×16` (**below** K=20 — deliberate truncation) |
| Optimizer | `adam`, default cosine lr `1e-3 → 1e-4` |
| Epochs / batch | `2500` / `32` |
| Output dir | `output/poisson/generalization/pls_k16/` (git-ignored) |
| Recorded | per-epoch relative-L2 on `K=16` (val) and on OOD `K=20` (`stats.ood_rel_l2`) |

## Run

Single configuration (one strength, with OOD eval):

```bash
python -m tensorpils.cli --loss pls --precond_kind blend --precond_strength 0.25 \
    --n_train 1024 --n_val 128 --n_test 256 -k 16 --ood_k 20 --ood_n_val 128 \
    --epochs 2500 --batch_size 32 --device cuda --output_dir output/poisson/generalization/pls_k16
```

Full sweep on Euler (SLURM array over the 4 strengths; submit from `~/TensorPILS`):

```bash
mkdir -p logs
sbatch experiments/poisson/generalization/pls_k16/sweep.sbatch
```

## Plot

```bash
python experiments/poisson/generalization/plot_generalization.py \
    --results_dir output/poisson/generalization/pls_k16/results
```

Produces, in `output/poisson/generalization/pls_k16/`, one figure per eval-K — each overlaying the
four `t`-curves (relative-L2 vs epoch):
- `generalization_K16.png` (in-distribution), `generalization_K20.png`.
