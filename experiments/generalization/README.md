# Experiment: OOD generalization vs. preconditioner strength

Ask whether the loss *geometry* set by the convex-blend strength `t` affects
**out-of-distribution generalization**. All runs train the PLS loss on the same `K=4`
sources (no labels); `t` only changes the preconditioner metric, interpolating from
residual-driven (`t→0`) to supervised-MSE-equivalent (`t=1`). We train long enough for the
ill-conditioned small-`t` runs to converge, and track relative-L2 **each epoch** on the
in-distribution `K=4` validation set plus out-of-distribution `K=6` and `K=8` sources
(higher-frequency `f` on the *same* `64²` grid / operator — genuine spectral extrapolation).

**Reading the results:** first confirm each `t`'s in-distribution (`K=4`) curve has
*plateaued* — otherwise a worse OOD number reflects under-training, not worse
generalization (compare at matched in-distribution error, not just matched epochs). The
signal to look for on `K=6`/`K=8` is whether the supervised `t=1` curve turns *up*
(overfitting the `K=4` spectrum) while smaller-`t` curves stay flatter.

## Facts

| | |
|---|---|
| Loss | `pls` (`½‖P(Au−b)‖²`) |
| Preconditioner | `blend`, `t ∈ {0.05, 0.1, 0.25, 1.0}` |
| Train dataset | 2D Poisson, `K=4`, grid `64²`, `n_train=1024`, `n_val=128`, `n_test=256`, `seed=42` |
| OOD eval sets | `K=6`, `K=8`, `n=128` each, `seed=123` (fixed across runs) |
| Optimizer | `adam`, default cosine lr `1e-3 → 1e-4` |
| Epochs / batch | `2500` / `32` |
| Output dir | `output/generalization/` (git-ignored) |
| Recorded | per-epoch relative-L2 on `K=4` (val) and on each OOD `K` (`stats.ood_rel_l2`) |

## Run

Single configuration (one strength, with OOD eval):

```bash
python -m tensorpils.cli --loss pls --precond_kind blend --precond_strength 0.25 \
    --n_train 1024 --n_val 128 --n_test 256 -k 4 --ood_k 6 8 --ood_n_val 128 \
    --epochs 2500 --batch_size 32 --device cuda --output_dir output/generalization
```

Full sweep on Euler (SLURM array over the 4 strengths; submit from `~/TensorPILS`):

```bash
mkdir -p logs
sbatch experiments/generalization/sweep_generalization.sbatch
```

## Plot

```bash
python experiments/generalization/plot_generalization.py \
    --results_dir output/generalization/results
```

Produces, in `output/generalization/`, one figure per eval-K — each overlaying the four
`t`-curves (relative-L2 vs epoch):
- `generalization_K4.png` (in-distribution), `generalization_K6.png`, `generalization_K8.png`.
