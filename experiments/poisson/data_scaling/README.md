# Experiment: dataset-size scaling at fixed optimization budget (→ infinite data)

Physics-informed training needs no labels, so a fresh sample costs only a random draw plus
closed-form field evaluation — the **infinite-data limit** (streaming: no sample ever seen
twice) is a practical operating point. This experiment sweeps the training-set size `n` at a
**fixed optimization budget** and contrasts it with the streaming run. In the classical
error decomposition (approximation + estimation + optimization), streaming eliminates the
estimation error **exactly**: each SGD step is an unbiased gradient of the population risk.
Prediction: the error-vs-`n` curve decreases monotonically and saturates at the streaming
value. See `../../../notes/preconditioner_notes/infinite_data_limit.tex` for the theory.

## Fairness protocol

- **Budget**: every run gets the same `EPOCHS × STEPS` optimizer steps via `--steps_per_epoch`
  ("virtual epochs" of fixed step count). Finite datasets are sampled i.i.d. **with
  replacement** — the empirical-measure analogue of the stream, so the only thing varied
  across runs is the sampled measure (`μ_n → μ`), not the algorithm.
- **Early stopping**: `--patience` on the **validation** error (never the test set); small-`n`
  runs plateau/overfit before exhausting the budget and report their best-val checkpoint.
- **Shared eval sets**: `--fixed_eval` draws val/test from dedicated seeds (`seed+1`/`seed+2`),
  identical for every `n` and for the stream — comparisons are paired, not independently noisy.
  The finite train sets are nested across `n` (same seed), which further smooths the curve.
- **Reporting**: y-axis is the final **test** relative-L2 of the best-val checkpoint (computed
  once per run, after training).

## Facts

| | |
|---|---|
| Loss | `pls` + `multigrid` (GMG) preconditioner — the practical configuration |
| Dataset sizes | `n ∈ {32, 64, 128, 256, 512, 1024, 2048, 4096}` + streaming (task 8) |
| Budget | `500` virtual epochs × `32` steps × batch `32` (= the existing 500-epoch runs on `n=1024`) |
| Early stop | patience `50` virtual epochs on val |
| Eval sets | `n_val=128`, `n_test=256`, fixed across all runs (`--fixed_eval`) |
| Grid / K / seed | `64²` / `K=4` / `42` |
| Output dir | `output/poisson/data_scaling/` (git-ignored) |
| Recorded | per-epoch val relative-L2, `test_rl2`, `stream`, `stopped_epoch` (results JSON) |

## Run

```bash
mkdir -p logs
sbatch experiments/poisson/data_scaling/sweep.sbatch      # array 0-8 (8 sizes + stream)
```

Re-run a single size: `sbatch --array=3 …` (task index into `NVALS`; the last task is the
stream). For seed replication later, add `--seed` variation and the plot aggregates
automatically (mean + min/max band).

## Plot

```bash
python experiments/poisson/data_scaling/plot_scaling.py --results_dir output/poisson/data_scaling/results
```

Pull artifacts from the cluster:
`rsync -avz euler:~/TensorPILS/output/poisson/data_scaling ~/Documents/TensorPILS/output/poisson/`
