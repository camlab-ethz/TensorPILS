# Experiment: out-of-distribution generalization vs. preconditioner strength

Does the loss *geometry* set by the convex-blend strength `t` (residual-driven at `t→0`,
supervised-MSE-equivalent at `t=1`) affect how a neural operator **generalizes to sources it
was not trained on**? Each run trains on one source-complexity `K`, then we track relative-L2
each epoch on the in-distribution val set *and* on higher-`K` (out-of-distribution) sources
on the same `64²` grid / operator.

The question is shared across regimes; only the **training distribution** (and, later, the
**loss/method**) change. Each such combination is a **scenario** in its own subfolder, sharing
the one plot script here (`plot_generalization.py`, which auto-discovers the eval-`K`s from
each run's results). Scenario folders are named `<method>_k<train-K>`.

## Scenarios

| Scenario | Loss | Train K | OOD K | FNO modes | Notes |
|---|---|---|---|---|---|
| [`pls_k4/`](pls_k4/README.md) | PLS (blend) | 4 | 6, 8 | 16 | OOD within the representable band |
| [`pls_k16/`](pls_k16/README.md) | PLS (blend) | 16 | 20 | 16 | Shizheng's regime; OOD **exceeds** the band (deliberate truncation floor) |
| [`data_k16/`](data_k16/README.md) | data-driven (MSE, H¹₀) | 16 | 20 | 16 | supervised losses; does the H¹₀ metric help OOD extrapolation vs MSE? |

Planned: Deep Ritz variants (`deepritz_k4/`, `deepritz_k16/`, …) — same layout, different loss.

## Conventions

- Each scenario writes to `output/generalization/<scenario>/` (git-ignored).
- Plot a scenario with the shared script, pointing at its results:
  `python experiments/generalization/plot_generalization.py --results_dir output/generalization/<scenario>/results`.
- Pull a scenario's artifacts from the cluster:
  `rsync -avz euler:~/TensorPILS/output/generalization/<scenario> ~/Documents/TensorPILS/output/generalization/`.

See each scenario's `README.md` for its facts table and exact run commands.
