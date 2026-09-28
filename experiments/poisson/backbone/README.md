# The ordering of the losses does not depend on the backbone

Reproduces the **backbone table** of the Stokes appendix. Stokes is the only experiment that
changes the neural operator (from the FNO to GAOT), so this checks, on the structured Poisson
problem where both apply, that the ordering of the three losses is a property of the objective and
not of the architecture.

## Setup

Poisson with `K = 4` on the 64 × 64 grid, 1024/128/256 samples, 500 epochs, batch 32, Adam with a
cosine schedule from `1e-3` to `1e-6`, seeds 42, 43 and 44. The losses are `L_data`, `L_PLS` (one
geometric multigrid V(2,2) cycle, Jacobi damping 2/3) and `L_LS`. GAOT uses its published
Poisson setting (latent grid 64 × 64, patch 2, transformer width 256 and depth 3); its learning rate
was selected on validation over `{1e-4, 3e-4, 1e-3, 3e-3}` with seed 42 and 150 epochs
(`lr_sweep.sh`), which picks `1e-3` for all three losses. The FNO runs use the same rate.

## Run

From the repository root, on a machine with a CUDA GPU:

```bash
bash experiments/poisson/backbone/lr_sweep.sh      # 12 GAOT runs, optional
bash experiments/poisson/backbone/run.sh           # 18 runs
python experiments/poisson/backbone/summarize.py
```

## Expected results

Test relative L² in %, mean ± half-range over three seeds:

| loss | FNO | GAOT |
|---|---|---|
| `L_data` | 0.59 ± 0.06 | 0.55 ± 0.05 |
| `L_PLS` | 0.66 ± 0.06 | 0.61 ± 0.07 |
| `L_LS` | 29.13 ± 2.22 | 37.57 ± 2.26 |
