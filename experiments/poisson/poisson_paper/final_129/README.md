# Poisson paper table at 129x129

A resolution check on the `65^2` finals: **one seed, seven arms, seven tasks**. The question is
whether the ordering of the arms survives a 4x finer grid, not what the run-to-run spread is —
the `65^2` finals already carry the error bars.

```bash
# on Euler
cd ~/TensorPILS && git pull && mkdir -p logs
sbatch --array=0 experiments/poisson/poisson_paper/final_129/sweep_129.sbatch   # smoke test
sbatch experiments/poisson/poisson_paper/final_129/sweep_129.sbatch             # all 7

# locally, when it finishes
rsync -avz --info=progress2 --exclude='checkpoints/' --exclude='visualization/' \
  mzeinhofer@euler.ethz.ch:TensorPILS/output/poisson/poisson_paper/final_129/ \
  output/poisson/poisson_paper/final_129/
.venv/bin/python experiments/poisson/poisson_paper/plot_final.py \
  --root output/poisson/poisson_paper/final_129
```

`plot_final.py` needs no changes. With one seed the min–max band collapses to zero width and the
script prints `WARNING: fewer than 3 seeds` — expected here, and worth keeping visible so a
single-seed number never gets read as if it had error bars.

## Memory

Measured on this repo, batch 32, `hidden_dim 64`, `num_layers 5`, `n_modes (16,16)`:

| what | 65^2 | 129^2 | note |
|---|---|---|---|
| FNO fwd+bwd activations | 1.4 GB | 2.9 GB | ~2x, not 4x — the spectral part is fixed by `n_modes` |
| multigrid buffers (on GPU) | 1.5 MB | 5.9 MB | negligible |
| multigrid build (**host** RAM) | 0.7 GB | 5.2 GB | see below |

**GPU: yes, ask for more, but the jump is smaller than the grid.** Activation memory roughly
doubles rather than quadruples, because `n_modes` is held at 16 so the spectral convolutions do
not grow — only the pointwise/spatial tensors do. `--gres=gpumem:20g` is set in the sbatch, which
is generous for the FNO arms; the arm to watch is **PI-DeepONet**, whose autodiff Laplacian builds
a double-backward graph over `B x Q = 32 x 16641` collocation points, 4x the `65^2` count. If any
task OOMs it will be that one, and the fix is `--batch_size 16` for that arm alone (noting the
change in the table).

**Host RAM is the surprise.** `GeometricMultigrid` assembles a *dense* `[N, N]` float64 operator at
every level before sparsifying it (`preconditioners/multigrid.py:111`). At level 0 that is
`16641^2 x 8 B = 2.2 GB`, plus a float32 copy — measured peak 5.2 GB versus 0.7 GB at `65^2`.
It is transient, one-time, and CPU-side, so `--mem-per-cpu=20G` covers it, but it is why this
sbatch asks for more than `sweep.sbatch` did. It also means the two preconditioned arms take a
minute or so longer to start.

## Two caveats worth knowing before reading the results

**Learning rates were tuned at `65^2` and are reused here.** This is the deliberate choice — holding
the model and the optimizer fixed is what makes this a resolution study — but it is not free. Our
losses reduce by summing over nodes, so a 4x node count changes the gradient scale, and the
per-arm optimum may have moved. If an arm looks worse at 129 the first thing to rule out is a
mistuned learning rate, not the method. A cheap check is the two preconditioned arms plus `data`
at `{lr/3, lr, 3lr}` — 9 tasks — before concluding anything about ordering.

**`--mg_levels 5`, not the default 4.** The `65^2` runs coarsened `65 -> 33 -> 17 -> 9`. Four levels
at 129 would stop at `17`, giving a shallower hierarchy and a different preconditioner; five levels
reproduce the same coarsest `9x9` grid, so `P` is the genuine analogue. Everything else — modes,
width, depth, batch, epochs, K, damping `8/9` — is unchanged from `sweep.sbatch`.

## Walltime

`--time=12:00:00` is a guess with headroom, not a measurement. Set it from what the `65^2` finals
actually cost:

```bash
sacct -j <final_jobid> --format=JobID,State,Elapsed | sort -k3 | tail -5
```

Expect roughly 2-3x that per task at 129 (activations double, the FEM operator applications
quadruple). If the slowest `65^2` task ran over 4 h, raise the limit before submitting.
