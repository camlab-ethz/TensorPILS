# Experiment: realistic_ac at K=16 (harder initial conditions)

Identical to [`../realistic_ac/`](../realistic_ac/README.md) in every respect — `128²`, `eps=32`,
`a=1`, `dt=0.01`, a 10-step pushforward rollout, `n_train=1024`, and the same four loss arms
(data-driven / minimizing-movement / bare least-squares / preconditioned LS) — **except the
multi-frequency initial condition uses `K=16` modes instead of `K=4`**. The higher-frequency, more
complex ICs make this a substantially harder learning problem. See the `realistic_ac` README for the
full protocol, the loss definitions, and the numerical-soundness discussion.

Everything else is shared:
- **Output tree** is separate (`output/allen_cahn/realistic_ac_K16/`) so K=16 runs never mix with
  the K=4 experiment in the analysis, which keys runs by *loss*, not by K. (Run filenames also encode
  K — `..._K16_...` — so nothing collides regardless.)
- **Analysis scripts are reused unchanged** from `../realistic_ac/` (`compute_rollout.py`,
  `plot_rollout.py`, `heatmaps.py`, `replot_curves.py`); only the paths differ. `compute_rollout`
  reads K from each run's JSON, so it rebuilds the convex–concave reference at K=16 automatically.

## Run (Euler)

```bash
mkdir -p logs
sbatch experiments/allen_cahn/realistic_ac_K16/sweep.sbatch      # 4-task array (data / mm / ls / pls)
squeue --me
# once training is done:
sbatch experiments/allen_cahn/realistic_ac_K16/analyze.sbatch    # metrics + figures + heatmaps
```

## Plot / pull

```bash
rsync -avz --exclude 'checkpoints' \
    euler:~/TensorPILS/output/allen_cahn/realistic_ac_K16 ~/Documents/TensorPILS/output/allen_cahn/
# regenerate the per-run loss curves locally from the saved stats (no retraining):
python experiments/allen_cahn/realistic_ac/replot_curves.py \
    --results_dir output/allen_cahn/realistic_ac_K16/results
```

Produces the same figures as `realistic_ac` (`rollout_error.png`, `rollout_energy.png`,
`heatmaps/heatmap_test_sample{0,1,2}.png`, `curves/*_loss.png`), for the K=16 runs.
