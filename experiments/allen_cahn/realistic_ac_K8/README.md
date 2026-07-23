# Experiment: realistic_ac at K=8 (moderately harder ICs)

Identical to [`../realistic_ac/`](../realistic_ac/README.md) in every respect — `128²`, `eps=32`,
`a=1`, `dt=0.01`, a 10-step pushforward rollout, `n_train=1024`, the same four loss arms (data-driven
/ minimizing-movement / bare least-squares / preconditioned LS) — **except the multi-frequency
initial condition uses `K=8` modes instead of K=4**. This sits between the working K=4 case and the
too-hard K=16 case (where none of the runs converged), so it is the interesting intermediate
difficulty. See the `realistic_ac` README for the full protocol and loss definitions.

Everything else is shared:
- **Output tree** is separate (`output/allen_cahn/realistic_ac_K8/`) so K=8 runs never mix with the
  other K experiments in the analysis, which keys runs by *loss*, not by K. (Run filenames also
  encode K — `..._K8_...` — so nothing collides regardless.)
- **Analysis scripts are reused unchanged** from `../realistic_ac/` (`compute_rollout.py`,
  `plot_rollout.py`, `heatmaps.py`, `replot_curves.py`); only the paths differ. `compute_rollout`
  reads K from each run's JSON, so it rebuilds the convex–concave reference at K=8 automatically.

## Run (Euler)

```bash
mkdir -p logs
sbatch experiments/allen_cahn/realistic_ac_K8/sweep.sbatch      # 4-task array (data / mm / ls / pls)
squeue --me
# once training is done:
sbatch experiments/allen_cahn/realistic_ac_K8/analyze.sbatch    # metrics + figures + heatmaps
```

## Plot / pull

```bash
rsync -avz --exclude 'checkpoints' \
    euler:~/TensorPILS/output/allen_cahn/realistic_ac_K8 ~/Documents/TensorPILS/output/allen_cahn/
# regenerate the per-run loss curves locally from the saved stats (no retraining):
python experiments/allen_cahn/realistic_ac/replot_curves.py \
    --results_dir output/allen_cahn/realistic_ac_K8/results
```

Produces the same figures as `realistic_ac` (`rollout_error.png`, `rollout_energy.png`,
`heatmaps/heatmap_test_sample{0,1,2}.png`, `curves/*_loss.png`), for the K=8 runs.
