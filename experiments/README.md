# Experiments

One directory per experiment of the paper. Each holds a `README.md` (setup, commands, expected
numbers, compute), the launchers (`run.sh` and, where the paper selected hyperparameters on
validation, `lr_sweep.sh` / `grid_search.sh`) and the scripts that turn the results into the
paper's tables and figures. Run everything from the repository root.

| Paper | Directory | Launchers | GPU-hours* |
|---|---|---|---|
| Figure 1 (Adam on a linear model) | [`graphical_abstract/`](graphical_abstract/README.md) | `run.sh` | CPU, minutes |
| Table 1, Poisson rows; Poisson hyperparameters | [`poisson/benchmark/`](poisson/benchmark/README.md) | `lr_sweep.sh`, `run.sh` | 14 + 4 |
| Figure 2 (conditioning sweep `P_t`) | [`poisson/blend_sweep/`](poisson/blend_sweep/README.md) | `run.sh` | 2 |
| Figure 3 (infinite-data limit) | [`poisson/infinite_data/`](poisson/infinite_data/README.md) | `run.sh` | 6 |
| Poisson dataset samples (appendix) | `poisson/dataset_samples/` | `plot_dataset_samples.py` | CPU, seconds |
| Table 1, Allen–Cahn rows; Figure 4 | [`allen_cahn/benchmark/`](allen_cahn/benchmark/README.md) | `lr_sweep.sh`, `run.sh` | 140 + 75 |
| Table 1, Stokes rows; Figure 5; Stokes samples, hyperparameters and preconditioner ablation (appendix) | [`stokes/benchmark/`](stokes/benchmark/README.md) | `grid_search.sh`, `run.sh` | 5 + 7 |
| Backbone table (appendix) | [`poisson/backbone/`](poisson/backbone/README.md) | `lr_sweep.sh`, `run.sh` | 2 + 6 |

\* Approximate, on an RTX 4090. The launchers that select hyperparameters are only needed to
reproduce the selection; the final runs use the selected values directly.

Not yet included: the scripts of the resolution figure (appendix on interpolated neural operators,
evaluating the checkpoints of `poisson/benchmark/`) and of the Allen–Cahn roll-out snapshots.

## Launchers

Every launcher is a list of training commands (`python -m tensorpils.cli ...`), one per task, and
shares the convention of [`common.sh`](common.sh):

```bash
bash experiments/poisson/benchmark/run.sh          # every task, in order
bash experiments/poisson/benchmark/run.sh list     # the numbered tasks
bash experiments/poisson/benchmark/run.sh 3        # task 3 only
```

On a SLURM cluster, an array job runs one task per GPU:

```bash
sbatch --array=0-14 --gpus=1 --time=01:00:00 --wrap "bash experiments/poisson/benchmark/run.sh"
```

`PY` selects the Python interpreter and `OUT` the output directory of a launcher.

## Outputs

Each run writes under its `--output_dir`: `results/<run>.json` (configuration, per-epoch
statistics and test errors; what every table and figure script reads), `checkpoints/<run>_best.pth`
(the best-validation model, unless `--no_checkpoint`), and diagnostic plots in `curves/`,
`visualization/` and `error/`. The run name encodes the configuration, so methods can share a
directory. Generated meshes are cached in `output/meshes/`.
