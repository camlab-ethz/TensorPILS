# Experiments

Each subfolder is one self-contained experiment: a `README.md` (description, facts, run &
plot commands), the SLURM `*.sbatch` sweep script, and the aggregation plot script. Run
artifacts land in `output/<name>/` (git-ignored); pull them back from the cluster with
`rsync -avz euler:~/TensorPILS/output/<name> ~/Documents/TensorPILS/output/`.

| Experiment | Question |
|---|---|
| [`sweep_blend/`](sweep_blend/README.md) | Does the blend preconditioner strength `t` (residual `t=0` → supervised `t=1`) control training conditioning? Overlay + collapse plots vs `κ(H)`. |
| [`generalization/`](generalization/README.md) | Does that loss geometry `t` affect out-of-distribution generalization to higher-frequency `K=6,8` sources (trained on `K=4`)? |

Theory write-up for the spectral preconditioners: `../preconditioner_notes/`.
