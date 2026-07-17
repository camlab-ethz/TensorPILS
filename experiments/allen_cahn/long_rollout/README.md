# Experiment: long rollout (extrapolation past the training horizon)

The models in `compare_bptt/` are trained on a 10-step rollout. Here we roll each **best
checkpoint** forward for **20 steps** (argparse `--steps`) from the test-set initial conditions and
ask two questions, averaged over the test set:

- **Energy** — is the Ginzburg-Landau energy `E = ½a²uᵀAu + ε²·¼uᵀM(u²−1)²` well-behaved? Allen-Cahn
  is a gradient flow, so `E` should be **non-increasing**; a curve that climbs means the rollout is
  blowing up. The convex-concave reference energy is drawn as a dashed baseline.
- **Error** — relative FEM-`L²` of the predicted frame vs the **convex-concave (Eyre) reference**
  trajectory generated for the same ICs (the ground truth from the learned stepper's perspective).
  Steps 11–20 are **beyond the training horizon** — pure time-extrapolation.

## Two-stage (checkpoints never leave Euler)

The `.pth` checkpoints are large (~1–2 GB for the whole sweep), so:

1. **`compute_long_rollout.py`** runs on **Euler** (`run.sbatch`), where the checkpoints already are.
   It globs `output/allen_cahn/compare_bptt/checkpoints/*_best.pth`, reads each run's metadata from the matching
   `results/*.json`, regenerates the test ICs (deterministic) + a 20-step convex-concave reference,
   rolls each model out, and writes a tiny **`output/allen_cahn/long_rollout/metrics.json`** (numbers only).
2. **`plot_long_rollout.py`** runs **anywhere** from that JSON.

rsync only `metrics.json` + the figures — no checkpoints cross the wire.

It is **checkpoint-agnostic**: no epoch/eps is hardcoded, so it works on the current `compare_bptt`
run and again (unchanged) after a longer sweep overwrites the checkpoints at the same paths.

## Figures (per eps)

Two per eps — `long_rollout_eps{E}_{energy,error}.png` — each with **8 curves**: colour = loss
(min-movement / Galerkin-LS / data-driven), line style = BPTT mode (full_bptt `-` / detach_prev `--`
/ pushforward `:`). A **diverged** rollout stops where it went non-finite, marked with a red `x` and
flagged `(⊗step)` in the legend; energy is capped and error is log-scaled so a blow-up is visible.
A dotted vertical line marks the step-10 training horizon.

## Heatmaps (`heatmaps.py`, cluster-only)

The scalar plots above compress each frame to one number. To *see* where a rollout goes wrong,
`heatmaps.py` renders the 2D fields: one figure per `(eps, sample)`, rows = rollout timesteps,
columns = `[CC reference | run₁ | run₂ | …]`, so each model is directly comparable to the
convex-concave **ground truth** at each time. `RdBu_r`, symmetric colour scale shared per row; each
model panel is annotated with its relative L2 vs CC; diverged panels are drawn grey and flagged.

This needs the actual fields, so unlike the two-stage scalar path it reads the **checkpoints** and
runs **on Euler**, emitting PNGs directly (download the finished figures for a paper). It reuses the
IC / CC-reference / rollout code from `compute_long_rollout.py` (imported by path, so run it plainly):

```bash
# all eps, samples 0 1 2, all 8 runs as columns, 3 snapshots in [0,10] + 3 beyond:
python experiments/allen_cahn/long_rollout/heatmaps.py --steps 20

# a subset — one eps, chosen samples, chosen columns, explicit timepoints:
python experiments/allen_cahn/long_rollout/heatmaps.py --eps 16 --samples 0 3 \
    --runs mm:pushforward ls:full_bptt data:pushforward --snap_steps 0 5 10 15 20
```

`--runs` tokens match a `loss:mode` key (`mm:pushforward`), a whole loss (`data`), or any prefix
substring. `--snap_steps` overrides the default 3-in / 3-beyond rows; `--samples` picks test
indices; `--eps` restricts which eps to draw.

## Facts

| | |
|---|---|
| Source | best checkpoints from `output/allen_cahn/compare_bptt/` |
| Horizon | `--steps 20` (training was 10); `--n_samples 256` (all test) |
| Reference | convex-concave (Eyre) sparse float64, 20 steps from the same ICs |
| Assumptions | model rebuilt from the checkpoint's stored `model_config` (else the CLI defaults: modes 16², hidden 64, 5 layers); `--seed 42 --grid_resolution 64 --ic_r 0.5` match the sweep |
| Output dir | `output/allen_cahn/long_rollout/` (git-ignored) |

## Run

```bash
# Euler (compute):
mkdir -p logs
sbatch experiments/allen_cahn/long_rollout/run.sbatch          # -> output/allen_cahn/long_rollout/metrics.json

# Mac (pull the small metrics + plot locally):
rsync -avz euler:~/TensorPILS/output/allen_cahn/long_rollout ~/Documents/TensorPILS/output/
python experiments/allen_cahn/long_rollout/plot_long_rollout.py --metrics output/allen_cahn/long_rollout/metrics.json
```
