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
   It globs `output/compare_bptt/checkpoints/*_best.pth`, reads each run's metadata from the matching
   `results/*.json`, regenerates the test ICs (deterministic) + a 20-step convex-concave reference,
   rolls each model out, and writes a tiny **`output/long_rollout/metrics.json`** (numbers only).
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

## Facts

| | |
|---|---|
| Source | best checkpoints from `output/compare_bptt/` |
| Horizon | `--steps 20` (training was 10); `--n_samples 256` (all test) |
| Reference | convex-concave (Eyre) sparse float64, 20 steps from the same ICs |
| Assumptions | model rebuilt from the checkpoint's stored `model_config` (else the CLI defaults: modes 16², hidden 64, 5 layers); `--seed 42 --grid_resolution 64 --ic_r 0.5` match the sweep |
| Output dir | `output/long_rollout/` (git-ignored) |

## Run

```bash
# Euler (compute):
mkdir -p logs
sbatch experiments/long_rollout/run.sbatch          # -> output/long_rollout/metrics.json

# Mac (pull the small metrics + plot locally):
rsync -avz euler:~/TensorPILS/output/long_rollout ~/Documents/TensorPILS/output/
python experiments/long_rollout/plot_long_rollout.py --metrics output/long_rollout/metrics.json
```
