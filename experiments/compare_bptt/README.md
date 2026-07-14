# Experiment: compare BPTT / autodiff modes

How much does the **autodiff strategy for the autoregressive rollout** matter for FNO training on
Allen-Cahn? We compare three modes (`--bptt_mode`) for each loss, across reaction strength `eps`.
See `notes/ac_autoregressive/` §"Minimizing movement: intended pushforward vs. implemented hybrid".

## The three modes (two orthogonal detach knobs)

| mode | rollout-input detach | loss coupling detach | gradient |
|---|:--:|:--:|---|
| `full_bptt` | no | no | backprop through the whole rollout |
| `detach_prev` | no | yes | full BPTT, previous frame `u^k` frozen in the loss |
| `pushforward` | yes | yes | one-step (Brandstetter, Worrall & Welling 2022) |

- **rollout-input detach** (`RolloutTrainer._rollout`): feed each step a detached input ⇒ one-step
  gradient. Applies to every loss.
- **coupling detach** (in the loss): freeze the previous frame in `J`/`R`. Applies to the pairwise
  losses (MM proximal centre; Galerkin `R`'s first arg). The **data** loss has no coupling, so
  `detach_prev` ≡ `full_bptt` there — data has only two distinct modes.

The current defaults are: MM = `detach_prev`, Galerkin/data = `full_bptt` (so `--bptt_mode` with no
value reproduces them; the sweep sets the mode explicitly).

## Sweep

Three losses × their modes × `eps = 4, 8, 12, 16, 20, 24` = **48 runs** (config-major array):

| configs | loss | modes |
|---|---|---|
| 0-2 | minimizing-movement | full_bptt, detach_prev, pushforward |
| 3-5 | Galerkin (LS, convex-concave) | full_bptt, detach_prev, pushforward |
| 6-7 | data-driven | full_bptt, pushforward |

## Facts

| | |
|---|---|
| PDE / data | Allen-Cahn `a=1`, convex-concave sparse float64 reference, zero Dirichlet |
| Fixed resolution | grid `64²`, `dt=0.0025`, `n_steps=rollout_steps=10` (not scaled with `eps`) |
| Dataset | `K=4`, `n_train=1024`, `n_val=128`, `n_test=256`, `seed=42` |
| Optimizer / epochs | `adam`, cosine `1e-3->1e-4`, `500` epochs, batch `32` |
| Metric | best-model test space-time / final-time relative FEM-`L²` (+ MSE), vs `eps` |
| Output dir | `output/compare_bptt/` (git-ignored) |

Runs are tagged `fno_ac_..._bptt-{full,detach,push}_..._eps{E}_...` so all 48 coexist.

## Run (Euler)

```bash
mkdir -p logs
sbatch experiments/compare_bptt/sweep.sbatch          # full 48-task array (0-47), %8 concurrent
# or a subset, e.g. minimizing-movement only:
# sbatch --array=0-17 experiments/compare_bptt/sweep.sbatch
squeue --me
```

## Plot (per loss)

```bash
python experiments/compare_bptt/plot_compare_bptt.py \
    --results_dir output/compare_bptt/results         # --metric {st_rel_l2,final_rel_l2,mse}
```

Produces one figure per loss — `compare_bptt_{mm,galerkin,data}_{metric}.png` — each overlaying the
BPTT modes vs `eps` (dotted line at the `eps~16` resolution cutoff). Prediction (see the notes):
`pushforward ≳ detach_prev ≫ full_bptt` for MM, with the gap widening at stiff `eps`.
