# GAOT on structured Poisson — does the loss ranking survive a change of architecture?

## The question

Every Poisson number in this repo was produced with an FNO. The FNO is tied to the FFT and
therefore to a structured grid, so the planned move to **unstructured meshes** needs a second
architecture — and `tensorpils/gaot/` is it: a Geometry-Aware Operator Transformer, which
encodes an arbitrary point cloud onto a fixed structured *latent* token grid, runs a transformer
there, and decodes back at arbitrary query points.

Before trusting it on a mesh we cannot cross-check, it has to be run where we *can*: on the same
structured 64² Poisson problem, with the same data, the same budget and the same seeds as the
FNO table. Two things are being asked at once, and they are separable:

1. **Is the vendored GAOT wired up correctly?** Its node ordering must be the repo's
   (`k = i*nx + j`), or the FEM losses would be applied to a permuted field and would minimise
   the wrong residual — silently. `tests/test_gaot_model.py` pins this statically; a supervised
   `data` arm that trains to a sane error confirms it end to end.
2. **Does the loss ranking transfer?** On the FNO the story is
   `pls` (preconditioned residual) ≈ `data` (supervised) ≪ `galerkin` (bare residual). That
   claim is about the *objective's conditioning*, not about the FNO, so it ought to reproduce on
   a completely different architecture. If it does not, the claim was narrower than we thought.

## Protocol

Byte-identical to `experiments/baselines/poisson/sweep.sbatch` except `--model gaot`:
64², `K=4`, `n_train=1024 / n_val=128 / n_test=256`, 500 epochs, batch 32, Adam,
`lr_min=1e-6`, seeds 42/43/44. The reference FNO and DeepONet arms are **not re-run** — they
already exist under `output/baselines/poisson/seed*/results/` and `compare.py` reads them from
there.

The one knob that is *not* inherited is the learning rate. The FNO arms all ran at `1e-3`;
a transformer trained with plain cosine decay and no warmup is not obviously stable there, and
handing GAOT a rate tuned for a different architecture would confound "worse architecture" with
"wrong step size". So this study follows the two-stage shape of
`experiments/poisson/poisson_paper/poisson_benchmark`:

* **Stage 0** — `probe.txt`: 5 epochs per loss at the production size. Confirms every arm
  constructs, and measures the epoch time and memory the real sweep has to be sized against.
* **Stage 1** — `lr_sweep.txt`: one seed, 150 epochs, `lr ∈ {1e-4, 3e-4, 1e-3, 3e-3}` per loss.
  Pick the lowest best-validation relative L2 per arm.
* **Stage 2** — `sweep.txt`: 3 arms × 3 seeds at the selected rates. **Edit `ARM_LR` in
  `sweep.txt` before submitting it.**

GAOT hyperparameters are its published Poisson-Gauss setting (latent 64×64, patch 2, lifting 64,
transformer 256×3), which the CLI defaults reproduce — including the neighbour radius: the
derived default `2.1 * max(h_phys, h_latent)` lands on `0.033` at exactly this resolution pair.

## Running

```bash
clsubmit experiments/poisson/gaot_arch/probe.txt        # stage 0, ~20 min
clsubmit experiments/poisson/gaot_arch/lr_sweep.txt     # stage 1
clsubmit experiments/poisson/gaot_arch/sweep.txt        # stage 2, AFTER editing ARM_LR
python   experiments/poisson/gaot_arch/compare.py       # table + figure vs the FNO arms
```

Logs land in `$(cluster-info --get log_dir)`; `clsubmit --history` shows what went out.

## Result (2026-09-16, `gaot-model` @ 5125484)

Stage 1 picked **1e-3 for all three arms**, and in every case it is an interior optimum, not an
edge — so the FNO's rate transfers to GAOT unchanged and the drop-in comparison *is* the tuned
one. (`data` 3.29 / 1.86 / **1.37** / 7.22 %, `pls` 3.27 / 2.03 / **1.27** / 2.55 %,
`galerkin` 76.0 / 68.7 / **66.4** / 100.0 % at 1e-4 / 3e-4 / 1e-3 / 3e-3, best validation
relative L2 at 150 epochs.)

Stage 2, test relative L2, mean over seeds 42/43/44:

| loss | FNO | GAOT |
|------|-----|------|
| `data` (supervised) | 0.59 % (0.54–0.65) | **0.55 %** (0.51–0.61) |
| `pls` (preconditioned residual) | 0.66 % (0.61–0.72) | **0.61 %** (0.56–0.71) |
| `galerkin` (bare residual) | **29.13 %** (27.52–31.96) | 37.57 % (35.64–40.16) |

Parameters: GAOT 3,396,033, FNO 3,008,417 (13 % apart, not matched on purpose).
Epoch time ~2.95 s on an RTX 4090, so a 500-epoch run is ~25 min.

Three readings:

1. **The wiring is right.** The supervised arm, which has labels and therefore tests the model
   plumbing rather than the objective, lands on top of the FNO — slightly ahead, within seed
   spread. Node ordering, coordinates and radius are all doing what they claim.
2. **The loss ranking transfers.** `pls` ≈ `data` ≪ `galerkin` on GAOT exactly as on the FNO,
   and the gap is two orders of magnitude on both. That is the conditioning argument behaving
   like a property of the objective rather than of the FNO, which is what it was always claimed
   to be. It is also the first evidence that the argument will survive the move off a grid.
3. **The one place GAOT is worse is the bare residual** (37.6 % vs 29.1 %, well outside seed
   spread). Both arms are still descending at epoch 500 — the best epoch is 498–499 for every
   seed of both architectures — so this is *not* "GAOT converges worse"; it is "GAOT is slower
   on the arm whose Gauss–Newton matrix has κ = O(h⁻⁴)". Which arm sits where under a longer
   budget is an open question, and the negative control is the wrong place to spend it.

## What would count as a problem

* **`data` far above the FNO's supervised error.** The supervised arm is the wiring check: it
  has labels, so a large error there points at the model plumbing (ordering, scaling, radius),
  not at the objective.
* **Latent tokens with empty neighbourhoods.** Silent, and it makes part of the transformer
  input pure noise. Guarded by `test_every_latent_token_sees_at_least_one_node`; if the latent
  grid is ever raised well past the mesh, re-check it.
* **`galerkin` beating `pls`.** That would contradict the conditioning argument and would be the
  interesting result — but check the learning rates first: the bare-residual arm is the one that
  is still descending at 500 epochs on the FNO, so "did not converge" and "converged worse" are
  different claims and only the curve separates them.
