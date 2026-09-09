# Allen–Cahn benchmark — paper-ready version

**Regime is fixed: `128^2`, `eps=32`.** We do not go to `256^2/eps=64`, despite `realistic_ac`'s
header suggesting it for production. The justification stands on its own: halving both grid and
`eps` holds the interface resolution `l/h = a(n-1)/eps ~ 4` fixed, so this is the same physical
regime at a quarter the cost.

**Status: `lr_sweep.sbatch` is ready to submit. `sweep.sbatch` (finals) is still a scaffold** — it
needs the rates this sweep produces.

Five arms, mirroring `experiments/poisson/poisson_paper/poisson_benchmark/`:

| # | arm | flags | labels? |
|---|---|---|---|
| 0 | $L_\text{data}$ | `--loss data` | supervised |
| 1 | $L_\text{LS}$ | `--loss galerkin --ac_loss_form galerkin --ac_loss_integrator convex_concave` | label-free |
| 2 | $L_\text{PLS}$ | ... `--ac_precond multigrid` | label-free |
| 3 | PINO | `--loss pino` | label-free |
| 4 | PI-DeepONet | `--model deeponet --loss pideeponet` | label-free |

## Where this comes from

Two existing pieces have to be reconciled, and they disagree on the regime:

**`experiments/allen_cahn/realistic_ac`** — the preliminary study, and the source of the physical
setup. Ran 4 arms (data, minimizing movement, bare LS, preconditioned LS) at `128^2`, `eps=32`,
`K=4`, `a=1`, `dt=0.01`, `n_steps=10`, `rollout_steps=10`, `n_train=1024`, batch 16, 500 epochs,
seed 42, `--bptt_mode pushforward`, `convex_concave`. Results (test space-time relative L2):

| arm | error |
|---|---|
| minimizing movement | 0.0103 |
| $L_\text{data}$ | 0.0199 |
| $L_\text{PLS}$ | 0.0483 |
| $L_\text{LS}$ | **0.9997** — collapses |

**`experiments/baselines/allen_cahn`** — has the PINO and PI-DeepONet wiring (6 arms), but at
`64^2`, `n_steps=20`, `rollout_steps=4`, 300 epochs. Never run. Its arm list is the useful part;
its regime is not the one we want.

So this folder = `realistic_ac`'s regime + the baselines' arm list, on the two-stage
sweep-then-finals protocol the Poisson benchmark uses.

## Decisions still open

1. **Does minimizing movement stay?** It is the *best* arm in the preliminary study, beating
   supervised. It is the AC analogue of Deep Ritz, which the Poisson table reports separately.
   Five arms as listed, or six?
3. **Learning rates.** The Poisson benchmark showed the per-arm optimum moves with the epoch
   budget and that a shared rate handicaps some arms. The baselines sbatch currently pins every
   arm at `1e-3`, which is a placeholder, not a result. A stage-1 sweep at the full budget is
   needed, and the PI-DeepONet grid may need extending downward as it did for Poisson.
4. **Seeds.** `realistic_ac` ran one. The bare-LS arm has a dedicated 3-seed diagnostic precisely
   because its failure mode (NaN vs collapse) is seed-dependent, which argues for >= 3 seeds here.
4. **PINO's reduction.** For Poisson the reference `rel` reduction normalises by `||f||`. The AC
   step residual has no right-hand side to normalise against, so `PINOACLoss` uses `mse` — see
   `notes/pino_allen_cahn/pino_allen_cahn.tex`. That is a real asymmetry between the two PDEs'
   PINO arms and should be stated in the paper, not silently absorbed.
5. **Boundary conditions.** Poisson's PINO now zeroes boundary nodes rather than using the
   `sin(pi x)sin(pi y)` mollifier. The AC arms must do the same, and PI-DeepONet still carries the
   mollifier — the same inconsistency the Poisson table has.

## Protocol (planned, mirrors poisson_benchmark)

* **Stage 1** `lr_sweep.sbatch` (**ready**): 5 arms x 6 rates x 500 epochs = 30 tasks, single seed,
  selected on lowest validation space-time relative L2. `--no_checkpoint` throughout. Extend the
  grid if a winner lands on an edge — appending, never reordering, so existing task indices hold.

  ```bash
  sbatch --array=0-29%15 experiments/allen_cahn/ac_paper/lr_sweep.sbatch
  ```
* **Stage 2** `sweep.sbatch`: 5 arms x 3 seeds at the selected rates.
* Plotting mirrors `plot_lr_sweep.py` / `plot_final.py`, keyed on the AC metric
  (`stats.val_st_rel_l2` rather than `val_rel_l2_errors`).


## The sixth arm: boundary-penalised PI-DeepONet

**Deferred — not part of the current sweep.** Recorded here so the reasoning is not lost. The mollified PI-DeepONet is arm 4; the penalised variant
(`lambda_bc = 100`, soft BC instead of the `sin(pi x)sin(pi y)` mollifier) needs three changes:

1. `PIDeepONetACLoss` has no `lambda_bc` argument — the *Poisson* version
   (`PIDeepONetPoissonLoss`) does, but the AC one was never given one.
2. `PIDeepONetACTrainer` raises on `mollify=False`, and the reason is substantive, not a
   formality: *"so the rollout values and the residual values coincide on the boundary."* In an
   autoregressive rollout the model's own output is fed back as the next input, so without a hard
   BC the boundary values are whatever the network emits and they propagate through all 10 steps.
   A soft penalty makes them small, not zero.
3. `cli.py` rejects `--mollify off` together with `--loss pideeponet`.

(1) and (3) are mechanical. **(2) is a design decision:** with a soft BC, what does the rollout
feed back on the boundary — the raw network output (honest, but the penalty is then the only thing
keeping the trajectory from drifting), or a projected-to-zero copy (stable, but then the BC is hard
at rollout time and soft only in the loss, which is arguably the worst of both)? Poisson never had
to answer this because it is a one-shot problem.

Once resolved, append the arm and submit `--array=30-35`; tasks 0–29 keep their meaning.
