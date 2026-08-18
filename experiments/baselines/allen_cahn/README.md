# Experiment: Allen–Cahn baselines — PINO and PI-DeepONet

**The question.** The same head-to-head as [`../poisson/`](../poisson/README.md), but for a
**nonlinear, time-dependent** problem — where the label-free argument is strongest, because here
the labels are genuinely expensive (a batched FEM Newton solve per trajectory, not a closed form).

## The task is held fixed across arms

All arms are the **autoregressive one-step stepper** the paper already uses: the operator maps
`uⁿ → uⁿ⁺¹`, seeded with one reference frame and rolled forward `--rollout_steps` times, with
`--bptt_mode pushforward`. This is deliberate. PI-DeepONet is more usually written with a
space-time trunk `(x, y, t)` mapping an initial condition to a whole trajectory — but that is a
*different task*, and a win or loss against it could not be attributed to the objective. Keeping
the task identical means the only things that vary are the architecture and how the Laplacian is
taken.

The residual is the strong form of the **same** convex–concave step the FEM arms use. Where
`ACProblem.residual` is

```
M (uⁿ⁺¹ − uⁿ)/τ  +  a² A uⁿ⁺¹  −  M ε² (uⁿ − (uⁿ⁺¹)³)
```

the baselines drop the mass matrix and substitute `A ↔ −Δ_h`:

```
(uⁿ⁺¹ − uⁿ)/τ  −  a² Δ_h uⁿ⁺¹  −  ε² (uⁿ − (uⁿ⁺¹)³)
```

with `Δ_h` the FD Laplacian (`pino`) or the autodiff one (`pideeponet`). Pinned by
`tests/test_baselines.py::test_pino_ac_residual_is_small_at_the_fem_reference`, which checks the
strong residual nearly vanishes on the FEM reference trajectory — a sign or integrator slip shows
up there as an `O(1)` residual.

## Arms

| arch | `--model` | `--loss` | labels | derivative |
|---|---|---|---|---|
| FNO | `fno` | `data` | yes | — |
| FNO | `fno` | `galerkin` | no | FEM (weak form) |
| FNO | `fno` | **`pino`** | no | FD |
| FNO | `fno` | `galerkin --ac_precond multigrid` | no | FEM + `P ≈ J₀⁻¹` |
| DeepONet | `deeponet` | `data` | yes | — |
| DeepONet | `deeponet` | **`pideeponet`** | no | autodiff |

## Facts

| | |
|---|---|
| Grid | `64²`, zero Dirichlet |
| Physics | `a=1`, `eps=32`, `dt=0.01`, `n_steps=20`, `rollout_steps=4` |
| Dataset | `K=4`, `n_train=512`, `n_val=64`, `n_test=128`, seeds 42/43/44. Reference = FEM convex–concave + Newton (float64) |
| Budget | 300 epochs, batch 32, `--bptt_mode pushforward` |
| BC | `pideeponet` requires the mollifier (hard BC), so the values fed back into the rollout and the values entering the residual coincide exactly; `pino` uses it too, following the reference |
| Metric | space-time relative FEM-`L2` over the rollout — the inherited `RolloutTrainer` evaluation, identical to every other AC table |
| Output | `output/baselines/allen_cahn/seed<N>/` (git-ignored) |

## Collocation points are the grid nodes — and here that is forced

Allen–Cahn has no analytical solution: the reference comes from a FEM Newton solve and therefore
exists **only at nodes**. Off-node collocation would require interpolating the reference, and the
experiment would then partly measure that interpolation. `--pi_n_colloc` still subsamples nodes if
the autodiff cost needs cutting; the rollout always evaluates every node, since the next frame has
to be fed back as a full grid.

## Run

```bash
mkdir -p logs
sbatch experiments/baselines/allen_cahn/sweep.sbatch
python experiments/baselines/allen_cahn/plot_ac_baselines.py \
    --results_dir output/baselines/allen_cahn
```

A single arm directly:

```bash
python -m tensorpils.cli --pde ac --loss pino --model fno \
    --grid_resolution 64 --n_train 512 --n_val 64 --n_test 128 -k 4 \
    --ac_a 1 --ac_eps 32 --dt 0.01 --n_steps 20 --rollout_steps 4 \
    --bptt_mode pushforward --epochs 300 --lr 1e-3 --seed 42 --device cuda \
    --output_dir output/baselines/allen_cahn/seed42
```

## Cost note

`pideeponet` evaluates the trunk per sample per collocation point and takes three backward passes
for the Laplacian, so a step costs several times an FNO step. Budget accordingly, or subsample with
`--pi_n_colloc`. Note this cuts the other way for the paper's cost argument: the expensive part of
the *supervised* arm is the Newton reference, which is paid once offline.
