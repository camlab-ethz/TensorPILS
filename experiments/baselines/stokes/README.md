# Experiment: Stokes baselines — PINO and PI-DeepONet on a saddle point

**The question.** The Poisson and Allen–Cahn tables compare our preconditioned weak-form loss
against the two published label-free operator methods. Stokes had no baseline arm, because
strong-form collocation has no canonical saddle-point formulation. This folder supplies the
port and the experiment: what a "PINO" and a "PI-DeepONet" *are* for `−μΔu + ∇p = f, ∇·u = 0`,
made as faithful to the Darcy recipe as a two-equation system allows, and scored exactly like
every other Stokes arm (velocity and pressure FE-L² separately, mean for selection).

## What the port is

| baseline | architecture | residual | derivatives | BC |
|---|---|---|---|---|
| **PINO** | FNO, `[B,2,H,W] → [B,3,H,W]` | strong-form momentum + continuity on the **velocity grid**, interior nodes | central differences (`FiniteDiff`) | velocity boundary ring zeroed (as `--pino_bc zero`, and as `_predict` projects every arm); pressure free |
| **PI-DeepONet** | DeepONet, 3 output channels | the same residual at interior collocation nodes | autodiff through the trunk | `--pideeponet_bc mollifier` (velocity × `sin(πx)sin(πy)`, pressure untouched) or `zero` (velocity boundary nodes zeroed + relative boundary penalty `--pi_lambda_bc`) |
| *(ours)* | FNO or DeepONet | weak form `½ rᵀPr` / `½‖Pr‖²` on the Taylor-Hood system | assembled `K` + block / monolithic `P` | projections |

Three decisions, each the Stokes form of something the Poisson port already fixed:

1. **Pressure everywhere, not strided.** The FEM arms read the FNO's pressure channel on the
   `[::2, ::2]` Q1 subgrid; a strong-form residual needs `∇p` at every interior velocity node,
   so PINO uses the full fine-grid channel (times `p_scale`). Evaluation is unchanged — the
   inherited `_predict` still strides, projects the gauge and scores against the Q1 reference.
2. **The reduction stacks the two equations** (`baselines.pino.stokes_reduce`): PINO's relative
   ratio becomes `‖(r_mom, w·r_div)‖ / ‖(f, 0)‖`, the strong-form analogue of our negative
   control `½‖Kc − b‖²` with `b = (M_u f, 0)`. The continuity weight `w = --pi_div_weight` is
   the one hyperparameter the saddle point adds: `div u ~ k·u` while `f ~ μk²u`, so the two
   rows have different units and no PINO example fixes their ratio. It is tuned on validation
   like a learning rate.
3. **No BC on the pressure.** Both mechanisms act on the velocity pair only
   (`DeepONetModel(bc_channels=(0, 1))`); the gauge is fixed by projection at evaluation and
   `∇p` never sees it. Consequently the velocity-only mollifier is exactly the "open control"
   the Poisson study asked for: on Poisson the mollifier was the entire PINO advantage, and
   here it cannot touch the field that carries the saddle-point difficulty.

Load-bearing checks in `tests/test_baselines.py`: the FD residual of a manufactured Stokes
solution converges at rate 2 (and pins the axis convention), the autodiff operators match the
analytic derivatives, and the FD and autodiff losses agree on a resolved field.

## Where the experiment lives

The Table 1 Stokes rows — our `pls` and `data` arms plus these two baselines, all at the Poisson
protocol (K=10, 65², 1024/128/256, 500 epochs, per-arm tuning on validation, three seeds) — are run
by [`experiments/stokes/stokes_paper/stokes_benchmark/`](../../stokes/stokes_paper/stokes_benchmark/README.md)
(sbatch pair, `run_local.sh`, `summarize.py`, results). This folder documents the *port*; that one
documents the *protocol* and carries the numbers.

PI-DeepONet memory: the autodiff residual retains a double-backward graph over every collocation
node for three fields; the benchmark subsamples with `--pi_n_colloc 1024` (fresh nodes each step,
the stochastic-collocation regime PI-DeepONet is normally trained in).
