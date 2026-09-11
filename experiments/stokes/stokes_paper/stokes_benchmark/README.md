# Stokes: the Table 1 rows

The Stokes column of the paper's summary table: four arms, one dataset, one budget, the **same
protocol as the Poisson rows** (`../../../poisson/poisson_paper/poisson_benchmark/`). It is also
the first experiment that runs the two published baselines on a saddle point; what "PINO" and
"PI-DeepONet" mean there is defined in [`experiments/baselines/stokes/`](../../../baselines/stokes/README.md).

| arm | flags | labels | derivative | BC |
|---|---|---|---|---|
| `pls` (ours) | `--loss pls --stokes_precond block --schur_omega ω` | no | FEM, `½ rᵀPr` with the block preconditioner as the norm weight | projections |
| `data` | `--loss data` | yes | — | projections |
| `pino` | `--loss pino --pi_div_weight w` | no | strong form, central differences on the velocity grid | velocity boundary ring zeroed (as our arms) |
| `pideeponet` | `--model deeponet --loss pideeponet --pideeponet_bc zero --pi_lambda_bc λ --pi_div_weight w --pi_n_colloc 1024` | no | strong form, autodiff through the trunk | velocity boundary nodes zeroed + relative boundary penalty `λ` |

Both baselines get the **zero-BC** treatment the Poisson rows use (no `sin(πx)sin(πy)` mollifier),
and both share the continuity weight `w` of the stacked residual `‖(r_mom, w·r_div)‖ / ‖(f, 0)‖`
— the one hyperparameter a saddle point adds, tuned on validation like a learning rate. The bare
least-squares arm is not in the table (as for Poisson); it is the negative control of
`experiments/stokes/h_refinement`.

## Facts

| | |
|---|---|
| Dataset | body force `f_c = Σ_{k,l≤K} a_{ckl}(k²+l²)^{-1/2} sin(kπx) sin(lπy)`, `a ~ U[-1,1]`, **`K=10`** (as Poisson), `μ=1`; reference = discrete Taylor-Hood solve (float64, batched); each sample normalised to unit velocity FE-L²; network-facing `f_scale`, `p_scale` |
| Spaces | Taylor-Hood Q2/Q1 on the structured quad9 mesh: velocity `65²` × 2, pressure `33²` (the `[::2, ::2]` subgrid); 9539 DOFs |
| Splits | `n_train/val/test = 1024/128/256`; the seed draws the dataset **and** the initialisation (as Poisson) |
| Budget | 500 epochs, batch 32, Adam, cosine `lr → lr/10` |
| FNO | modes `16×16`, hidden 64, 5 layers, 2 → 3 channels (3,008,803 parameters) |
| DeepONet | `p=128`, width 256, depth 4, 64 random Fourier trunk features (scale 2), 3 output channels |
| Preconditioner | block `P = diag(GMG/μ, ω_S μ diag(M_p)⁻¹)`, `mg_levels 4` (65→33→17→9), smoothing 2/2, **`mg_omega 2/3`** (every Stokes measurement in the repo uses it; the Poisson rows use 8/9), `schur_omega 0.5`, weighted form |
| Selection | best **validation** `0.5·(rel-L²_u + rel-L²_p)` (`stats.val_rel_l2_errors`, the trainer's own model-selection metric); seed 42, full budget |
| Reported | test relative FE-L² of velocity and pressure **separately**, at the best-validation checkpoint, mean ± half-range over seeds 42/43/44 |
| Output | `output/stokes/stokes_paper/stokes_benchmark/{lr_sweep/<arm>/<cell>/, final/<arm>/<cell>/seed<S>/}`, `<cell> = lr<lr>[_dw<w>][_lam<λ>]` (run prefixes encode neither lr nor seed; `-dw` is omitted from the prefix at `w=1`) |
| Cost | RTX 4090, one process alone, real setting: `data` 1.36 s/epoch (1.9 GB), `pls` 1.63 (2.0 GB), `pino` 1.05 (1.9 GB), `pideeponet` at 1024 collocation nodes 1.63 (4.2 GB; all 3969 interior nodes: 6.5 s/epoch, 14.7 GB) — i.e. **9–14 min per 500-epoch cell**; the dataset build (batched float64 Taylor-Hood solve, 1408 samples) takes seconds. `PAR=4` fits comfortably in 24 GB |
| Scales (K=10) | `f_scale = 193.3`, `p_scale = 46.4`, `u_l2_scale = 1`, `p_l2_scale = 45.6` (the physics ties `‖f‖ ~ 10²`, `‖p‖ ~ 50` to unit velocity; see CLAUDE.md "Stokes field scaling") |

## Protocol

**Stage 1** (`lr_sweep.sbatch` / `STAGE=1`): seed 42, full 500-epoch budget, the grid in
`common.sh`:

| arm | grid | why |
|---|---|---|
| `data`, `pls` | `lr ∈ {1e-4, 3e-4, 1e-3, 3e-3}`; `pls` additionally `ω ∈ {0.5, 4, 16}` at the two best `lr` | Poisson selected 3e-4 for both FNO arms; the existing K=4 Stokes runs used 1e-3 / 2e-3. `ω` is the Schur weight of the block preconditioner, the counterpart of the baselines' continuity weight (the conditioning study puts its optimum near 16; the draft used 0.5) |
| `pino` | `lr ∈ {3e-4, 1e-3, 3e-3, 1e-2} × w ∈ {0.3, 1, 3, 10}` | Poisson PINO selected 3e-3, so the grid reaches 1e-2 to keep it interior; `div u ~ k·u` against `f ~ μk²u` puts equal units at `w ~ μk ≈ 3–30` for K=10; joint because `w` changes the loss scale |
| `pideeponet` | `lr ∈ {3e-5, 1e-4, 3e-4} × λ ∈ {0.01, 0.03, 0.1}` at `w=1`, then **stage 1b**: `w ∈ {3, 10, 30, 100}` at the selected `(lr, λ)` | Poisson zero-BC selected `(1e-4, 0.03)` with a sharp U in `λ`; `w` is checked in a second pass at the selected `(lr, λ)` rather than jointly |

**Edge rule.** If a selected value is the smallest or largest tried for that arm (`summarize.py`
prints `EDGE:`), append the next half-decade point in that direction to `STAGE1_CELLS` in
`common.sh` (other hyperparameters at their selected values), re-run stage 1 (finished cells are
skipped) and re-select, until the optimum is interior or the extension is worse.

**Stage 2** (`sweep.sbatch` / `STAGE=2`): seeds 43, 44 at each arm's selected cell; seed 42 is the
stage-1 run. Output is config-named (`final/<arm>/<cell>/`) so a re-selection never mixes cells.

## Run

From a **login** node (the route the Poisson table used):

```bash
cd ~/TensorPILS && mkdir -p logs
sbatch experiments/stokes/stokes_paper/stokes_benchmark/lr_sweep.sbatch      # 58 cells, seed 42
python experiments/stokes/stokes_paper/stokes_benchmark/summarize.py          # ranking, selection, EDGE
# append edge extensions / the stage-1b cells to common.sh, submit them with --array=33-…,
# paste the printed ARMS_LR / ARMS_W / ARMS_LAM into sweep.sbatch, then
sbatch experiments/stokes/stokes_paper/stokes_benchmark/sweep.sbatch         # 4 arms x 3 seeds
python experiments/stokes/stokes_paper/stokes_benchmark/summarize.py          # the Table 1 rows
```

On a **compute** node (inside an allocation), the same protocol end to end, restartable:

```bash
PAR=4 bash experiments/stokes/stokes_paper/stokes_benchmark/run_local.sh          # STAGE=all
STAGE=1 PAR=4 bash .../run_local.sh      # stage 1 only (inspect EDGE lines before going on)
DRY_RUN=1 bash .../run_local.sh          # list what would run
```

One cell directly:

```bash
python -m tensorpils.cli --pde stokes --loss pls --stokes_precond block --schur_omega 0.5 \
    -k 10 --grid_resolution 65 --n_train 1024 --n_val 128 --n_test 256 \
    --epochs 500 --batch_size 32 --optimizer adam --lr 3e-4 --lr_min 3e-5 \
    --seed 42 --device cuda --output_dir output/stokes/stokes_paper/stokes_benchmark/lr_sweep/pls/lr3e-4
```

## Results

Run 2026-09-10/11 on Euler RTX 4090s (this compute node, `run_local.sh`, plus `extras.sbatch` for
the ω tuning, the monolithic arms and the 129² refinement). `summarize.py` reproduces every stage-1
and stage-2 number below; the diagnostics live in `diagnostics/`.

**Table 1 (three seeds, test relative FE-L², mean ± half-range).** Every selected value is interior
to its grid after the edge-rule extensions.

| arm | selected (validation) | velocity | pressure | s/epoch (alone) |
|---|---|---|---|---|
| **`pls`** (block P) | `lr 1e-3`, `ω 256` | **2.38 ± 0.31 %** | **1.69 ± 0.27 %** | 1.6 |
| `data` | `lr 3e-4` | 2.85 ± 0.25 % | 2.81 ± 0.25 % | 1.4 |
| **`pino`** | `lr 3e-3`, `w 300` | **2.09 ± 0.26 %** | **1.50 ± 0.24 %** | 1.1 |
| `pideeponet` | `lr 1e-4`, `λ 0.1`, `w 100` | 33.0 ± 1.2 % | 34.2 ± 1.5 % | 1.6 |

Other three-seed cells in `final/`: `pls` at `ω 0.5` (the draft's default) 3.64 ± 0.44 % / 2.60 ± 0.30 %,
at `ω 64` 2.47 ± 0.37 % / 1.78 ± 0.22 %.

**The weight of the continuity block is the decisive knob, for both formulations.** Validation
mean(u, p) on seed 42:

| PINO `w` @ `lr 3e-3` | 0.3 | 1 | 3 | 10 | 30 | 100 | **300** | 1000 |
|---|---|---|---|---|---|---|---|---|
| | 67 % | 30 % | 6.9 % | 2.4 % | 1.70 % | 1.65 % | **1.55 %** | 2.0 % |

| PLS `ω` @ `lr 1e-3` | 0.5 | 4 | 16 | 64 | **256** | 1024 |
|---|---|---|---|---|---|---|
| | 3.14 % | 2.37 % | 2.14 % | 2.05 % | **1.94 %** | 2.09 % |

Mechanism (measured with a 100-epoch probe at `w = 0.3`): the relative reduction normalises by
`‖f‖ ≈ 193` while the divergence of a unit-velocity field is `O(10)`, so at small `w` an
irrotational velocity error `∇φ` is absorbed by the pressure (`p → p + μΔφ`) at almost no cost —
the probe reached residual 0.09 with 223 % velocity error, smooth and irrotational (Nyquist-band
energy fraction 0.000; 3×3 smoothing left the error unchanged). `w` is the block-diagonal norm
weight `B = diag(I, w²I)` of the paper's Section 2; `ω` plays the same role inside the block `P`.
PI-DeepONet: `λ_bc` 0.01/0.03/0.1/0.3/1 → 61/59/49/66/64 %, then `w` 1…1000 → 49/48/36/32/32/36/67 %;
its best epoch is 100–230 and the error rises afterwards.

**Controls (seed 42).**

| | velocity | pressure |
|---|---|---|
| bare FEM least squares (`galerkin`), `lr 3e-3` / `1e-3` | 21.6 % / 40.9 % | 8.6 % / 13.2 % |
| monolithic P (`applied_fe`), `lr 3e-4` / `1e-3` — 13.3 s/epoch | 2.80 % / 2.88 % | 2.61 % / 2.62 % |
| 129²: PINO `w 300` | 1.81 % | 1.16 % |
| 129²: PLS `ω 16` | 2.35 % | 1.62 % |
| 129²: supervised | 2.82 % | 2.74 % |
| 129²: PLS `ω 0.5` | 3.50 % | 2.27 % |

**Two things that went wrong on the way, both fixed and pinned.** The first PINO/PI-DeepONet cells
trained the strong-form residual through `rel_lp(residual, f)`, which subtracts `f` twice and makes
the minimiser the solution for `2f` (velocity error → exactly 100 %); every affected cell was rerun
after the fix (`tests/test_baselines.py::test_stokes_rel_reduction_is_the_ratio_of_residual_to_force`).
And the driver was edited while running once (bash reads scripts incrementally), which is why the
log shows several "passes": each pass only ran the cells that had no results JSON yet.


## Conventions

- Numbers here are **not comparable** to `experiments/stokes/h_refinement` (K=4, 512 samples,
  300 epochs, `lr 2e-3 → 1e-4`) or to `tab:stokes_head_to_head` of the progress draft.
- K=10 on the 33² pressure grid is about 3.3 nodes per wavelength for the highest mode: the
  pressure target is rough, and the reference is the *discrete* solution, so the residual arms
  have no discretisation floor while the supervised arm fits a rougher target than on Poisson.
- `stats.epoch_times` (seconds per training epoch, recorded since this study) feeds the table's
  time-per-epoch column via `summarize.py` (median over epochs and seeds). Stage runs share the
  GPU four at a time, so their per-epoch times are contended; the uncontended numbers are the
  probe values in the Facts table.
