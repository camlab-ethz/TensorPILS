# Experiment: Poisson baselines — PINO and PI-DeepONet

**The question.** Does preconditioned physics-informed training actually beat the published
label-free methods, or only our own bare-residual straw man? Seven arms, at the settings that
produced the paper's Poisson numbers.

See [`../README.md`](../README.md) for why PINO is ported with finite differences (its Dirichlet
example, Darcy, uses them), what else was copied from the reference implementation, and the
measurement showing the FD and FEM residuals are the same PDE at the same `O(h²)` accuracy.

## Arms

| arch | `--model` | `--loss` | labels | derivative | status |
|---|---|---|---|---|---|
| FNO | `fno` | `data` | yes | — | reference (supervised gold standard) |
| FNO | `fno` | `galerkin` | no | FEM | the existing negative control (`t=0`) |
| FNO | `fno` | **`pino`** | no | FD | **new baseline** |
| FNO | `fno` | `pls` | no | FEM + multigrid `P` | ours |
| DeepONet | `deeponet` | `data` | yes | — | **new** — required to read the row below |
| DeepONet | `deeponet` | **`pideeponet`** | no | autodiff | **new baseline** |
| DeepONet | `deeponet` | `pls` | no | FEM + multigrid `P` | **new** — ours, other architecture |

Within each architecture block only the loss changes, so the loss effect is not confounded with
the architecture. The supervised DeepONet row is what separates "PI-DeepONet's objective is bad"
from "a DeepONet cannot fit this problem at all" — without it the baseline row is uninterpretable.

## Facts

| | |
|---|---|
| Grid | `64²`, `Q1`, zero Dirichlet |
| Dataset | `K=4` multi-frequency, `n_train=1024`, `n_val=128`, `n_test=256`, seeds 42/43/44 |
| Budget | 500 epochs, batch 32, cosine `lr → 1e-6` |
| FNO | modes `(16,16)`, hidden 64, 5 layers |
| DeepONet | `p=128`, width 256, depth 4, 64 random Fourier trunk features (scale 2) |
| BC | `pino`/`pideeponet` impose it **hard** via the `sin(pi x)sin(pi y)` mollifier (the reference's choice); other arms are projected at eval as usual |
| Metric | FEM relative-`L2` on the test set, boundary projected — identical to every other table in the paper |
| Output | `output/baselines/poisson/seed<N>/` (git-ignored) |

## Run

```bash
mkdir -p logs

# 1. tune the four NEW arms (100 epochs x 5 learning rates each), then read the winners
sbatch experiments/baselines/poisson/lr_sweep.sbatch
python experiments/baselines/poisson/plot_baselines.py \
    --results_dir output/baselines/poisson/lr --lr_table
#    ... transcribe the 'best' column into ARMS_LR in sweep.sbatch ...

# 2. the 7-arm table, 3 seeds
sbatch experiments/baselines/poisson/sweep.sbatch

# 3. aggregate
python experiments/baselines/poisson/plot_baselines.py \
    --results_dir output/baselines/poisson
```

A single arm directly:

```bash
python -m tensorpils.cli --pde poisson --loss pino --model fno \
    --grid_resolution 64 --n_train 1024 --n_val 128 --n_test 256 -k 4 \
    --epochs 500 --lr 1e-3 --seed 42 --device cuda \
    --output_dir output/baselines/poisson/seed42

python -m tensorpils.cli --pde poisson --loss pideeponet --model deeponet ...
```

## Reading the result

Both figures matter, and the curve one is not decoration. The bare `galerkin` arm is known to be
*still descending* at 500 epochs, so a final-number-only table would license the overclaim "it does
not train" when the measured statement is "it trains far more slowly". The same caution applies to
the two baselines: check `baselines.png` before writing any sentence with "fails" in it.

Expected, and worth stating in advance so the result is not read backwards: PINO's FD residual has
the **same** `O(h⁻⁴)` Hessian conditioning as our bare FEM least-squares loss, so the prediction is
that PINO lands near the `galerkin` arm and not near `pls`. If it lands materially better, the
mollifier and the relative-`Lp` reduction are the two candidate explanations, and the h-refinement
follow-up (below) is what distinguishes "better constant" from "better scaling".

## Follow-up worth running

`pino` at `33/65/129` alongside the existing `h_refinement` arms. Conditioning predicts the
baseline's error degrades under refinement the way the bare residual does; that experiment ties the
baseline to the paper's *central* claim rather than merely adding a table row.
