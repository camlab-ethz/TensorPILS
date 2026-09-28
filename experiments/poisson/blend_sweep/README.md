# Ill-conditioning prevents physics-informed training (P_t sweep)

Reproduces **Figure 2**: validation curves of the preconditioned least-squares loss with the
spectral blend

    P_t = (1 - t) I + t A⁻¹,

which interpolates between the unpreconditioned loss (`t = 0`, `L_LS`) and the supervised one
(`t = 1`), together with the geometric multigrid preconditioner; and the test error against the
condition number `κ(H_t)` of the loss Hessian `H_t = A P_t² A`.

## Setup

Poisson with `K = 4` on the 64 × 64 grid (bilinear elements), FNO, 1024/128/256 samples, 500
epochs, batch 32, Adam with learning rate `1e-3` cosine-annealed to `1e-4`, seed 42. `P_t` is
realised exactly from one dense eigendecomposition of the stiffness matrix, which also gives
`κ(H_t)` in closed form (it is stored in each run's results JSON). The multigrid reference is one
V(2,2) cycle with the default Jacobi damping 2/3.

## Run

From the repository root, on a machine with a CUDA GPU:

```bash
bash experiments/poisson/blend_sweep/run.sh           # 8 runs, ~15 minutes each
python experiments/poisson/blend_sweep/plot_paper.py  # -> output/poisson/blend_sweep/figures/
```

## Expected results

`κ(H_0) = 6.5 · 10⁵` and `κ(H_1) = 1`. The unpreconditioned run (`t = 0`) stays near 30 % relative
error after 500 epochs, the supervised end (`t = 1`) reaches below 1 %, and intermediate `t`
interpolate between the two; below `κ(H_t) ≈ 10³–10⁴` (`t` between 0.05 and 0.1) further
preconditioning brings little. The multigrid run is indistinguishable from `t = 1`.
