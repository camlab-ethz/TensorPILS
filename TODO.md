# TODO

Open items that are known but deliberately deferred. Newest first.

## Allen-Cahn: conditioning study is at 128, the finals are at 129 (low priority)

`experiments/allen_cahn/ac_paper/conditioning/jacobian_conditioning.py` and its results
(`output/allen_cahn/ac_paper/conditioning/`) were computed at `--grid 128` (h = 1/127), which
matches the lr sweep but NOT the finals the paper reports: `sweep.sbatch` sets `GRID=129`
(h = 1/128, commit 4af935a). Two consequences:

- the quoted condition numbers describe a slightly coarser grid than the reported runs;
- at 129 the multigrid hierarchy is nested (129 -> 65 -> 33 -> 17), while at 128 it is not
  (127 intervals cannot be halved), so the V-cycle quality reported there is, if anything,
  pessimistic.

Rerun with `--grid 129` (~11 min on a laptop CPU) when the conditioning numbers are written
up; nothing else depends on them.

## Poisson: data-driven labels do not match the paper's L_data (low priority)

The paper (Appendix A, "Loss Functions") defines

    L_data = 1/(2 N_b) * sum_b || u(theta, f_b) - u*_b ||_2^2,   A u*_b = f_b,

i.e. the error against the **finite element solution**, on the interior nodes, with a 1/2.
The Poisson benchmark runs (`experiments/poisson/poisson_paper/poisson_benchmark/`) instead
trained and were scored with:

- **labels:** the closed-form solution sampled at the nodes (`--dataset_solution analytic`, the
  default), not the FEM solution;
- **boundary:** `--bc_mode penalty` (default), so the data MSE runs over all n_x^2 grid nodes of
  the raw network output, boundary included (label 0 there), not over the interior coefficients;
- **scale:** sum over nodes, mean over batch, no 1/2 (`DataLoss` in `tensorpils/losses.py`). Under
  Adam a constant factor is immaterial.

Expected impact: small. The FEM solution differs from the analytic one by the discretisation
error (~0.7% relative at 65^2, K=10), below the reported errors (~5%).

To make experiments and paper agree, rerun the Poisson benchmark with
`--dataset_solution fem` (labels for train/val/test become the discrete solution of
A u = M f) and `--bc_mode hard` for the data arm (interior-only MSE, boundary projected at
eval). Note that `fem` also changes what *every* arm's reported error is measured against
(the discrete solution instead of the analytic one), so all rows of the Poisson table and
Figures 2-3 would need regenerating together, not just the data arm. Also update the
"Neural Operator" paragraph of Appendix A, which currently says the zero boundary is
hard-enforced by projection for all arms; for the data arm under `penalty` it is not.
