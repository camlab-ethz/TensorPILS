"""Command-line entry point: ``tensorpils`` (or ``python -m tensorpils.cli``).

Trains an FNO to solve one of four PDEs (``--pde``):

* ``poisson`` (static) with one of four losses: ``data`` / ``galerkin`` / ``deepritz`` / ``pls``.
* ``wave`` (time-dependent) as an autoregressive time-stepper with a ``galerkin`` (label-free
  central-difference residual) and/or ``data`` (supervised) loss.
* ``ac`` (Allen–Cahn, time-dependent nonlinear) as an autoregressive time-stepper with a
  ``galerkin`` (label-free backward-Euler residual) and/or ``data`` (supervised) loss.
* ``stokes`` (static saddle point) on a Taylor-Hood Q2/Q1 pair, with ``data`` (supervised),
  ``galerkin`` (bare ``½‖Kc−b‖²``) or ``pls`` (block-preconditioned ``½ rᵀPr``).
"""

from argparse import ArgumentParser

import numpy as np
import torch

from .data import (create_datasets, create_scaling_datasets, PoissonDataset,
                   create_wave_datasets, create_ac_datasets, create_stokes_datasets)
from .models import FNOModel
from .gaot import GAOTModel
from .trainer import (PoissonTrainer, WaveTrainer, ACTrainer, StokesTrainer,
                      GAOTPoissonTrainer, GAOTWaveTrainer, GAOTACTrainer,
                      GAOTStokesTrainer)
from .baselines import (DeepONetModel, MollifiedModel, ZeroBoundaryModel,
                        PINOPoissonTrainer, PINOACTrainer, PINOStokesTrainer,
                        PIDeepONetPoissonTrainer, PIDeepONetACTrainer, PIDeepONetStokesTrainer,
                        DeepONetPoissonTrainer, DeepONetACTrainer, DeepONetStokesTrainer)


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="FNO training for 2D Poisson / wave / Allen-Cahn / Stokes "
                                   "(data / Galerkin / Deep Ritz / PLS losses).")
    p.add_argument("--pde", choices=["poisson", "wave", "ac", "stokes"], default="poisson",
                   help="Which PDE to train on. 'poisson'/'stokes' (static), "
                        "'wave'/'ac' (time-dependent).")
    p.add_argument("--loss",
                   choices=["data", "data_l2", "data_h1", "galerkin", "deepritz", "pls",
                            "pino", "pideeponet"],
                   default="galerkin",
                   help="poisson: data/data_l2/data_h1/galerkin/deepritz/pls. wave/ac: data or "
                        "galerkin (preset for the lambda_galerkin/lambda_data mix). "
                        "BASELINES (poisson/ac/stokes): 'pino' = strong-form residual by finite "
                        "differences (hard zero Dirichlet BC by zeroing the boundary nodes, as "
                        "our own arms do; see --pino_bc); 'pideeponet' = the same strong form by "
                        "autodiff through the trunk, and requires --model deeponet.")
    p.add_argument("--n_train", type=int, default=1024)
    p.add_argument("--n_val", type=int, default=128)
    p.add_argument("--n_test", type=int, default=256)
    p.add_argument("-k", "--k", type=int, default=4, help="K x K source/IC complexity")

    # -------- Data-scaling / streaming mode (poisson) --------
    p.add_argument("--stream", action="store_true",
                   help="Poisson: train on an infinite stream of fresh samples (none seen twice; "
                        "the infinite-data limit). Ignores --n_train for training; requires "
                        "--steps_per_epoch. Implies --fixed_eval.")
    p.add_argument("--steps_per_epoch", type=int, default=None,
                   help="Poisson: fixed number of optimizer steps per epoch, independent of "
                        "n_train (finite datasets are sampled i.i.d. with replacement). "
                        "Equalizes the optimization budget across dataset sizes.")
    p.add_argument("--patience", type=int, default=None,
                   help="Poisson: early-stop when the validation error has not improved for "
                        "this many epochs (default: off).")
    p.add_argument("--dataset_solution", choices=["analytic", "fem"], default="analytic",
                   help="Poisson labels: 'analytic' (default) samples the closed-form solution "
                        "at the nodes; 'fem' solves A u = M f with Q1 on the regular grid, i.e. "
                        "labels become the discrete solution the FEM losses target. Applies to "
                        "train, val AND test, so under 'fem' the reported error is measured "
                        "against the discrete solution and the residual losses lose their "
                        "discretisation floor (~0.7%% relative at 65^2, K=10).")
    p.add_argument("--fixed_eval", action="store_true",
                   help="Poisson: draw val/test from dedicated seeds (seed+1/seed+2), identical "
                        "for every --n_train — all runs of a dataset-size sweep share the same "
                        "eval sets. Default: the usual joint train/val/test pool.")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # -------- Poisson boundary handling --------
    p.add_argument("--lambda_bc", type=float, default=100.0,
                   help="BC penalty weight for Deep Ritz when --bc_mode penalty.")
    p.add_argument("--bc_mode", choices=["penalty", "hard"], default="penalty",
                   help="Poisson boundary handling. Deep Ritz: 'penalty' (soft, uses lambda_bc) or "
                        "'hard' (project u->0 before energy). Data loss: 'penalty' (full-grid "
                        "MSE) or 'hard' (interior-only MSE, boundary unconstrained).")
    p.add_argument("--precondition", action="store_true",
                   help="Deep Ritz only: precondition the energy gradient with a GMG V-cycle "
                        "(M~A^-1). Flattens the A-norm dynamics toward the supervised/Newton "
                        "direction. Implies hard-BC.")

    # -------- Preconditioner selection (PLS, or preconditioned Deep Ritz) --------
    p.add_argument("--precond_kind", choices=["multigrid", "amg", "blend", "power"],
                   default="multigrid",
                   help="Preconditioner P≈A^-1. 'multigrid' (default): geometric-multigrid "
                        "V-cycle (computational path). 'amg': algebraic (AmgX) V-cycle — same "
                        "contract, but its hierarchy comes from the matrix instead of the grid, "
                        "so it is the path to unstructured meshes; CUDA-only, and it uses "
                        "--mg_pre_smooth for both pre- and post-smoothing (they must be equal). "
                        "'blend': convex mix (1-t)I+tA^-1. "
                        "'power': fractional power A^-s. blend/power are exact spectral "
                        "operators for illustrating the residual->supervised transition.")
    p.add_argument("--precond_strength", type=float, default=1.0,
                   help="Strength for blend (t) / power (s) in [0,1]. 0 -> P=I "
                        "(no preconditioning); 1 -> P=A^-1 (supervised). Ignored for multigrid.")
    p.add_argument("--precond_method", choices=["dense", "sine"], default="dense",
                   help="Realization for blend/power: 'dense' (default) eigendecomposition, or "
                        "'sine' = fast DST equivalent for a uniform grid (needed at 128²/256², "
                        "where the dense route is infeasible). Ignored for multigrid.")

    # -------- Multigrid preconditioner settings (used when --precond_kind multigrid) --------
    p.add_argument("--mg_levels", type=int, default=4,
                   help="Number of GMG levels for PLS (incl. finest).")
    p.add_argument("--mg_pre_smooth", type=int, default=2,
                   help="Pre-smoothing Jacobi sweeps per V-cycle level.")
    p.add_argument("--mg_post_smooth", type=int, default=2,
                   help="Post-smoothing Jacobi sweeps per V-cycle level.")
    p.add_argument("--mg_omega", type=float, default=2.0 / 3.0,
                   help="Damping factor for weighted Jacobi smoother.")

    # -------- Wave equation (time-dependent) --------
    p.add_argument("--wave_c", type=float, default=1.0, help="Wave speed c.")
    p.add_argument("--wave_r", type=float, default=0.5,
                   help="Spectral decay exponent r for WaveMultiFrequency.")
    p.add_argument("--dt", type=float, default=0.005, help="Wave time step.")
    p.add_argument("--n_steps", type=int, default=20,
                   help="Number of analytical trajectory frames generated per sample.")
    p.add_argument("--rollout_steps", type=int, default=4,
                   help="Autoregressive rollout length used for the loss and eval.")
    p.add_argument("--discount_factor", type=float, default=1.0,
                   help="Per-step weighting of the Galerkin residual across the rollout.")
    p.add_argument("--lambda_galerkin", type=float, default=None,
                   help="Weight of the Galerkin residual (wave/ac; default from --loss preset).")
    p.add_argument("--lambda_data", type=float, default=None,
                   help="Weight of the trajectory MSE (wave/ac; default from --loss preset).")

    # -------- Allen–Cahn (time-dependent, nonlinear) --------
    p.add_argument("--ac_a", type=float, default=1.0, help="Allen–Cahn diffusion coefficient a.")
    p.add_argument("--ac_eps", type=float, default=2.0, help="Allen–Cahn reaction strength eps.")
    p.add_argument("--ac_r", type=float, default=0.5,
                   help="Spectral decay exponent r for the multi-frequency initial condition.")
    p.add_argument("--ac_newton_tol", type=float, default=1e-8,
                   help="Newton tolerance for the FEM reference solver.")
    p.add_argument("--ac_newton_max", type=int, default=20,
                   help="Max Newton iterations per step for the FEM reference solver.")
    p.add_argument("--ac_ref_chunk", type=int, default=64,
                   help="Sample chunk size for the batched FEM reference solve (memory control).")
    p.add_argument("--ac_ref_device", choices=["cpu", "cuda"], default="cpu",
                   help="Device for the FEM reference solve. 'cpu' (default): scipy sparse direct "
                        "solver, exact and deterministic. 'cuda': without cupy this is torch_sla's "
                        "iterative PBiCGStab, which can break down and has produced NaN labels.")
    p.add_argument("--ac_integrator", choices=["convex_concave", "backward_euler"],
                   default="convex_concave",
                   help="Allen–Cahn time integrator for BOTH the reference data and the physics "
                        "(Galerkin) residual loss (kept coherent). Default: convex_concave (Eyre).")
    p.add_argument("--ac_loss_form", choices=["galerkin", "min_movement"], default="galerkin",
                   help="Form of the AC label-free physics loss: 'galerkin' (½‖R‖² least-squares "
                        "residual) or 'min_movement' (convex–concave JKO objective J; Deep-Ritz "
                        "analogue). Default: galerkin.")
    p.add_argument("--ac_loss_integrator", choices=["convex_concave", "backward_euler"],
                   default=None,
                   help="Override the residual integrator used by the Galerkin physics loss, "
                        "decoupled from the reference data (--ac_integrator). Default: follow the "
                        "data. Lets a BE-residual loss train on convex_concave reference data.")
    p.add_argument("--ac_precond", choices=["none", "multigrid"], default="none",
                   help="Precondition the AC least-squares residual (--ac_loss_form galerkin only): "
                        "'multigrid' trains ½‖P R‖² with P ≈ J0^-1, J0 = a²A + cM the frozen (u²=1) "
                        "Newton Jacobian (c = 1/dt + 3ε²), removing the κ² conditioning of the bare "
                        "residual. Uses the --mg_* V-cycle settings. Default: none (bare ½‖R‖²).")
    p.add_argument("--bptt_mode", choices=["full_bptt", "detach_prev", "pushforward"], default=None,
                   help="Autodiff/backprop-through-time strategy for the rollout. 'full_bptt': no "
                        "detach (backprop through the whole rollout). 'detach_prev': detach the "
                        "previous-frame coupling in the loss (MM proximal centre / Galerkin R) but "
                        "keep full BPTT. 'pushforward': also detach each rollout input, so gradients "
                        "are one-step (Brandstetter et al.). Default: None = current per-loss "
                        "behaviour (MM detaches the coupling; Galerkin/data do full BPTT).")

    # -------- Optimizer (the place to plug in Shampoo via build_optimizer) --------
    p.add_argument("--optimizer", type=str, default="adam",
                   help="adam | adamw | sgd | <register-yours-in build_optimizer>")

    # -------- Out-of-distribution generalization eval (poisson): relative-L2 each epoch on
    #          datasets with different source complexity K (same grid). E.g. --ood_k 6 8. --------
    p.add_argument("--ood_k", type=int, nargs="+", default=[],
                   help="Extra K values to evaluate each epoch (out-of-distribution sources).")
    p.add_argument("--ood_n_val", type=int, default=128,
                   help="Number of samples per OOD eval dataset.")
    p.add_argument("--ood_seed", type=int, default=123,
                   help="Seed for the OOD eval datasets (fixed across runs for fair comparison).")

    # -------- Stokes (saddle point, Taylor-Hood Q2/Q1) --------
    p.add_argument("--stokes_mu", type=float, default=1.0,
                   help="Stokes: dynamic viscosity mu.")
    p.add_argument("--stokes_r", type=float, default=-0.5,
                   help="Stokes: spectral decay exponent of the random body force "
                        "(amplitude ~ (k^2+l^2)^r), matching the Poisson source convention.")
    p.add_argument("--stokes_ref_chunk", type=int, default=64,
                   help="Stokes: right-hand sides per batched reference solve (memory control).")
    p.add_argument("--stokes_no_normalize", action="store_true",
                   help="Stokes: do NOT rescale each sample to unit reference velocity norm "
                        "(the rescaling is exact — Stokes is linear — and just fixes the "
                        "output scale the FNO must hit).")
    p.add_argument("--schur_omega", type=float, default=0.5,
                   help="Stokes pls: relaxation omega on the lumped-pressure-mass Schur surrogate "
                        "S^-1 = omega*mu*diag(M_p)^-1. The note's (0,1] restriction applies to "
                        "omega inside an Uzawa *iteration*; as a norm weight it is unconstrained, "
                        "and kappa(KPK) is minimized near omega=16 (see "
                        "experiments/stokes/conditioning/omega_scan.py).")
    p.add_argument("--stokes_precond", choices=["block", "monolithic"], default="block",
                   help="Stokes pls: 'block' = P = diag(A_hat^-1, S_hat^-1), only NORM-equivalent "
                        "to K^-1, used as a norm weight (kappa = O(h^-2)). 'monolithic' = a full "
                        "Stokes V-cycle with a symmetric Uzawa smoother, a genuine P ~ K^-1, which "
                        "unlocks the applied form at O(1) conditioning.")
    p.add_argument("--stokes_pls_form",
                   choices=["auto", "weighted", "applied", "applied_fe"], default="auto",
                   help="Stokes pls: 'weighted' = 0.5*r^T P r; 'applied' = 0.5*||P r||^2 in nodal "
                        "units; 'applied_fe' = the same with each field block divided by its "
                        "dataset FE norm, which removes the ~350x pressure domination the nodal "
                        "norm carries here. 'auto' (default) picks weighted for --stokes_precond "
                        "block and applied_fe for monolithic. Forcing applied+block is the note's "
                        "documented dead end and exists only as a control.")
    p.add_argument("--uzawa_pre", type=int, default=4,
                   help="Stokes monolithic: symmetric Uzawa sweeps before the coarse-grid "
                        "correction (nu_1).")
    p.add_argument("--uzawa_post", type=int, default=4,
                   help="Stokes monolithic: symmetric Uzawa sweeps after (nu_2).")
    p.add_argument("--cheb_degree", type=int, default=8,
                   help="Stokes monolithic: Chebyshev-Jacobi degree for A_hat^-1 inside the Uzawa "
                        "smoother (0 falls back to weighted Jacobi).")
    p.add_argument("--cheb_ratio", type=float, default=30.0,
                   help="Stokes monolithic: Chebyshev targets [lambda_max/ratio, lambda_max] of "
                        "D^-1 A — a smoother damps the top of the spectrum and leaves the rest to "
                        "the coarse grid.")
    p.add_argument("--stokes_precond_strength", type=float, default=1.0,
                   help="Stokes pls: blend strength t in [0,1] for P_t = (1-t)*alpha*I + "
                        "t*P_block. t=1 (default) is the block preconditioner; t=0 reproduces the "
                        "bare least-squares loss. Sweeping t moves kappa(KPK) continuously from "
                        "O(h^-4) to O(h^-2).")

    # -------- FNO model --------
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--n_modes", type=int, nargs=2, default=[16, 16])
    p.add_argument("--grid_resolution", type=int, default=64)

    # -------- architecture / baselines (experiments/baselines/) --------
    p.add_argument("--model", choices=["fno", "deeponet", "gaot"], default="fno",
                   help="Neural-operator architecture. All three share the [B,C,H,W] -> "
                        "[B,C_out,H,W] signature, so any of them can be paired with ANY loss. "
                        "'deeponet' additionally exposes a coordinate query, which is what "
                        "--loss pideeponet differentiates through. 'gaot' is the geometry-aware "
                        "operator transformer: it works on a point cloud internally and is the "
                        "arm that carries over to an unstructured mesh, where the FNO's FFT "
                        "cannot go -- see tensorpils/gaot/.")
    p.add_argument("--deeponet_p", type=int, default=128,
                   help="DeepONet: number of basis functions (branch/trunk latent width).")
    p.add_argument("--deeponet_width", type=int, default=256,
                   help="DeepONet: hidden width of the branch and trunk MLPs.")
    p.add_argument("--deeponet_depth", type=int, default=4,
                   help="DeepONet: number of Linear layers in each of branch and trunk.")
    p.add_argument("--trunk_fourier", type=int, default=64,
                   help="DeepONet: random Fourier features on the trunk input (0 disables). "
                        "A plain MLP trunk has a strong spectral bias and underfits "
                        "multi-frequency data for reasons unrelated to the loss.")
    p.add_argument("--trunk_fourier_scale", type=float, default=2.0,
                   help="DeepONet: std of the random Fourier frequency matrix.")
    # -------- GAOT (geometry-aware operator transformer, tensorpils/gaot/) --------
    # Defaults reproduce GAOT's published Poisson-Gauss setting at 64^2: latent 64x64, patch 2,
    # lifting 64, transformer 256 wide and 3 deep, and a radius of 0.033 (see --gaot_radius).
    p.add_argument("--gaot_latent", type=int, nargs=2, default=[64, 64],
                   help="GAOT: structured latent token grid (H W). Sets the transformer's cost, "
                        "independently of how the physical points are arranged -- which is why "
                        "the same processor serves a structured and an unstructured mesh.")
    p.add_argument("--gaot_patch", type=int, default=2,
                   help="GAOT: patch side on the latent grid; (H/P)*(W/P) transformer tokens. "
                        "Raise it to cut attention cost quadratically.")
    p.add_argument("--gaot_radius", type=float, default=None,
                   help="GAOT: neighbour-ball radius in domain units for both MAGNO searches. "
                        "Default (None) derives it as --gaot_radius_scale * max(h_phys, "
                        "h_latent), so changing resolution keeps the neighbour count roughly "
                        "fixed instead of emptying or exploding the graph.")
    p.add_argument("--gaot_radius_scale", type=float, default=2.1,
                   help="GAOT: multiplier in the derived radius (~13 neighbours per query on a "
                        "uniform grid; 2.1 reproduces GAOT's 0.033 at 64^2 latent 64^2).")
    p.add_argument("--gaot_scales", type=float, nargs="+", default=[1.0],
                   help="GAOT: multiscale radii as multiples of the base radius; MAGNO averages "
                        "the encodings over them.")
    p.add_argument("--gaot_lifting", type=int, default=64,
                   help="GAOT: channels per latent token (the width the transformer sees).")
    p.add_argument("--gaot_magno_hidden", type=int, default=64,
                   help="GAOT: hidden width of the AGNO kernel MLPs.")
    p.add_argument("--gaot_mlp_layers", type=int, default=3,
                   help="GAOT: depth of the AGNO kernel MLPs.")
    p.add_argument("--gaot_hidden", type=int, default=256,
                   help="GAOT: transformer hidden size.")
    p.add_argument("--gaot_layers", type=int, default=3,
                   help="GAOT: number of transformer blocks (UViT: half encoder, half decoder "
                        "with long-range skips, plus a middle block when odd).")
    p.add_argument("--gaot_heads", type=int, default=8,
                   help="GAOT: attention heads.")
    p.add_argument("--gaot_no_geoembed", action="store_true",
                   help="GAOT: disable the geometric embedding of each neighbourhood. It encodes "
                        "local point density, so it is close to inert on a uniform grid and "
                        "earns its keep on an irregular one -- this flag is the ablation.")
    p.add_argument("--gaot_attention", choices=["cosine", "dot_product"], default="cosine",
                   help="GAOT: neighbour weighting inside the AGNO kernel integral.")
    p.add_argument("--gaot_node_embedding", action="store_true",
                   help="GAOT: sinusoidal encoding of coordinates before the kernel MLP.")
    p.add_argument("--gaot_pos_embedding", choices=["absolute", "rope"], default="absolute",
                   help="GAOT: transformer positional embedding ('rope' needs "
                        "rotary-embedding-torch).")

    p.add_argument("--mollify", choices=["auto", "on", "off"], default="auto",
                   help="Impose the zero Dirichlet BC hard as part of the model. 'auto' turns it "
                        "on for the baseline losses (pino/pideeponet) and off otherwise so "
                        "existing arms are unchanged. HOW it is imposed depends on the arm: "
                        "PI-DeepONet multiplies by sin(pi x)sin(pi y) (the reference's "
                        "mollifier); PINO zeroes the boundary nodes -- see --pino_bc.")
    p.add_argument("--pino_bc", choices=["zero", "mollifier"], default="zero",
                   help="PINO only: how the hard zero Dirichlet BC is imposed. 'zero' (default) "
                        "zeroes the boundary nodes, exactly as losses.py does to our own arms via "
                        "apply_zero_boundary, so PINO differs from the galerkin arm only in the "
                        "residual. 'mollifier' multiplies by sin(pi x)sin(pi y), reproducing the "
                        "reference -- but that injects a decay profile close to this dataset's own "
                        "structure, which our arms never get.")
    p.add_argument("--pideeponet_bc", choices=["mollifier", "zero"], default="mollifier",
                   help="pideeponet only: how the zero Dirichlet BC enters. 'mollifier' (default) "
                        "multiplies every query by sin(pi x)sin(pi y), as the reference does. "
                        "'zero' is the counterpart of --pino_bc zero: the grid output's boundary "
                        "nodes are set to 0 exactly as our arms do, and -- because the autodiff "
                        "Laplacian at interior points cannot see that -- the residual gets the "
                        "original PI-DeepONet soft boundary penalty, weighted by --pi_lambda_bc.")
    p.add_argument("--pi_lambda_bc", type=float, default=1.0,
                   help="pideeponet with --pideeponet_bc zero: weight of the boundary penalty. "
                        "The penalty is expressed in the residual's own units (see "
                        "PIDeepONetPoissonLoss), so 1 is a meaningful default.")
    p.add_argument("--pi_div_weight", type=float, default=1.0,
                   help="pino/pideeponet (stokes): weight of the continuity residual div(u) "
                        "relative to the momentum residual in the stacked strong-form loss. "
                        "The two equations have different units, so this is tuned on "
                        "validation like a learning rate (see baselines.pino.stokes_reduce).")
    p.add_argument("--pi_n_colloc", type=int, default=0,
                   help="pideeponet: collocation points per step, drawn afresh from the grid "
                        "nodes each step. 0 (default) uses all nodes, which matches every "
                        "other arm's discretisation budget.")
    p.add_argument("--pino_reduction", choices=["rel", "mse"], default="rel",
                   help="pino/pideeponet (poisson): 'rel' is the reference implementation's "
                        "relative-Lp ratio ||Lu-f||/||f||; 'mse' is the plain mean square.")

    # -------- Misc --------
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_dir", type=str, default="output")

    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--no_checkpoint", action="store_true",
                   help="Do not write the best-model .pth. Model selection is unaffected (the best "
                        "state is kept in memory and restored at the end); this only skips the file, "
                        "which is worth doing in learning-rate sweeps -- a checkpoint carries the "
                        "optimizer state as well, ~3x the parameter count per run.")
    p.add_argument("--sample_idx", type=int, nargs="+", default=[0])
    return p


def run_poisson(args, device):
    print(f"loss        : {args.loss}"
          + ("  (preconditioned)" if (args.loss == "deepritz" and args.precondition) else ""))
    if args.loss == "deepritz" and not args.precondition:
        print(f"bc_mode     : {args.bc_mode}"
              + (f"  (lambda_bc={args.lambda_bc})" if args.bc_mode == "penalty" else ""))
    if args.loss == "data":
        print(f"bc_mode     : {args.bc_mode}"
              + ("  (interior-only MSE)" if args.bc_mode == "hard" else "  (full-grid MSE)"))
    if args.loss == "pls" or (args.loss == "deepritz" and args.precondition):
        if args.precond_kind == "multigrid":
            print(f"precond     : multigrid  levels={args.mg_levels}  "
                  f"smooth={args.mg_pre_smooth}/{args.mg_post_smooth}  omega={args.mg_omega:.3f}")
        else:
            print(f"precond     : {args.precond_kind}  strength={args.precond_strength:.3f}"
                  f"  method={args.precond_method}")
    if args.stream and args.steps_per_epoch is None:
        raise SystemExit("--stream requires --steps_per_epoch (an epoch has no natural "
                         "length on an infinite stream).")
    if args.stream or args.steps_per_epoch or args.patience or args.fixed_eval:
        print(f"data mode   : {'stream (fresh samples every step)' if args.stream else 'finite'}"
              + (f"  steps/epoch={args.steps_per_epoch}" if args.steps_per_epoch else "")
              + (f"  patience={args.patience}" if args.patience else "")
              + ("  fixed-eval split" if (args.stream or args.fixed_eval) else ""))
    _print_common(args, device)

    print("Building datasets...")
    if args.stream or args.fixed_eval:
        train_ds, val_ds, test_ds = create_scaling_datasets(
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
            K=args.k, grid_resolution=args.grid_resolution, seed=args.seed,
            stream_samples_per_epoch=(args.steps_per_epoch * args.batch_size
                                      if args.stream else None),
            solution=args.dataset_solution,
        )
    else:
        train_ds, val_ds, test_ds = create_datasets(
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
            K=args.k, grid_resolution=args.grid_resolution, seed=args.seed,
            solution=args.dataset_solution,
        )
    ntr = f"stream({len(train_ds)}/epoch)" if args.stream else len(train_ds)
    print(f"  train={ntr}, val={len(val_ds)}, test={len(test_ds)}, "
          f"grid={train_ds.grid_size}\n")

    # Out-of-distribution eval datasets (higher source complexity K, same grid).
    eval_datasets = {
        f"K{k}": PoissonDataset(num_samples=args.ood_n_val, K=k, seed=args.ood_seed,
                                grid_resolution=args.grid_resolution,
                                solution=args.dataset_solution)
        for k in args.ood_k
    }
    if eval_datasets:
        print(f"  OOD eval sets: {', '.join(eval_datasets)} "
              f"(n={args.ood_n_val} each, seed={args.ood_seed})\n")

    model = _build_model(args, in_channels=1, train_ds=train_ds)
    common = dict(
        model=model,
        train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
        loss_type=args.loss,
        optimizer_name=args.optimizer,
        lr=args.lr, lr_min=args.lr_min, weight_decay=args.weight_decay,
        batch_size=args.batch_size, epochs=args.epochs,
        device=device, output_dir=args.output_dir,
        lambda_bc=args.lambda_bc, bc_mode=args.bc_mode, precondition=args.precondition,
        precond_kind=args.precond_kind, precond_strength=args.precond_strength,
        precond_method=args.precond_method,
        mg_levels=args.mg_levels, mg_pre_smooth=args.mg_pre_smooth,
        mg_post_smooth=args.mg_post_smooth, mg_omega=args.mg_omega,
        steps_per_epoch=args.steps_per_epoch, patience=args.patience,
        eval_datasets=eval_datasets,
    )
    # The baseline trainers subclass PoissonTrainer, so validation, model selection and the
    # reported test metric are the inherited ones -- every arm is scored identically.
    if args.loss == "pino":
        trainer = PINOPoissonTrainer(pino_reduction=args.pino_reduction, **common)
    elif args.loss == "pideeponet":
        zero_bc = (args.pideeponet_bc == "zero")
        trainer = PIDeepONetPoissonTrainer(n_colloc=args.pi_n_colloc,
                                           pi_reduction=args.pino_reduction,
                                           pi_bc=args.pideeponet_bc,
                                           lambda_bc_pi=(args.pi_lambda_bc if zero_bc else 0.0),
                                           **common)
    elif args.model in ("deeponet", "gaot"):
        # Same losses, other architecture. The arch-tagged prefix keeps these runs from
        # landing on (and overwriting) the identically-configured FNO run's files.
        trainer = (DeepONetPoissonTrainer if args.model == "deeponet"
                   else GAOTPoissonTrainer)(**common)
    else:
        trainer = PoissonTrainer(**common)
    _run(trainer, train_ds, test_ds, args)


def run_wave(args, device):
    if args.loss not in ("data", "galerkin"):
        raise SystemExit(f"--pde wave supports --loss data|galerkin, got {args.loss!r}")
    # --loss is a preset for the (lambda_galerkin, lambda_data) mix; explicit flags override.
    lg = args.lambda_galerkin
    ld = args.lambda_data
    if lg is None:
        lg = 1.0 if args.loss == "galerkin" else 0.0
    if ld is None:
        ld = 1.0 if args.loss == "data" else 0.0

    h = 1.0 / (args.grid_resolution - 1)
    cfl = h / (args.wave_c * (2 ** 0.5))
    print(f"loss        : {args.loss}  (lambda_galerkin={lg}, lambda_data={ld})")
    print(f"wave        : c={args.wave_c}  r={args.wave_r}  dt={args.dt}  n_steps={args.n_steps}  "
          f"rollout={args.rollout_steps}")
    print(f"CFL         : dt={args.dt:.4g} vs h/(c*sqrt2)={cfl:.4g}  "
          + ("(OK)" if args.dt <= cfl else "(WARNING: above CFL)"))
    _print_common(args, device)

    print("Building datasets...")
    train_ds, val_ds, test_ds = create_wave_datasets(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        K=args.k, grid_resolution=args.grid_resolution,
        dt=args.dt, n_steps=args.n_steps, c=args.wave_c, r=args.wave_r, seed=args.seed,
    )
    print(f"  train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, "
          f"grid={train_ds.grid_size}, frames/sample={args.n_steps + 1}\n")

    model = _build_model(args, in_channels=2)
    trainer = (GAOTWaveTrainer if args.model == "gaot" else WaveTrainer)(
        model=model,
        train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
        loss_type=args.loss,
        optimizer_name=args.optimizer,
        lr=args.lr, lr_min=args.lr_min, weight_decay=args.weight_decay,
        batch_size=args.batch_size, epochs=args.epochs,
        device=device, output_dir=args.output_dir,
        lambda_galerkin=lg, lambda_data=ld,
        rollout_steps=args.rollout_steps, discount_factor=args.discount_factor,
    )
    _run(trainer, train_ds, test_ds, args)


def run_ac(args, device):
    if args.loss not in ("data", "galerkin", "pino", "pideeponet"):
        raise SystemExit("--pde ac supports --loss data|galerkin|pino|pideeponet, "
                         f"got {args.loss!r}")
    # --loss is a preset for the (lambda_galerkin, lambda_data) mix; explicit flags override.
    # The baselines are physics-only, so they sit on the galerkin side of the preset.
    lg = args.lambda_galerkin
    ld = args.lambda_data
    if lg is None:
        lg = 0.0 if args.loss == "data" else 1.0
    if ld is None:
        ld = 1.0 if args.loss == "data" else 0.0

    total = args.n_train + max(args.n_val, 0) + max(args.n_test, 0)
    print(f"loss        : {args.loss}  (lambda_galerkin={lg}, lambda_data={ld})")
    print(f"allen-cahn  : a={args.ac_a}  eps={args.ac_eps}  r={args.ac_r}  dt={args.dt}  "
          f"n_steps={args.n_steps}  rollout={args.rollout_steps}")
    loss_integ = args.ac_loss_integrator or args.ac_integrator
    print(f"integrator  : data={args.ac_integrator}  loss-residual={loss_integ}")
    print(f"phys loss   : {args.ac_loss_form}  "
          f"({'least-squares residual ½‖R‖²' if args.ac_loss_form == 'galerkin' else 'minimizing-movement objective J'})")
    if args.ac_precond != "none":
        c_shift = 1.0 / args.dt + 3.0 * args.ac_eps ** 2
        print(f"precond     : {args.ac_precond}  ½‖P R‖², P≈(a²A+cM)⁻¹  "
              f"(a²={args.ac_a ** 2:g}, c=1/dt+3ε²={c_shift:g}, mg_levels={args.mg_levels})")
    print(f"bptt mode   : {args.bptt_mode or 'default (per-loss)'}")
    print(f"reference   : FEM {args.ac_integrator} + Newton on {args.ac_ref_device} "
          f"(build cost ~ {total}x{args.n_steps} steps; "
          f"no analytical solution exists for AC)")
    _print_common(args, device)

    print("Building datasets (FEM reference solve; may take a moment)...")
    train_ds, val_ds, test_ds = create_ac_datasets(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        K=args.k, grid_resolution=args.grid_resolution,
        dt=args.dt, n_steps=args.n_steps, a=args.ac_a, eps=args.ac_eps, r=args.ac_r,
        newton_tol=args.ac_newton_tol, newton_max=args.ac_newton_max,
        ref_chunk=args.ac_ref_chunk, integrator=args.ac_integrator, seed=args.seed,
        ref_device=args.ac_ref_device,
    )
    print(f"  train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, "
          f"grid={train_ds.grid_size}, frames/sample={args.n_steps + 1}\n")

    model = _build_model(args, in_channels=1, train_ds=train_ds)
    common = dict(
        model=model,
        train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
        loss_type=args.loss,
        optimizer_name=args.optimizer,
        lr=args.lr, lr_min=args.lr_min, weight_decay=args.weight_decay,
        batch_size=args.batch_size, epochs=args.epochs,
        device=device, output_dir=args.output_dir,
        lambda_galerkin=lg, lambda_data=ld,
        rollout_steps=args.rollout_steps, discount_factor=args.discount_factor,
        ac_loss_form=args.ac_loss_form, ac_integrator=args.ac_loss_integrator,
        bptt_mode=args.bptt_mode,
        ac_precond=("" if args.ac_precond == "none" else args.ac_precond),
        mg_levels=args.mg_levels, mg_pre_smooth=args.mg_pre_smooth,
        mg_post_smooth=args.mg_post_smooth, mg_omega=args.mg_omega,
    )
    # Both baselines inherit ACTrainer's rollout evaluation, so their reported per-step /
    # space-time relative L2 is produced by exactly the same code as the FEM arms.
    arch_cls = {"deeponet": DeepONetACTrainer, "gaot": GAOTACTrainer}.get(args.model, ACTrainer)
    cls = {"pino": PINOACTrainer, "pideeponet": PIDeepONetACTrainer}.get(args.loss, arch_cls)
    trainer = cls(**common)
    _run(trainer, train_ds, test_ds, args)


def run_stokes(args, device):
    if args.loss not in ("data", "galerkin", "pls", "pino", "pideeponet"):
        raise SystemExit(f"--pde stokes supports --loss data|galerkin|pls|pino|pideeponet, "
                         f"got {args.loss!r}")
    if args.grid_resolution % 2 == 0:
        raise SystemExit(
            f"--pde stokes needs an odd --grid_resolution (the Q2 velocity grid has "
            f"2*n_p-1 nodes), got {args.grid_resolution}. Try {args.grid_resolution + 1}.")
    n_p = (args.grid_resolution + 1) // 2

    mono = (args.stokes_precond == "monolithic")
    applied = args.stokes_pls_form.startswith("applied") or (args.stokes_pls_form == "auto" and mono)
    blurb = {
        "data": "supervised FE-L2 vs the discrete Taylor-Hood solution",
        "galerkin": "bare least squares ½‖Kc−b‖²  (kappa = O(h^-4): the negative control)",
        "pls": (("applied preconditioned least squares ½‖Pr‖²" if applied
                 else "preconditioned least squares ½ rᵀPr")
                + (",  P = monolithic Stokes V-cycle ~ K^-1" if mono
                   else ",  P = diag(mg/mu, w*mu/diag(M_p))")),
        "pino": (f"PINO baseline: strong-form momentum + continuity by central differences on "
                 f"the velocity grid, interior nodes, {args.pino_reduction} reduction, "
                 f"div weight {args.pi_div_weight:g}"),
        "pideeponet": (f"PI-DeepONet baseline: strong-form momentum + continuity by autodiff "
                       f"through the trunk, {args.pino_reduction} reduction, div weight "
                       f"{args.pi_div_weight:g}, BC={args.pideeponet_bc}"),
    }[args.loss]
    print(f"loss        : {args.loss}  ({blurb})")
    print(f"stokes      : mu={args.stokes_mu}  force decay r={args.stokes_r}")
    print(f"spaces      : Taylor-Hood Q2/Q1 — velocity {args.grid_resolution}^2 (2 comps), "
          f"pressure {n_p}^2")
    if args.loss == "pls" and mono:
        print(f"precond     : monolithic MG, levels={args.mg_levels}, "
              f"Uzawa pre/post={args.uzawa_pre}/{args.uzawa_post}, "
              f"cheb_degree={args.cheb_degree} (ratio {args.cheb_ratio:g}), "
              f"schur_omega={args.schur_omega}")
    elif args.loss == "pls":
        print(f"precond     : block, mg_levels={args.mg_levels} "
              f"(pre/post={args.mg_pre_smooth}/{args.mg_post_smooth}), "
              f"schur_omega={args.schur_omega}, "
              f"blend t={args.stokes_precond_strength:g}")
    print("reference   : discrete Taylor-Hood solve of K c = (M_u f, 0) "
          "(batched sparse, float64)")
    _print_common(args, device)

    print("Building datasets (Stokes reference solve; may take a moment)...")
    train_ds, val_ds, test_ds = create_stokes_datasets(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        K=args.k, grid_resolution=args.grid_resolution, mu=args.stokes_mu,
        r=args.stokes_r, ref_chunk=args.stokes_ref_chunk,
        normalize=not args.stokes_no_normalize, seed=args.seed,
    )
    print(f"  train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, "
          f"velocity grid={train_ds.grid_size}, pressure grid={train_ds.pgrid_size}\n")

    # f = (f_x, f_y) in; (u_x, u_y, p) out — pressure is read on the [::2, ::2] subgrid.
    # No train_ds here: StokesTrainer._predict already feeds every model f / f_scale, so a
    # DeepONet's own branch scale must stay 1.
    model = _build_model(args, in_channels=2, out_channels=3)
    common = dict(
        model=model,
        train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
        loss_type=args.loss,
        optimizer_name=args.optimizer,
        lr=args.lr, lr_min=args.lr_min, weight_decay=args.weight_decay,
        batch_size=args.batch_size, epochs=args.epochs,
        device=device, output_dir=args.output_dir,
        mg_levels=args.mg_levels, mg_pre_smooth=args.mg_pre_smooth,
        mg_post_smooth=args.mg_post_smooth, mg_omega=args.mg_omega,
        schur_omega=args.schur_omega, precond_strength=args.stokes_precond_strength,
        precond_kind=args.stokes_precond, pls_form=args.stokes_pls_form,
        uzawa_pre=args.uzawa_pre, uzawa_post=args.uzawa_post,
        cheb_degree=args.cheb_degree, cheb_ratio=args.cheb_ratio,
    )
    # The baseline trainers subclass StokesTrainer, so evaluation (both fields, projections,
    # Q1 pressure subgrid) and model selection are the inherited ones for every arm.
    if args.loss == "pino":
        trainer = PINOStokesTrainer(pino_reduction=args.pino_reduction,
                                    div_weight=args.pi_div_weight, **common)
    elif args.loss == "pideeponet":
        zero_bc = (args.pideeponet_bc == "zero")
        trainer = PIDeepONetStokesTrainer(n_colloc=args.pi_n_colloc,
                                          pi_reduction=args.pino_reduction,
                                          pi_bc=args.pideeponet_bc,
                                          lambda_bc_pi=(args.pi_lambda_bc if zero_bc else 0.0),
                                          div_weight=args.pi_div_weight, **common)
    elif args.model in ("deeponet", "gaot"):
        trainer = (DeepONetStokesTrainer if args.model == "deeponet"
                   else GAOTStokesTrainer)(**common)
    else:
        trainer = StokesTrainer(**common)
    _run(trainer, train_ds, test_ds, args)


def _is_baseline_loss(loss: str) -> bool:
    return loss in ("pino", "pideeponet")


def _apply_hard_bc(model, args):
    """Wrap a grid-valued model so its output satisfies the zero Dirichlet BC exactly.

    Both wrappers act on the emitted ``[B, C, H, W]`` grid alone, so they apply unchanged to
    any architecture wearing that signature -- FNO and GAOT alike.
    """
    nx = ny = args.grid_resolution
    if args.pino_bc == "mollifier":
        print("  mollified: output x sin(pi x)sin(pi y)  (hard zero Dirichlet BC)")
        return MollifiedModel(model, nx, ny)
    # Default. The same operation losses.py applies to our own arms, so the PINO
    # comparison isolates the residual instead of also varying the BC treatment.
    print("  zero-BC: boundary nodes set to 0  (hard zero Dirichlet BC, as our arms)")
    return ZeroBoundaryModel(model, nx, ny)


def _build_model(args, in_channels: int, out_channels: int = 1, train_ds=None):
    print("Building model...")
    # Hard zero-Dirichlet BC as part of the model. On by default for the baselines only: it
    # removes a boundary-penalty weight from their hyperparameters rather than adding one.
    # The *mechanism* differs by arm -- see --pino_bc and the FNO branch below.
    mollify = (args.mollify == "on") or (args.mollify == "auto" and _is_baseline_loss(args.loss))

    if args.model == "deeponet":
        nx = ny = args.grid_resolution
        # Branch inputs are standardized like StokesDataset.f_scale already does: an MLP on a
        # raw O(1e2) field trains badly, while the FNO's lifting layer absorbs the scale itself.
        f_scale = 1.0
        if train_ds is not None and getattr(train_ds, "fs", None):
            f_scale = float(torch.stack(list(train_ds.fs)).abs().mean().clamp_min(1e-8))
        # --pideeponet_bc zero swaps the mollifier for the boundary-node zeroing our arms use
        # (plus the soft penalty in the loss); see PIDeepONetPoissonLoss.
        zero_bc = (args.loss == "pideeponet" and args.pideeponet_bc == "zero")
        mollify_don = mollify and not zero_bc
        # Stokes: the hard BC (either mechanism) acts on the velocity pair only; the pressure
        # has a gauge, not a boundary condition, and the trainer fixes it by projection.
        bc_channels = (0, 1) if args.pde == "stokes" else None
        cfg = dict(grid_size=(nx, ny), in_channels=in_channels, p=args.deeponet_p,
                   width=args.deeponet_width, depth=args.deeponet_depth,
                   trunk_fourier=args.trunk_fourier, fourier_scale=args.trunk_fourier_scale,
                   f_scale=f_scale, mollify=mollify_don, zero_boundary=zero_bc,
                   out_channels=out_channels, bc_channels=bc_channels, seed=args.seed)
        model = DeepONetModel(**cfg)
        print(f"  DeepONet  p={args.deeponet_p} width={args.deeponet_width} "
              f"depth={args.deeponet_depth} fourier={args.trunk_fourier}"
              f"(scale {args.trunk_fourier_scale})  f_scale={f_scale:.3g}"
              f"{'  mollified' if mollify_don else ''}"
              f"{f'  zero-BC + boundary penalty (lambda={args.pi_lambda_bc:g})' if zero_bc else ''}")
    elif args.model == "gaot":
        nx = ny = args.grid_resolution
        # GAOT consumes a point cloud; GAOTModel wraps it in the FNO's grid signature and builds
        # the node coordinates in the repo's row-major order, so grid_to_node on its output is
        # the field the FEM losses expect. An unstructured mesh means passing coords= here
        # instead and calling forward_nodes -- nothing downstream changes.
        cfg = dict(grid_size=(nx, ny), in_channels=in_channels, out_channels=out_channels,
                   latent_grid=tuple(args.gaot_latent), patch_size=args.gaot_patch,
                   radius=args.gaot_radius, radius_scale=args.gaot_radius_scale,
                   scales=tuple(args.gaot_scales), lifting_channels=args.gaot_lifting,
                   magno_hidden=args.gaot_magno_hidden, mlp_layers=args.gaot_mlp_layers,
                   transformer_hidden=args.gaot_hidden, transformer_layers=args.gaot_layers,
                   num_heads=args.gaot_heads, use_geoembed=not args.gaot_no_geoembed,
                   attention_type=args.gaot_attention,
                   node_embedding=args.gaot_node_embedding,
                   positional_embedding=args.gaot_pos_embedding)
        model = GAOTModel(**cfg)
        Hl, Wl = cfg["latent_grid"]
        P = cfg["patch_size"]
        print(f"  GAOT  latent={Hl}x{Wl} patch={P} -> {(Hl // P) * (Wl // P)} tokens  "
              f"lifting={args.gaot_lifting}  transformer={args.gaot_hidden}x{args.gaot_layers}"
              f"({args.gaot_heads} heads)  radius={model.radius:.4f}  "
              f"scales={list(cfg['scales'])}  attn={args.gaot_attention}"
              f"{'' if cfg['use_geoembed'] else '  (no geoembed)'}")
        if mollify and args.pde != "stokes":
            model = _apply_hard_bc(model, args)
    else:
        cfg = dict(n_modes=tuple(args.n_modes), hidden_channels=args.hidden_dim,
                   in_channels=in_channels, out_channels=out_channels, n_layers=args.num_layers)
        model = FNOModel(**cfg)
        # Stokes PINO needs no wrapper: PINOStokesTrainer zeroes the velocity boundary ring
        # itself (the pressure channel must stay free), exactly as _predict projects it.
        if mollify and args.pde != "stokes":
            model = _apply_hard_bc(model, args)
    # Stash the constructor config so a checkpoint can be reloaded without re-guessing it later
    # (e.g. the long_rollout analysis rebuilds the model from this).
    model.build_config = cfg
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {n_params:,}\n")
    return model


def _print_common(args, device):
    print(f"K           : {args.k}")
    print(f"labels      : {args.dataset_solution}"
          + ("  (Q1 FEM solve of A u = M f; train+val+test)"
             if args.dataset_solution == "fem" else "  (closed form at nodes)"))
    print(f"samples     : train={args.n_train}, val={args.n_val}, test={args.n_test}")
    print(f"epochs/bs   : {args.epochs} / {args.batch_size}")
    print(f"lr          : {args.lr}  ->  {args.lr_min}")
    print(f"optimizer   : {args.optimizer}  (weight_decay={args.weight_decay})")
    if args.model == "gaot":
        print(f"GAOT        : latent={tuple(args.gaot_latent)} patch={args.gaot_patch}   "
              f"lifting={args.gaot_lifting}   transformer={args.gaot_hidden}x{args.gaot_layers}"
              f"   grid={args.grid_resolution}^2")
    elif args.model == "deeponet":
        print(f"DeepONet    : p={args.deeponet_p}   width={args.deeponet_width}   "
              f"depth={args.deeponet_depth}   grid={args.grid_resolution}^2")
    else:
        print(f"FNO modes   : {tuple(args.n_modes)}   hidden={args.hidden_dim}   "
              f"layers={args.num_layers}   grid={args.grid_resolution}^2")
    print(f"device      : {device}")
    print("=" * 60 + "\n")


def _run(trainer, train_ds, test_ds, args):
    trainer.save_checkpoints = not args.no_checkpoint
    if args.eval_only:
        if not trainer.load_checkpoint(args.checkpoint):
            return
        _report_test(trainer.test(), args.pde)
        for idx in args.sample_idx:
            if idx < len(train_ds):
                trainer.visualize_sample(train_ds, "train", idx)
            if idx < len(test_ds):
                trainer.visualize_sample(test_ds, "test", idx)
        trainer.compute_error_distribution()
    else:
        result = trainer.train()
        print("\n" + "=" * 60)
        print("Done.")
        print(f"  best val       : {trainer.stats.best_val_error:.2e}")
        _report_test(result, args.pde)
        print("=" * 60)


def _report_test(result, pde: str = "poisson"):
    """Poisson trainers return (mse, l2, rel_l2); rollout trainers a rollout MSE; Stokes the
    mean of the velocity and pressure relative FE-L2 errors (the per-field numbers are
    printed by the trainer) — a squared error would be dominated by pressure, whose scale
    is set by the physics and cannot be normalized independently of velocity."""
    if isinstance(result, tuple):
        mse, l2, rl2 = result
        print(f"  final test     : MSE {mse:.2e}  FEM-L2 {l2:.2e}  rel-L2 {rl2:.2%}")
    elif pde == "stokes":
        print(f"  final test     : combined rel-L2 (velocity+pressure)/2 = {result:.2%}")
    else:
        print(f"  final test     : rollout MSE {result:.2e}")


def _validate_baseline_args(args):
    """Fail fast and legibly on baseline flag combinations that cannot work."""
    if _is_baseline_loss(args.loss) and args.pde not in ("poisson", "ac", "stokes"):
        raise SystemExit(f"--loss {args.loss} is implemented for --pde poisson|ac|stokes only "
                         f"(got {args.pde!r}); see experiments/baselines/README.md.")
    if args.loss == "pideeponet" and args.model != "deeponet":
        raise SystemExit("--loss pideeponet differentiates through the trunk's coordinate "
                         "input, so it needs --model deeponet.")
    if args.model == "deeponet" and args.pde == "wave":
        raise SystemExit("--model deeponet is wired for --pde poisson|ac|stokes only (got 'wave').")
    if (args.model == "deeponet" and args.mollify == "off" and args.loss == "pideeponet"
            and args.pideeponet_bc != "zero"):
        raise SystemExit("--loss pideeponet with --mollify off has no boundary condition at all "
                         "(the autodiff residual cannot see the boundary). Use "
                         "--pideeponet_bc zero (boundary nodes zeroed + soft penalty, the "
                         "counterpart of --pino_bc zero) or keep the mollifier.")
    if args.loss == "pideeponet" and args.pideeponet_bc == "zero" and args.pde not in ("poisson", "stokes"):
        raise SystemExit("--pideeponet_bc zero is implemented for --pde poisson|stokes only; the "
                         "Allen-Cahn rollout needs the residual and fed-back values to coincide "
                         "on the boundary, which the mollifier guarantees.")


def main():
    args = build_parser().parse_args()
    _validate_baseline_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    print("=" * 60)
    print(f"TensorPILS — {args.model.upper()} + {args.pde.capitalize()} 2D")
    print("=" * 60)

    if args.pde == "wave":
        run_wave(args, device)
    elif args.pde == "ac":
        run_ac(args, device)
    elif args.pde == "stokes":
        run_stokes(args, device)
    else:
        run_poisson(args, device)


if __name__ == "__main__":
    main()
