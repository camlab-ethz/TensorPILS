"""Command-line entry point: ``tensorpils`` (or ``python -m tensorpils.cli``).

Trains a neural operator on one of the three problems of the paper (``--pde``):

* ``poisson`` — static :math:`-\\Delta u = f` on the unit square (FNO or GAOT), with the
  supervised ``data`` loss, the least-squares loss ``galerkin`` (``L_LS``) or the preconditioned
  least-squares loss ``pls`` (``L_PLS``; geometric multigrid, or the exact spectral blend).
* ``ac`` — the Allen–Cahn convex–concave time stepper (FNO, autoregressive rollout), with the
  ``data`` loss or the ``galerkin`` step residual, optionally preconditioned
  (``--ac_precond multigrid``).
* ``stokes`` — stationary Stokes past an obstacle, Taylor-Hood P2/P1 on an unstructured mesh
  (GAOT), with ``data``, ``galerkin`` or ``pls`` (block preconditioner with an algebraic V-cycle).

The baselines are selected with ``--loss pino`` (Poisson, Allen–Cahn) and ``--loss pideeponet``
(all three problems, with ``--model deeponet``).
"""

from argparse import ArgumentParser

import numpy as np
import torch

from .data import create_datasets, create_scaling_datasets, create_ac_datasets, create_stokes_datasets
from .models import FNOModel
from .gaot import GAOTModel
from .trainer import (PoissonTrainer, ACTrainer, StokesTrainer, GAOTPoissonTrainer,
                      GAOTStokesTrainer)
from .baselines import (DeepONetModel, MollifiedModel, ZeroBoundaryModel,
                        PINOPoissonTrainer, PINOACTrainer,
                        PIDeepONetPoissonTrainer, PIDeepONetACTrainer, PIDeepONetStokesTrainer)


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Preconditioned physics-informed neural operator training "
                                   "(Poisson / Allen-Cahn / Stokes).")
    p.add_argument("--pde", choices=["poisson", "ac", "stokes"], default="poisson",
                   help="Which problem to train on.")
    p.add_argument("--loss", choices=["data", "galerkin", "pls", "pino", "pideeponet"],
                   default="galerkin",
                   help="'data': supervised. 'galerkin': FEM least-squares residual (L_LS). "
                        "'pls': preconditioned least squares (L_PLS; for --pde ac use "
                        "--loss galerkin --ac_precond multigrid). BASELINES: 'pino' = strong-form "
                        "residual by finite differences (poisson/ac); 'pideeponet' = the strong "
                        "form by autodiff through the trunk, requires --model deeponet.")
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
    p.add_argument("--dataset_solution", choices=["analytic", "fem"], default="analytic",
                   help="Poisson labels: 'analytic' (default) samples the closed-form solution "
                        "at the nodes; 'fem' solves A u = M f with Q1 on the regular grid, i.e. "
                        "labels become the discrete solution the FEM losses target. Applies to "
                        "train, val AND test.")
    p.add_argument("--fixed_eval", action="store_true",
                   help="Poisson: draw val/test from dedicated seeds (seed+1/seed+2), identical "
                        "for every --n_train — all runs of a dataset-size sweep share the same "
                        "eval sets. Default: the usual joint train/val/test pool.")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--optimizer", type=str, default="adam", help="adam | adamw | sgd")

    # -------- Preconditioner (poisson pls) --------
    p.add_argument("--precond_kind", choices=["multigrid", "blend"], default="multigrid",
                   help="Preconditioner P≈A^-1 for --loss pls. 'multigrid' (default): one "
                        "geometric-multigrid V-cycle. 'blend': the exact spectral blend "
                        "(1-t)I + tA^-1 of the conditioning study (dense eigendecomposition).")
    p.add_argument("--precond_strength", type=float, default=1.0,
                   help="Blend parameter t in [0,1] for --precond_kind blend. 0 -> P=I "
                        "(no preconditioning); 1 -> P=A^-1 (supervised).")

    # -------- Geometric multigrid V-cycle (poisson pls, ac --ac_precond multigrid) --------
    p.add_argument("--mg_levels", type=int, default=4,
                   help="Number of multigrid levels (incl. finest).")
    p.add_argument("--mg_pre_smooth", type=int, default=2,
                   help="Pre-smoothing Jacobi sweeps per V-cycle level.")
    p.add_argument("--mg_post_smooth", type=int, default=2,
                   help="Post-smoothing Jacobi sweeps per V-cycle level.")
    p.add_argument("--mg_omega", type=float, default=2.0 / 3.0,
                   help="Damping factor for the weighted Jacobi smoother (8/9 is optimal for Q1 "
                        "in 2D).")

    # -------- Allen–Cahn (time-dependent, nonlinear) --------
    p.add_argument("--dt", type=float, default=0.005, help="Time step tau.")
    p.add_argument("--n_steps", type=int, default=20,
                   help="Number of reference time steps generated per sample.")
    p.add_argument("--rollout_steps", type=int, default=4,
                   help="Autoregressive rollout length used for the loss and eval.")
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
    p.add_argument("--ac_precond", choices=["none", "multigrid"], default="none",
                   help="Precondition the Allen–Cahn least-squares residual: 'multigrid' trains "
                        "½‖P R‖² with P ≈ J0^-1, J0 = a²A + cM the frozen (u²=1) Newton Jacobian "
                        "(c = 1/dt + 3ε²). Uses the --mg_* V-cycle settings. Default: none.")

    # -------- Stokes (Taylor-Hood P2/P1 on the obstacle mesh) --------
    p.add_argument("--stokes_mu", type=float, default=1.0,
                   help="Stokes: dynamic viscosity mu.")
    p.add_argument("--stokes_r", type=float, default=-0.5,
                   help="Stokes: spectral decay exponent of the random body force "
                        "(amplitude ~ (k^2+l^2)^r), matching the Poisson source convention.")
    p.add_argument("--stokes_ref_chunk", type=int, default=64,
                   help="Stokes: right-hand sides per batched reference solve (memory control).")
    p.add_argument("--schur_omega", type=float, default=0.5,
                   help="Stokes pls: weight omega of the lumped-pressure-mass Schur surrogate "
                        "S^-1 = omega*mu*diag(M_p)^-1 in the block preconditioner. As a norm "
                        "weight it is a free parameter; the paper selects 16 on validation.")
    p.add_argument("--amg_sweeps", type=int, default=2,
                   help="Stokes pls: pre- and post-smoothing sweeps of the algebraic V-cycle on "
                        "the velocity block (equal, so the cycle is symmetric).")
    p.add_argument("--mesh_h", type=float, default=0.035,
                   help="Stokes: Gmsh target edge length of the obstacle mesh. 0.035 gives the "
                        "paper's mesh (3998 P2 / 1035 P1 nodes).")
    p.add_argument("--obstacle_center", type=float, nargs=2, default=[0.40, 0.50],
                   help="Stokes: centre of the circular hole. Off-centre by default -- a centred "
                        "hole makes the solution inherit the domain's symmetry.")
    p.add_argument("--obstacle_radius", type=float, default=0.14,
                   help="Stokes: radius of the circular hole.")
    p.add_argument("--mesh_cache_dir", type=str, default=None,
                   help="Stokes: directory to cache the generated mesh in. The file name encodes "
                        "every geometric parameter, so a cached mesh is only reused for the exact "
                        "geometry it was built for.")

    # -------- FNO model --------
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--n_modes", type=int, nargs=2, default=[16, 16])
    p.add_argument("--grid_resolution", type=int, default=64,
                   help="Poisson / Allen–Cahn: nodes per side of the structured grid.")

    # -------- architecture --------
    p.add_argument("--model", choices=["fno", "deeponet", "gaot"], default="fno",
                   help="Neural-operator architecture. 'fno' (Poisson, Allen–Cahn); 'gaot', the "
                        "geometry-aware operator transformer (Poisson, and Stokes, which has no "
                        "grid); 'deeponet' carries the PI-DeepONet baseline.")
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
                        "independently of how the physical points are arranged.")
    p.add_argument("--gaot_patch", type=int, default=2,
                   help="GAOT: patch side on the latent grid; (H/P)*(W/P) transformer tokens.")
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
                   help="GAOT: disable the geometric embedding of each neighbourhood.")
    p.add_argument("--gaot_attention", choices=["cosine", "dot_product"], default="cosine",
                   help="GAOT: neighbour weighting inside the AGNO kernel integral.")
    p.add_argument("--gaot_node_embedding", action="store_true",
                   help="GAOT: sinusoidal encoding of coordinates before the kernel MLP.")
    p.add_argument("--gaot_pos_embedding", choices=["absolute", "rope"], default="absolute",
                   help="GAOT: transformer positional embedding ('rope' needs "
                        "rotary-embedding-torch).")

    # -------- baselines --------
    p.add_argument("--mollify", choices=["auto", "on", "off"], default="auto",
                   help="Impose the zero Dirichlet BC hard as part of the model. 'auto' turns it "
                        "on for the baseline losses (pino/pideeponet) and off otherwise. HOW it is "
                        "imposed depends on the arm: PI-DeepONet multiplies by sin(pi x)sin(pi y) "
                        "(the reference's mollifier) unless --pideeponet_bc zero; PINO zeroes the "
                        "boundary nodes -- see --pino_bc.")
    p.add_argument("--pino_bc", choices=["zero", "mollifier"], default="zero",
                   help="PINO only: how the hard zero Dirichlet BC is imposed. 'zero' (default) "
                        "zeroes the boundary nodes, exactly as our own arms do, so PINO differs "
                        "from the galerkin arm only in the residual. 'mollifier' multiplies by "
                        "sin(pi x)sin(pi y), reproducing the reference implementation.")
    p.add_argument("--pideeponet_bc", choices=["mollifier", "zero"], default="mollifier",
                   help="pideeponet only: how the zero Dirichlet BC enters. 'mollifier' (default) "
                        "multiplies every query by sin(pi x)sin(pi y), as the reference does. "
                        "'zero' sets the grid output's boundary nodes to 0 as our arms do and adds "
                        "the original PI-DeepONet soft boundary penalty, weighted by "
                        "--pi_lambda_bc (required for --pde stokes).")
    p.add_argument("--pi_lambda_bc", type=float, default=1.0,
                   help="pideeponet with --pideeponet_bc zero: weight of the boundary penalty, "
                        "expressed in the residual's own units.")
    p.add_argument("--pi_div_weight", type=float, default=1.0,
                   help="pideeponet (stokes): weight of the continuity residual div(u) relative "
                        "to the momentum residual in the stacked strong-form loss. The two "
                        "equations have different units, so this is tuned on validation.")
    p.add_argument("--pi_n_colloc", type=int, default=0,
                   help="pideeponet (poisson, stokes): collocation points per step, drawn afresh "
                        "from the interior nodes each step. 0 (default) uses all of them.")
    p.add_argument("--pino_reduction", choices=["rel", "mse"], default="rel",
                   help="pino/pideeponet (poisson, stokes): 'rel' is the reference "
                        "implementation's relative-Lp ratio ||Lu-f||/||f||; 'mse' is the plain "
                        "mean square.")

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
    print(f"loss        : {args.loss}")
    if args.loss == "pls":
        if args.precond_kind == "multigrid":
            print(f"precond     : multigrid  levels={args.mg_levels}  "
                  f"smooth={args.mg_pre_smooth}/{args.mg_post_smooth}  omega={args.mg_omega:.3f}")
        else:
            print(f"precond     : {args.precond_kind}  strength={args.precond_strength:.3f}")
    if args.stream and args.steps_per_epoch is None:
        raise SystemExit("--stream requires --steps_per_epoch (an epoch has no natural "
                         "length on an infinite stream).")
    if args.stream or args.steps_per_epoch or args.fixed_eval:
        print(f"data mode   : {'stream (fresh samples every step)' if args.stream else 'finite'}"
              + (f"  steps/epoch={args.steps_per_epoch}" if args.steps_per_epoch else "")
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

    model = _build_model(args, in_channels=1, train_ds=train_ds)
    common = dict(
        model=model,
        train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
        loss_type=args.loss,
        optimizer_name=args.optimizer,
        lr=args.lr, lr_min=args.lr_min, weight_decay=args.weight_decay,
        batch_size=args.batch_size, epochs=args.epochs,
        device=device, output_dir=args.output_dir,
        precond_kind=args.precond_kind, precond_strength=args.precond_strength,
        mg_levels=args.mg_levels, mg_pre_smooth=args.mg_pre_smooth,
        mg_post_smooth=args.mg_post_smooth, mg_omega=args.mg_omega,
        steps_per_epoch=args.steps_per_epoch,
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
    elif args.model == "gaot":
        # Same losses, other architecture. The arch-tagged prefix keeps these runs from
        # landing on (and overwriting) the identically-configured FNO run's files.
        trainer = GAOTPoissonTrainer(**common)
    else:
        trainer = PoissonTrainer(**common)
    _run(trainer, train_ds, test_ds, args)


def run_ac(args, device):
    # --loss is a preset for the (physics, data) weights of the rollout loss. The baselines are
    # physics-only, so they sit on the physics side of the preset.
    lg = 0.0 if args.loss == "data" else 1.0
    ld = 1.0 if args.loss == "data" else 0.0

    total = args.n_train + max(args.n_val, 0) + max(args.n_test, 0)
    print(f"loss        : {args.loss}  (lambda_physics={lg}, lambda_data={ld})")
    print(f"allen-cahn  : a={args.ac_a}  eps={args.ac_eps}  r={args.ac_r}  dt={args.dt}  "
          f"n_steps={args.n_steps}  rollout={args.rollout_steps}  (convex-concave, pushforward)")
    if args.ac_precond != "none":
        c_shift = 1.0 / args.dt + 3.0 * args.ac_eps ** 2
        print(f"precond     : {args.ac_precond}  ½‖P R‖², P≈(a²A+cM)⁻¹  "
              f"(a²={args.ac_a ** 2:g}, c=1/dt+3ε²={c_shift:g}, mg_levels={args.mg_levels})")
    print(f"reference   : FEM convex-concave + Newton on {args.ac_ref_device} "
          f"(build cost ~ {total}x{args.n_steps} steps; "
          f"no analytical solution exists for AC)")
    _print_common(args, device)

    print("Building datasets (FEM reference solve; may take a moment)...")
    train_ds, val_ds, test_ds = create_ac_datasets(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        K=args.k, grid_resolution=args.grid_resolution,
        dt=args.dt, n_steps=args.n_steps, a=args.ac_a, eps=args.ac_eps, r=args.ac_r,
        newton_tol=args.ac_newton_tol, newton_max=args.ac_newton_max,
        ref_chunk=args.ac_ref_chunk, seed=args.seed, ref_device=args.ac_ref_device,
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
        rollout_steps=args.rollout_steps,
        ac_precond=("" if args.ac_precond == "none" else args.ac_precond),
        mg_levels=args.mg_levels, mg_pre_smooth=args.mg_pre_smooth,
        mg_post_smooth=args.mg_post_smooth, mg_omega=args.mg_omega,
    )
    # Both baselines inherit ACTrainer's rollout evaluation, so their reported per-step /
    # space-time relative L2 is produced by exactly the same code as the FEM arms.
    cls = {"pino": PINOACTrainer, "pideeponet": PIDeepONetACTrainer}.get(args.loss, ACTrainer)
    trainer = cls(**common)
    _run(trainer, train_ds, test_ds, args)


def run_stokes(args, device):
    """Stokes past an obstacle: Taylor-Hood P2/P1 on an unstructured triangulation."""
    print(f"loss        : {args.loss}")
    print(f"domain      : unit square with a circular hole at "
          f"({args.obstacle_center[0]:g}, {args.obstacle_center[1]:g}) r={args.obstacle_radius:g}; "
          f"unstructured P2/P1, chara_length={args.mesh_h}")
    print(f"stokes      : mu={args.stokes_mu}  force decay r={args.stokes_r}")
    if args.loss == "pls":
        print(f"precond     : block P = diag(A^-1/mu, w*mu/diag(M_p)), velocity half = algebraic "
              f"V-cycle ({args.amg_sweeps}/{args.amg_sweeps} sweeps), schur_omega={args.schur_omega:g}")
    print("reference   : discrete Taylor-Hood solve of K c = (M_u f, 0) (batched sparse, float64)")
    _print_common(args, device)

    print("Building datasets (Gmsh P2 mesh + one sparse factorisation for the references)...")
    train_ds, val_ds, test_ds = create_stokes_datasets(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test, K=args.k,
        chara_length=args.mesh_h, mu=args.stokes_mu, r=args.stokes_r,
        ref_chunk=args.stokes_ref_chunk, cx=args.obstacle_center[0],
        cy=args.obstacle_center[1], radius=args.obstacle_radius, seed=args.seed,
        cache_dir=args.mesh_cache_dir,
    )
    print(f"  train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, "
          f"velocity(P2) nodes={train_ds.n_u}, pressure(P1) nodes={train_ds.n_p}\n")

    # f = (f_x, f_y) in, (u_x, u_y, p) out -- all three on the P2 node set; the pressure
    # channel is gathered at the corner nodes.
    model = _build_model(args, in_channels=2, out_channels=3,
                         coords=train_ds.mesh.points.float(), train_ds=train_ds)
    common = dict(
        model=model,
        train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
        loss_type=args.loss,
        optimizer_name=args.optimizer,
        lr=args.lr, lr_min=args.lr_min, weight_decay=args.weight_decay,
        batch_size=args.batch_size, epochs=args.epochs,
        device=device, output_dir=args.output_dir,
        amg_sweeps=args.amg_sweeps, schur_omega=args.schur_omega,
    )
    if args.loss == "pideeponet":
        trainer = PIDeepONetStokesTrainer(n_colloc=args.pi_n_colloc,
                                          pi_reduction=args.pino_reduction,
                                          lambda_bc_pi=args.pi_lambda_bc,
                                          div_weight=args.pi_div_weight, **common)
    else:
        trainer = GAOTStokesTrainer(**common)
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


def _build_model(args, in_channels: int, out_channels: int = 1, train_ds=None,
                 coords=None):
    print("Building model...")
    # Hard zero-Dirichlet BC as part of the model. On by default for the baselines only: it
    # removes a boundary-penalty weight from their hyperparameters rather than adding one.
    # The *mechanism* differs by arm -- see --pino_bc and the FNO branch below.
    mollify = (args.mollify == "on") or (args.mollify == "auto" and _is_baseline_loss(args.loss))

    if args.model == "deeponet":
        nx = ny = args.grid_resolution
        # Branch inputs are standardized: an MLP on a raw O(1e2) field trains badly, while the
        # FNO's lifting layer absorbs the scale itself.
        f_scale = 1.0
        if train_ds is not None and getattr(train_ds, "fs", None):
            f_scale = float(torch.stack(list(train_ds.fs)).abs().mean().clamp_min(1e-8))
        # --pideeponet_bc zero swaps the mollifier for the boundary-node zeroing our arms use
        # (plus the soft penalty in the loss); see PIDeepONetPoissonLoss.
        zero_bc = (args.loss == "pideeponet" and args.pideeponet_bc == "zero")
        mollify_don = mollify and not zero_bc
        # An unstructured sensor set (Stokes): the branch is an MLP over flattened sensor values
        # and never knew they were on a grid, and the trunk was always coordinate-based. The
        # hard-BC mechanisms are grid-shaped, so on a mesh the BC is the loss's boundary penalty.
        unstructured = coords is not None
        cfg = dict(grid_size=(nx, ny), in_channels=in_channels, p=args.deeponet_p,
                   width=args.deeponet_width, depth=args.deeponet_depth,
                   trunk_fourier=args.trunk_fourier, fourier_scale=args.trunk_fourier_scale,
                   f_scale=f_scale,
                   mollify=False if unstructured else mollify_don,
                   zero_boundary=False if unstructured else zero_bc,
                   out_channels=out_channels, seed=args.seed,
                   coords=coords)
        model = DeepONetModel(**cfg)
        print(f"  DeepONet  p={args.deeponet_p} width={args.deeponet_width} "
              f"depth={args.deeponet_depth} fourier={args.trunk_fourier}"
              f"(scale {args.trunk_fourier_scale})  f_scale={f_scale:.3g}"
              f"{'  mollified' if mollify_don and not unstructured else ''}"
              f"{f'  zero-BC + boundary penalty (lambda={args.pi_lambda_bc:g})' if zero_bc else ''}")
    elif args.model == "gaot":
        nx = ny = args.grid_resolution
        # GAOT consumes a point cloud; GAOTModel wraps it in the FNO's grid signature and builds
        # the node coordinates in the repo's row-major order, so grid_to_node on its output is
        # the field the FEM losses expect. On the obstacle mesh, coords= is the mesh's own points
        # and the trainer calls forward_nodes -- nothing downstream changes.
        cfg = dict(grid_size=(nx, ny), in_channels=in_channels, out_channels=out_channels,
                   latent_grid=tuple(args.gaot_latent), patch_size=args.gaot_patch,
                   radius=args.gaot_radius, radius_scale=args.gaot_radius_scale,
                   scales=tuple(args.gaot_scales), lifting_channels=args.gaot_lifting,
                   magno_hidden=args.gaot_magno_hidden, mlp_layers=args.gaot_mlp_layers,
                   transformer_hidden=args.gaot_hidden, transformer_layers=args.gaot_layers,
                   num_heads=args.gaot_heads, use_geoembed=not args.gaot_no_geoembed,
                   attention_type=args.gaot_attention,
                   node_embedding=args.gaot_node_embedding,
                   positional_embedding=args.gaot_pos_embedding,
                   coords=coords)
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
        if mollify:
            model = _apply_hard_bc(model, args)
    # Stash the constructor config so a checkpoint can be reloaded without re-guessing it later.
    model.build_config = cfg
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {n_params:,}\n")
    return model


def _print_common(args, device):
    print(f"K           : {args.k}")
    if args.pde == "poisson":
        print(f"labels      : {args.dataset_solution}"
              + ("  (Q1 FEM solve of A u = M f; train+val+test)"
                 if args.dataset_solution == "fem" else "  (closed form at nodes)"))
    print(f"samples     : train={args.n_train}, val={args.n_val}, test={args.n_test}")
    print(f"epochs/bs   : {args.epochs} / {args.batch_size}")
    print(f"lr          : {args.lr}  ->  {args.lr_min}")
    print(f"optimizer   : {args.optimizer}  (weight_decay={args.weight_decay})")
    grid = "" if args.pde == "stokes" else f"   grid={args.grid_resolution}^2"
    if args.model == "gaot":
        print(f"GAOT        : latent={tuple(args.gaot_latent)} patch={args.gaot_patch}   "
              f"lifting={args.gaot_lifting}   transformer={args.gaot_hidden}x{args.gaot_layers}"
              f"{grid}")
    elif args.model == "deeponet":
        print(f"DeepONet    : p={args.deeponet_p}   width={args.deeponet_width}   "
              f"depth={args.deeponet_depth}{grid}")
    else:
        print(f"FNO modes   : {tuple(args.n_modes)}   hidden={args.hidden_dim}   "
              f"layers={args.num_layers}{grid}")
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
    """Poisson trainers return (mse, l2, rel_l2); the Allen–Cahn trainer a rollout MSE; Stokes
    the mean of the velocity and pressure relative FE-L2 errors (the per-field numbers are
    printed by the trainer)."""
    if isinstance(result, tuple):
        mse, l2, rl2 = result
        print(f"  final test     : MSE {mse:.2e}  FEM-L2 {l2:.2e}  rel-L2 {rl2:.2%}")
    elif pde == "stokes":
        print(f"  final test     : combined rel-L2 (velocity+pressure)/2 = {result:.2%}")
    else:
        print(f"  final test     : rollout MSE {result:.2e}")


def _validate_args(args):
    """Refuse the combinations that are not implemented, before any data is built."""
    if args.loss == "pideeponet" and args.model != "deeponet":
        raise SystemExit("--loss pideeponet differentiates through the trunk's coordinate "
                         "input, so it needs --model deeponet.")
    if args.model == "deeponet" and args.loss != "pideeponet":
        raise SystemExit("--model deeponet carries the PI-DeepONet baseline; use it with "
                         "--loss pideeponet.")
    if args.loss == "pideeponet" and args.mollify == "off" and args.pideeponet_bc != "zero":
        raise SystemExit("--loss pideeponet with --mollify off has no boundary condition at all "
                         "(the autodiff residual cannot see the boundary). Use "
                         "--pideeponet_bc zero (boundary nodes zeroed + soft penalty) or keep "
                         "the mollifier.")
    if args.pde == "poisson":
        if args.model == "gaot" and _is_baseline_loss(args.loss):
            raise SystemExit("--model gaot is run with --loss data|galerkin|pls.")
    elif args.pde == "ac":
        if args.loss == "pls":
            raise SystemExit("--pde ac: the preconditioned loss is "
                             "--loss galerkin --ac_precond multigrid.")
        if args.model == "gaot":
            raise SystemExit("--pde ac is run with --model fno (or deeponet for pideeponet).")
        if args.loss == "pideeponet" and args.pideeponet_bc == "zero":
            raise SystemExit("--pde ac: PI-DeepONet uses the mollifier; the rollout needs the "
                             "residual and fed-back values to coincide on the boundary.")
    elif args.pde == "stokes":
        if args.loss == "pino":
            raise SystemExit("--pde stokes lives on an unstructured mesh; PINO takes finite "
                             "differences on an image and has no counterpart there.")
        if args.loss == "pideeponet" and args.pideeponet_bc != "zero":
            raise SystemExit(
                "--pde stokes with --loss pideeponet requires --pideeponet_bc zero: no "
                "closed-form mollifier vanishes on both boundary components of a domain with a "
                "hole, so the BC is the boundary penalty (--pi_lambda_bc).")
        if args.loss != "pideeponet" and args.model != "gaot":
            raise SystemExit("--pde stokes has no grid, so the model must consume a point cloud: "
                             "use --model gaot.")


def main():
    args = build_parser().parse_args()
    _validate_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    print("=" * 60)
    print(f"TensorPILS — {args.model.upper()} + {args.pde.capitalize()} 2D")
    print("=" * 60)

    if args.pde == "ac":
        run_ac(args, device)
    elif args.pde == "stokes":
        run_stokes(args, device)
    else:
        run_poisson(args, device)


if __name__ == "__main__":
    main()
