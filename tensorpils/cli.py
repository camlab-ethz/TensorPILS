"""Command-line entry point: ``tensorpils`` (or ``python -m tensorpils.cli``).

Trains an FNO to solve one of three PDEs (``--pde``):

* ``poisson`` (static) with one of four losses: ``data`` / ``galerkin`` / ``deepritz`` / ``pls``.
* ``wave`` (time-dependent) as an autoregressive time-stepper with a ``galerkin`` (label-free
  central-difference residual) and/or ``data`` (supervised) loss.
* ``ac`` (Allen–Cahn, time-dependent nonlinear) as an autoregressive time-stepper with a
  ``galerkin`` (label-free backward-Euler residual) and/or ``data`` (supervised) loss.
"""

from argparse import ArgumentParser

import numpy as np
import torch

from .data import (create_datasets, create_scaling_datasets, PoissonDataset,
                   create_wave_datasets, create_ac_datasets)
from .models import FNOModel
from .trainer import PoissonTrainer, WaveTrainer, ACTrainer


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="FNO training for 2D Poisson / wave "
                                   "(data / Galerkin / Deep Ritz / PLS losses).")
    p.add_argument("--pde", choices=["poisson", "wave", "ac"], default="poisson",
                   help="Which PDE to train on. 'poisson' (static), 'wave' or 'ac' (time-dependent).")
    p.add_argument("--loss",
                   choices=["data", "data_l2", "data_h1", "galerkin", "deepritz", "pls"],
                   default="galerkin",
                   help="poisson: data/data_l2/data_h1/galerkin/deepritz/pls. wave/ac: data or "
                        "galerkin (preset for the lambda_galerkin/lambda_data mix).")
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
    p.add_argument("--precond_kind", choices=["multigrid", "blend", "power"],
                   default="multigrid",
                   help="Preconditioner P≈A^-1. 'multigrid' (default): geometric-multigrid "
                        "V-cycle (computational path). 'blend': convex mix (1-t)I+tA^-1. "
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

    # -------- FNO model --------
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--n_modes", type=int, nargs=2, default=[16, 16])
    p.add_argument("--grid_resolution", type=int, default=64)

    # -------- Misc --------
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_dir", type=str, default="output")

    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
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
        )
    else:
        train_ds, val_ds, test_ds = create_datasets(
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
            K=args.k, grid_resolution=args.grid_resolution, seed=args.seed,
        )
    ntr = f"stream({len(train_ds)}/epoch)" if args.stream else len(train_ds)
    print(f"  train={ntr}, val={len(val_ds)}, test={len(test_ds)}, "
          f"grid={train_ds.grid_size}\n")

    # Out-of-distribution eval datasets (higher source complexity K, same grid).
    eval_datasets = {
        f"K{k}": PoissonDataset(num_samples=args.ood_n_val, K=k, seed=args.ood_seed,
                                grid_resolution=args.grid_resolution)
        for k in args.ood_k
    }
    if eval_datasets:
        print(f"  OOD eval sets: {', '.join(eval_datasets)} "
              f"(n={args.ood_n_val} each, seed={args.ood_seed})\n")

    model = _build_model(args, in_channels=1)
    trainer = PoissonTrainer(
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
    trainer = WaveTrainer(
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
    if args.loss not in ("data", "galerkin"):
        raise SystemExit(f"--pde ac supports --loss data|galerkin, got {args.loss!r}")
    # --loss is a preset for the (lambda_galerkin, lambda_data) mix; explicit flags override.
    lg = args.lambda_galerkin
    ld = args.lambda_data
    if lg is None:
        lg = 1.0 if args.loss == "galerkin" else 0.0
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
    print(f"bptt mode   : {args.bptt_mode or 'default (per-loss)'}")
    print(f"reference   : FEM {args.ac_integrator} + Newton (build cost ~ {total}x{args.n_steps} steps; "
          f"no analytical solution exists for AC)")
    _print_common(args, device)

    print("Building datasets (FEM reference solve; may take a moment)...")
    train_ds, val_ds, test_ds = create_ac_datasets(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        K=args.k, grid_resolution=args.grid_resolution,
        dt=args.dt, n_steps=args.n_steps, a=args.ac_a, eps=args.ac_eps, r=args.ac_r,
        newton_tol=args.ac_newton_tol, newton_max=args.ac_newton_max,
        ref_chunk=args.ac_ref_chunk, integrator=args.ac_integrator, seed=args.seed,
    )
    print(f"  train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, "
          f"grid={train_ds.grid_size}, frames/sample={args.n_steps + 1}\n")

    model = _build_model(args, in_channels=1)
    trainer = ACTrainer(
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
    )
    _run(trainer, train_ds, test_ds, args)


def _build_model(args, in_channels: int):
    print("Building model...")
    cfg = dict(n_modes=tuple(args.n_modes), hidden_channels=args.hidden_dim,
               in_channels=in_channels, out_channels=1, n_layers=args.num_layers)
    model = FNOModel(**cfg)
    # Stash the constructor config so a checkpoint can be reloaded without re-guessing it later
    # (e.g. the long_rollout analysis rebuilds the model from this).
    model.build_config = cfg
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {n_params:,}\n")
    return model


def _print_common(args, device):
    print(f"K           : {args.k}")
    print(f"samples     : train={args.n_train}, val={args.n_val}, test={args.n_test}")
    print(f"epochs/bs   : {args.epochs} / {args.batch_size}")
    print(f"lr          : {args.lr}  ->  {args.lr_min}")
    print(f"optimizer   : {args.optimizer}  (weight_decay={args.weight_decay})")
    print(f"FNO modes   : {tuple(args.n_modes)}   hidden={args.hidden_dim}   "
          f"layers={args.num_layers}   grid={args.grid_resolution}^2")
    print(f"device      : {device}")
    print("=" * 60 + "\n")


def _run(trainer, train_ds, test_ds, args):
    if args.eval_only:
        if not trainer.load_checkpoint(args.checkpoint):
            return
        _report_test(trainer.test())
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
        _report_test(result)
        print("=" * 60)


def _report_test(result):
    """Poisson trainers return (mse, l2, rel_l2); rollout trainers return a single rollout MSE."""
    if isinstance(result, tuple):
        mse, l2, rl2 = result
        print(f"  final test     : MSE {mse:.2e}  FEM-L2 {l2:.2e}  rel-L2 {rl2:.2%}")
    else:
        print(f"  final test     : rollout MSE {result:.2e}")


def main():
    args = build_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    print("=" * 60)
    print(f"TensorPILS — FNO + {args.pde.capitalize()} 2D")
    print("=" * 60)

    if args.pde == "wave":
        run_wave(args, device)
    elif args.pde == "ac":
        run_ac(args, device)
    else:
        run_poisson(args, device)


if __name__ == "__main__":
    main()
