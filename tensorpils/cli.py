"""Command-line entry point: ``tensorpils`` (or ``python -m tensorpils.cli``).

Trains an FNO to solve 2D Poisson with one of four losses
(``data`` / ``galerkin`` / ``deepritz`` / ``pls``).
"""

from argparse import ArgumentParser

import numpy as np
import torch

from .data import create_datasets
from .models import FNOModel
from .trainer import Trainer


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="FNO training for 2D Poisson "
                                   "(data / Galerkin / Deep Ritz / PLS losses).")
    p.add_argument("--loss", choices=["data", "galerkin", "deepritz", "pls"], default="galerkin")
    p.add_argument("--n_train", type=int, default=1024)
    p.add_argument("--n_val", type=int, default=128)
    p.add_argument("--n_test", type=int, default=256)
    p.add_argument("-k", "--k", type=int, default=4, help="K x K source complexity")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lambda_bc", type=float, default=100.0,
                   help="BC penalty weight for Deep Ritz when --bc_mode penalty.")
    p.add_argument("--bc_mode", choices=["penalty", "hard"], default="penalty",
                   help="Boundary handling. Deep Ritz: 'penalty' (soft, uses lambda_bc) or "
                        "'hard' (project u->0 before energy). Data loss: 'penalty' (full-grid "
                        "MSE) or 'hard' (interior-only MSE, boundary unconstrained).")
    p.add_argument("--precondition", action="store_true",
                   help="Deep Ritz only: precondition the energy gradient with a GMG V-cycle "
                        "(M~A^-1). Flattens the A-norm dynamics toward the supervised/Newton "
                        "direction. Implies hard-BC.")

    # Preconditioner selection (PLS, or preconditioned Deep Ritz)
    p.add_argument("--precond_kind", choices=["multigrid", "blend", "power"],
                   default="multigrid",
                   help="Preconditioner P≈A^-1. 'multigrid' (default): geometric-multigrid "
                        "V-cycle (computational path). 'blend': convex mix (1-t)I+tA^-1. "
                        "'power': fractional power A^-s. blend/power are exact spectral "
                        "operators for illustrating the residual->supervised transition.")
    p.add_argument("--precond_strength", type=float, default=1.0,
                   help="Strength for blend (t) / power (s) in [0,1]. 0 -> P=I "
                        "(no preconditioning); 1 -> P=A^-1 (supervised). Ignored for multigrid.")

    # Multigrid preconditioner settings (used when --precond_kind multigrid)
    p.add_argument("--mg_levels", type=int, default=4,
                   help="Number of GMG levels for PLS (incl. finest).")
    p.add_argument("--mg_pre_smooth", type=int, default=2,
                   help="Pre-smoothing Jacobi sweeps per V-cycle level.")
    p.add_argument("--mg_post_smooth", type=int, default=2,
                   help="Post-smoothing Jacobi sweeps per V-cycle level.")
    p.add_argument("--mg_omega", type=float, default=2.0 / 3.0,
                   help="Damping factor for weighted Jacobi smoother.")

    # Optimizer (the place to plug in Shampoo via build_optimizer)
    p.add_argument("--optimizer", type=str, default="adam",
                   help="adam | adamw | sgd | <register-yours-in build_optimizer>")

    # FNO model
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--n_modes", type=int, nargs=2, default=[16, 16])
    p.add_argument("--grid_resolution", type=int, default=64)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output_dir", type=str, default="output")

    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--sample_idx", type=int, nargs="+", default=[0])
    return p


def main():
    args = build_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    print("=" * 60)
    print("TensorPILS — FNO + Poisson 2D")
    print("=" * 60)
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
            print(f"precond     : {args.precond_kind}  strength={args.precond_strength:.3f}")
    print(f"K           : {args.k}")
    print(f"samples     : train={args.n_train}, val={args.n_val}, test={args.n_test}")
    print(f"epochs/bs   : {args.epochs} / {args.batch_size}")
    print(f"lr          : {args.lr}  ->  {args.lr_min}")
    print(f"optimizer   : {args.optimizer}  (weight_decay={args.weight_decay})")
    print(f"FNO modes   : {tuple(args.n_modes)}   hidden={args.hidden_dim}   "
          f"layers={args.num_layers}   grid={args.grid_resolution}^2")
    print(f"device      : {device}")
    print("=" * 60 + "\n")

    print("Building datasets...")
    train_ds, val_ds, test_ds = create_datasets(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        K=args.k, grid_resolution=args.grid_resolution, seed=args.seed,
    )
    print(f"  train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, "
          f"grid={train_ds.grid_size}\n")

    print("Building model...")
    model = FNOModel(
        n_modes=tuple(args.n_modes),
        hidden_channels=args.hidden_dim,
        n_layers=args.num_layers,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {n_params:,}\n")

    trainer = Trainer(
        model=model,
        train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
        loss_type=args.loss,
        optimizer_name=args.optimizer,
        lr=args.lr, lr_min=args.lr_min, weight_decay=args.weight_decay,
        batch_size=args.batch_size, epochs=args.epochs,
        device=device, output_dir=args.output_dir,
        lambda_bc=args.lambda_bc,
        bc_mode=args.bc_mode,
        precondition=args.precondition,
        precond_kind=args.precond_kind,
        precond_strength=args.precond_strength,
        mg_levels=args.mg_levels,
        mg_pre_smooth=args.mg_pre_smooth,
        mg_post_smooth=args.mg_post_smooth,
        mg_omega=args.mg_omega,
    )

    if args.eval_only:
        if not trainer.load_checkpoint(args.checkpoint):
            return
        test_mse, test_l2, test_rl2 = trainer.test()
        print(f"Test MSE: {test_mse:.2e}  Test FEM-L2: {test_l2:.2e}  Test rel-L2: {test_rl2:.2%}")
        for idx in args.sample_idx:
            if idx < len(train_ds):
                trainer.visualize_sample(train_ds, "train", idx)
            if idx < len(test_ds):
                trainer.visualize_sample(test_ds, "test", idx)
        trainer.compute_error_distribution()
    else:
        test_mse, test_l2, test_rl2 = trainer.train()
        print("\n" + "=" * 60)
        print("Done.")
        print(f"  best val MSE   : {trainer.stats.best_val_error:.2e}")
        print(f"  final test MSE : {test_mse:.2e}")
        print(f"  final test L2  : {test_l2:.2e}")
        print(f"  final test rL2 : {test_rl2:.2%}")
        print("=" * 60)


if __name__ == "__main__":
    main()
