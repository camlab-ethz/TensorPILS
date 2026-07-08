"""Training loops, evaluation, checkpointing, and plotting orchestration.

``BaseTrainer`` holds the machinery shared by every PDE: optimizer/scheduler,
the epoch loop, best-model tracking, checkpointing (keyed by ``_file_prefix``),
and the boundary projection / loss-curve helpers. Two subclasses specialize it:

* :class:`PoissonTrainer` — static Poisson, one of four losses (grid-space MSE eval).
* :class:`WaveTrainer`    — autoregressive wave time-stepper (trajectory-MSE eval).

Evaluation is **always** grid-space MSE against the analytical solution, regardless of
the training loss, so model selection and the reported error stay comparable.
"""

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .meshing import node_to_grid, grid_to_node
from .physics import apply_zero_boundary
from .preconditioners import Preconditioner, GeometricMultigrid, build_preconditioner
from .losses import build_loss, build_wave_loss, build_ac_loss
from .optim import build_optimizer
from . import viz

__all__ = ["BaseTrainer", "Trainer", "PoissonTrainer", "RolloutTrainer",
           "WaveTrainer", "ACTrainer", "TrainingStats"]


@dataclass
class TrainingStats:
    train_losses: List[float] = field(default_factory=list)
    val_errors: List[float] = field(default_factory=list)
    val_l2_errors: List[float] = field(default_factory=list)
    val_rel_l2_errors: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    # Per-epoch relative-L2 on extra (out-of-distribution) eval sets, keyed by label.
    ood_rel_l2: dict = field(default_factory=dict)
    best_epoch: int = 0
    best_val_error: float = float("inf")
    # Preconditioner identity + conditioning (for the sweep / collapse plot).
    precond_kind: str = ""
    precond_strength: float = float("nan")
    precond_cond_pa: float = float("nan")
    precond_cond_h: float = float("nan")


class BaseTrainer:
    """Shared training scaffolding. Subclasses build their datasets/loaders/criterion,
    then implement ``train_epoch``, ``validate``/``test``, ``_file_prefix`` and the
    end-of-training visualization hook ``_after_train_viz``."""

    def __init__(
        self,
        model: nn.Module,
        loss_type: str,
        K: int,
        optimizer_name: str = "adam",
        lr: float = 1e-3,
        lr_min: float = 1e-6,
        weight_decay: float = 0.0,
        epochs: int = 500,
        device: str = "cuda",
        output_dir: str = "output",
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.output_dir = output_dir
        self.loss_type = loss_type
        self.K = K

        self.optimizer = build_optimizer(optimizer_name, model.parameters(),
                                         lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr_min,
        )

        self.stats = TrainingStats()
        self.best_state = None

        # Subclasses set these before train() (problem carries A/M + boundary_mask).
        self.problem = None
        self.grid_size = None
        self.eval_project_bc = False

        for sub in ("checkpoints", "curves", "visualization", "error"):
            os.makedirs(f"{output_dir}/{sub}", exist_ok=True)

    # -------------------- boundary projection --------------------
    def _project_zero_bc(self, u_grid: torch.Tensor) -> torch.Tensor:
        """Project a grid field onto the zero Dirichlet boundary. ``[H, W]`` or ``[..., H, W]``."""
        mask = self.problem.boundary_mask.to(u_grid.device)
        nx, ny = self.grid_size
        node = grid_to_node(u_grid, nx, ny)
        node = apply_zero_boundary(node, mask)
        return node_to_grid(node, nx, ny)

    def _apply_eval_bc(self, u_grid: torch.Tensor) -> torch.Tensor:
        """Project the prediction onto the zero Dirichlet BC at eval time when the boundary
        was unconstrained during training; otherwise pass through."""
        if not self.eval_project_bc:
            return u_grid
        return self._project_zero_bc(u_grid)

    # -------------------- training loop --------------------
    def _before_train(self):
        """Hook run once before the epoch loop (e.g. multigrid sanity check)."""

    def _after_train_viz(self):
        """Hook run after training to write sample panels / error distribution."""

    def train(self) -> float:
        print(f"Training FNO with {self.loss_type} loss  (device: {self.device})\n")
        self._before_train()

        with tqdm(range(self.epochs), desc="Training", unit="epoch", colour="green") as bar:
            for epoch in bar:
                tr = self.train_epoch()
                vl = self.validate()
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]

                self.stats.train_losses.append(tr)
                self.stats.val_errors.append(vl)
                self.stats.learning_rates.append(lr)

                if vl < self.stats.best_val_error:
                    self.stats.best_val_error = vl
                    self.stats.best_epoch = epoch
                    self.best_state = deepcopy(self.model.state_dict())
                    self._save_checkpoint(epoch, vl)

                bar.set_postfix(loss=f"{tr:.2e}", val=f"{vl:.2e}",
                                best=f"{self.stats.best_val_error:.2e}", lr=f"{lr:.2e}")

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state, strict=False)
            print(f"\nRestored best model from epoch {self.stats.best_epoch} "
                  f"(val={self.stats.best_val_error:.2e})")

        test_err = self.test()
        print(f"Test MSE: {test_err:.2e}")
        self.plot_loss_curve()
        self._after_train_viz()
        return test_err

    # -------------------- checkpointing --------------------
    def _file_prefix(self) -> str:
        raise NotImplementedError

    def _save_checkpoint(self, epoch, val_error):
        path = f"{self.output_dir}/checkpoints/{self._file_prefix()}_best.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_error": val_error,
            "stats": self.stats,
        }, path)

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        if path is None:
            path = f"{self.output_dir}/checkpoints/{self._file_prefix()}_best.pth"
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return False
        ck = torch.load(path, map_location=self.device, weights_only=False)
        sd = ck["model_state_dict"]
        sd.pop("_metadata", None)
        self.model.load_state_dict(sd, strict=False)
        self.stats = ck.get("stats", TrainingStats())
        print(f"Loaded checkpoint from epoch {ck.get('epoch', '?')}, "
              f"val={ck.get('val_error', float('nan')):.2e}")
        return True

    # -------------------- plotting --------------------
    def plot_loss_curve(self):
        path = f"{self.output_dir}/curves/{self._file_prefix()}_loss.png"
        viz.plot_loss_curve(self.stats, self.loss_type, self.K, path)


class PoissonTrainer(BaseTrainer):
    """Trainer for the static Poisson problem (``data`` / ``galerkin`` / ``deepritz`` / ``pls``)."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        test_dataset,
        loss_type: str = "galerkin",
        optimizer_name: str = "adam",
        lr: float = 1e-3,
        lr_min: float = 1e-6,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        epochs: int = 500,
        device: str = "cuda",
        output_dir: str = "output",
        lambda_bc: float = 100.0,
        bc_mode: str = "penalty",
        precondition: bool = False,
        precond_kind: str = "multigrid",
        precond_strength: float = 1.0,
        mg_levels: int = 4,
        mg_pre_smooth: int = 2,
        mg_post_smooth: int = 2,
        mg_omega: float = 2.0 / 3.0,
        eval_datasets: Optional[dict] = None,
    ):
        super().__init__(model, loss_type, train_dataset.K, optimizer_name,
                         lr, lr_min, weight_decay, epochs, device, output_dir)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.grid_size = train_dataset.grid_size

        # All splits share one PoissonProblem; move it (and its A, M) to the device once.
        self.problem = train_dataset.problem.to(device)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                       shuffle=True, collate_fn=self._collate)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                     shuffle=False, collate_fn=self._collate)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                      shuffle=False, collate_fn=self._collate)

        # Extra (out-of-distribution) eval loaders, scored each epoch. Same grid as the
        # training set, so self.problem.M is the correct metric for their relative-L2.
        self.eval_loaders = {
            label: DataLoader(ds, batch_size=batch_size, shuffle=False,
                              collate_fn=self._collate)
            for label, ds in (eval_datasets or {}).items()
        }

        self.bc_mode = bc_mode
        self.precondition = precondition

        # Preconditioner for PLS or preconditioned Deep Ritz (multigrid or spectral).
        self.precond: Optional[Preconditioner] = None
        self.precond_kind = precond_kind
        self.precond_strength = precond_strength
        self.mg_settings = (mg_levels, mg_pre_smooth, mg_post_smooth, mg_omega)
        needs_precond = (loss_type == "pls") or (loss_type == "deepritz" and precondition)
        if needs_precond:
            self.precond = build_preconditioner(
                kind=precond_kind, problem=self.problem, grid_size=self.grid_size,
                mg_levels=mg_levels, mg_pre_smooth=mg_pre_smooth,
                mg_post_smooth=mg_post_smooth, mg_omega=mg_omega,
                strength=precond_strength, device=device,
            )

        # The homogeneous Dirichlet BC (u=0 on the boundary) is known data, so we always
        # enforce it at eval by projecting the prediction's boundary to zero — for every
        # loss, so the reported error never counts a boundary the BC already fixes.
        # (Assumes homogeneous BC; a non-zero Dirichlet problem would project to the known
        # boundary values instead.)
        self.eval_project_bc = True

        self.criterion = build_loss(loss_type, self.problem, lambda_bc,
                                    bc_mode=bc_mode, precond=self.precond,
                                    precondition=precondition)

        # OOD histories + preconditioner diagnostics on the shared stats object
        # (BaseTrainer.__init__ already built optimizer/scheduler/stats/best_state).
        self.stats.ood_rel_l2 = {label: [] for label in self.eval_loaders}
        if self.precond is not None:
            self.stats.precond_kind = self.precond_kind
            self.stats.precond_strength = self.precond_strength
            self.stats.precond_cond_pa = getattr(self.precond, "cond_PA", float("nan"))
            self.stats.precond_cond_h = getattr(self.precond, "cond_H", float("nan"))
        os.makedirs(f"{self.output_dir}/results", exist_ok=True)

    @staticmethod
    def _collate(batch):
        fs_grid = torch.stack([item[0][0] for item in batch], dim=0)    # [B, 1, H, W]
        us_grid = torch.stack([item[1] for item in batch], dim=0)        # [B, H, W]
        grid_size = batch[0][0][1]
        fs_node = torch.stack([item[0][2] for item in batch], dim=0)     # [B, N]
        us_node = torch.stack([item[0][3] for item in batch], dim=0)     # [B, N]
        return (fs_grid, grid_size, fs_node, us_node), us_grid

    # -------------------- forward pass --------------------
    def _forward(self, fs_grid, us_grid, fs_node, us_node):
        u_pred_grid = self.model(fs_grid).squeeze(1)                     # [B, H, W]
        if self.loss_type == "data":
            return self.criterion(u_pred_grid, us_grid)
        # Galerkin / Deep Ritz / PLS: convert back to node order.
        nx, ny = self.grid_size
        u_pred_node = grid_to_node(u_pred_grid, nx, ny)                  # [B, N]
        return self.criterion(u_pred_node, fs_node, us_node)

    def train_epoch(self) -> float:
        self.model.train()
        total, nb = 0.0, 0
        for batch_data, us in self.train_loader:
            self.optimizer.zero_grad()
            fs_grid, _, fs_node, us_node = batch_data
            fs_grid, us = fs_grid.to(self.device), us.to(self.device)
            fs_node, us_node = fs_node.to(self.device), us_node.to(self.device)
            loss = self._forward(fs_grid, us, fs_node, us_node)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            nb += 1
        return total / nb

    @torch.no_grad()
    def _eval_loader(self, loader):
        self.model.eval()
        total_mse, total_l2, total_rel_l2, n = 0.0, 0.0, 0.0, 0
        nx, ny = self.grid_size
        for batch_data, us in loader:
            fs_grid, _, _, us_node = batch_data
            fs_grid, us = fs_grid.to(self.device), us.to(self.device)
            us_node = us_node.to(self.device)
            u_pred = self.model(fs_grid).squeeze(1)                      # [B, H, W]
            u_pred = self._apply_eval_bc(u_pred)
            mse = ((u_pred - us) ** 2).mean().item()
            total_mse += mse * fs_grid.shape[0]
            e = grid_to_node(u_pred, nx, ny) - us_node                  # [B, N]
            Me = self.problem._spmm(self.problem.M, e)                   # [B, N]
            l2_per_sample = (e * Me).sum(dim=1).sqrt()                   # [B]
            total_l2 += l2_per_sample.sum().item()
            Mu = self.problem._spmm(self.problem.M, us_node)             # [B, N]
            u_norm = (us_node * Mu).sum(dim=1).sqrt()                    # [B]
            total_rel_l2 += (l2_per_sample / u_norm.clamp(min=1e-12)).sum().item()
            n += fs_grid.shape[0]
        return total_mse / n, total_l2 / n, total_rel_l2 / n

    def validate(self):
        return self._eval_loader(self.val_loader)

    def test(self):
        return self._eval_loader(self.test_loader)

    # -------------------- multigrid sanity check (GMG only) --------------------
    @torch.no_grad()
    def _check_mg(self, n_samples: int = 4, n_cycles: int = 3):
        """Apply V-cycles to random zero-boundary residuals and report ‖r−A V(r)‖/‖r‖.
        A healthy GMG reduces the residual by ~0.05–0.2 per cycle (mesh-independent)."""
        if not isinstance(self.precond, GeometricMultigrid):
            return
        nx, ny = self.grid_size
        N = nx * ny
        mask = self.problem.boundary_mask.to(self.device)

        torch.manual_seed(0)
        r0 = torch.randn(n_samples, N, device=self.device)
        r0 = apply_zero_boundary(r0, mask)
        A_sp = self.precond._A_sparse(0).to(self.device)

        print(f"\n[MG sanity] n_levels={self.precond.n_levels}, "
              f"pre/post={self.precond.pre_smooth}/{self.precond.post_smooth}, "
              f"omega={self.precond.omega:.3f}")
        print(f"[MG sanity] level dims: {self.precond.dims}")
        r = r0.clone()
        e_total = torch.zeros_like(r)
        norms = [r.norm(dim=1).mean().item()]
        for _ in range(n_cycles):
            e_total = e_total + self.precond.v_cycle(r)
            Ae = torch.sparse.mm(A_sp, e_total.T).T
            r = r0 - Ae
            norms.append(r.norm(dim=1).mean().item())
        for k in range(n_cycles):
            print(f"[MG sanity] after cycle {k+1}: ||r||/||r0|| = {norms[k+1]/norms[0]:.3e}   "
                  f"(reduction this cycle: {norms[k+1]/max(norms[k],1e-30):.3e})")
        print()

    # -------------------- training loop --------------------
    def train(self) -> float:
        print(f"Training FNO with {self.loss_type} loss")
        print(f"  Train: {len(self.train_dataset)}  Val: {len(self.val_dataset)}  "
              f"Test: {len(self.test_dataset)}  Device: {self.device}\n")
        if self.precond is not None:
            self.precond.report()
            self._check_mg()

        with tqdm(range(self.epochs), desc="Training", unit="epoch", colour="green") as bar:
            for epoch in bar:
                tr = self.train_epoch()
                vl, l2, rl2 = self.validate()
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]

                self.stats.train_losses.append(tr)
                self.stats.val_errors.append(vl)
                self.stats.val_l2_errors.append(l2)
                self.stats.val_rel_l2_errors.append(rl2)
                self.stats.learning_rates.append(lr)

                # Out-of-distribution generalization: relative-L2 on each extra eval set.
                for label, loader in self.eval_loaders.items():
                    self.stats.ood_rel_l2[label].append(self._eval_loader(loader)[2])

                if vl < self.stats.best_val_error:
                    self.stats.best_val_error = vl
                    self.stats.best_epoch = epoch
                    self.best_state = deepcopy(self.model.state_dict())
                    self._save_checkpoint(epoch, vl)

                bar.set_postfix(loss=f"{tr:.2e}", val=f"{vl:.2e}",
                                l2=f"{l2:.2e}", rl2=f"{rl2:.2%}",
                                best=f"{self.stats.best_val_error:.2e}", lr=f"{lr:.2e}")

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state, strict=False)
            print(f"\nRestored best model from epoch {self.stats.best_epoch} "
                  f"(val={self.stats.best_val_error:.2e})")

        test_mse, test_l2, test_rl2 = self.test()
        print(f"Test MSE: {test_mse:.2e}  Test FEM-L2: {test_l2:.2e}  Test rel-L2: {test_rl2:.2%}")
        self.plot_loss_curve()
        self.visualize_sample(self.train_dataset, "train", 0)
        self.visualize_sample(self.test_dataset, "test", 0)
        self.compute_error_distribution()
        self._save_results_json((test_mse, test_l2, test_rl2))
        return test_mse, test_l2, test_rl2

    def _save_results_json(self, test_metrics) -> None:
        """Dump full stats + config to ``results/{prefix}.json`` for cross-run aggregation."""
        import json
        from dataclasses import asdict

        test_mse, test_l2, test_rl2 = test_metrics
        record = {
            "prefix": self._file_prefix(),
            "loss_type": self.loss_type,
            "bc_mode": self.bc_mode,
            "precondition": self.precondition,
            "precond_kind": self.precond_kind,
            "precond_strength": self.precond_strength,
            "K": self.K,
            "n_train": len(self.train_dataset),
            "epochs": self.epochs,
            "test_mse": test_mse, "test_l2": test_l2, "test_rl2": test_rl2,
            "stats": asdict(self.stats),
        }
        path = f"{self.output_dir}/results/{self._file_prefix()}.json"
        with open(path, "w") as fh:
            json.dump(record, fh)
        print(f"Results -> {path}")

    # -------------------- checkpointing --------------------
    def _precond_tag(self) -> str:
        """Short tag describing the active preconditioner (for run file names)."""
        if self.precond_kind == "multigrid":
            L, pre, post, _ = self.mg_settings
            return f"mg-L{L}-s{pre}{post}"
        return f"{self.precond_kind}-{self.precond_strength:.2f}"

    def _file_prefix(self) -> str:
        if self.loss_type == "deepritz":
            tag = f"_precond-{self._precond_tag()}" if self.precondition else f"_bc-{self.bc_mode}"
        elif self.loss_type == "data":
            tag = "_bc-hard" if self.bc_mode == "hard" else ""
        elif self.loss_type == "pls":
            tag = f"_{self._precond_tag()}"
        else:
            tag = ""
        return (f"fno_{self.loss_type}{tag}_K{self.K}_"
                f"samples-{len(self.train_dataset)}-{len(self.val_dataset)}-{len(self.test_dataset)}")

    def visualize_sample(self, dataset, split: str, sample_idx: int = 0):
        suffix = "" if sample_idx == 0 else f"_sample{sample_idx}"
        path = f"{self.output_dir}/visualization/{self._file_prefix()}_{split}{suffix}.png"
        viz.visualize_sample(self.model, dataset, split, sample_idx, self.device,
                             self._apply_eval_bc, self.loss_type, self.K, path)

    def compute_error_distribution(self):
        path = f"{self.output_dir}/error/{self._file_prefix()}_error_dist.png"
        return viz.compute_error_distribution(self.model, self.test_dataset, self.device,
                                              self._apply_eval_bc, self.loss_type, self.K, path)

    def _after_train_viz(self):
        self.visualize_sample(self.train_dataset, "train", 0)
        self.visualize_sample(self.test_dataset, "test", 0)
        self.compute_error_distribution()


# Backwards-compatible alias: the historical name for the Poisson trainer.
Trainer = PoissonTrainer


class RolloutTrainer(BaseTrainer):
    r"""Shared trainer for time-dependent PDEs as autoregressive FNO steppers.

    The FNO maps the last ``n_seed_frames`` frames (an ``n_seed_frames``-channel grid) to the
    next frame. Each batch is seeded with the first ``n_seed_frames`` ground-truth frames and
    rolled forward ``rollout_steps`` times; predicted frames are projected onto the zero
    Dirichlet boundary before being fed back. The loss is ``λ_gal·galerkin + λ_data·data``:
    the Galerkin term is the weak-form residual over the rollout (supplied by ``self.criterion``,
    which the subclass builds), the optional data term is the trajectory MSE against the
    reference. Evaluation is always grid-space rollout MSE versus the reference trajectory.

    Subclasses set ``self.criterion`` after ``super().__init__`` and implement ``_file_prefix``.
    ``n_seed_frames`` = 2 for the (second-order) wave equation, 1 for (first-order) Allen–Cahn.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        test_dataset,
        loss_type: str,
        n_seed_frames: int,
        optimizer_name: str = "adam",
        lr: float = 1e-3,
        lr_min: float = 1e-6,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        epochs: int = 500,
        device: str = "cuda",
        output_dir: str = "output",
        lambda_galerkin: float = 1.0,
        lambda_data: float = 0.0,
        rollout_steps: int = 4,
    ):
        super().__init__(model, loss_type, train_dataset.K, optimizer_name,
                         lr, lr_min, weight_decay, epochs, device, output_dir)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.grid_size = train_dataset.grid_size

        self.problem = train_dataset.problem.to(device)
        self.dt = train_dataset.dt
        self.n_steps = train_dataset.n_steps
        self.n_seed_frames = n_seed_frames
        # Rollout produces preds at frame indices n_seed .. n_seed+R-1, which must exist.
        self.rollout_steps = max(1, min(rollout_steps, self.n_steps - n_seed_frames + 1))
        self.lambda_galerkin = lambda_galerkin
        self.lambda_data = lambda_data

        # Predictions are unconstrained on the boundary during training (the residual masks
        # it), so project to zero-BC at eval — matching the zero-boundary reference.
        self.eval_project_bc = True

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                       shuffle=True, collate_fn=self._collate)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                     shuffle=False, collate_fn=self._collate)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                      shuffle=False, collate_fn=self._collate)

        self.criterion = None   # set by the subclass

    @staticmethod
    def _collate(batch):
        trajs_grid = torch.stack([item[0] for item in batch], dim=0)     # [B, T+1, H, W]
        trajs_node = torch.stack([item[1] for item in batch], dim=0)     # [B, T+1, N]
        return trajs_grid, trajs_node

    # -------------------- autoregressive rollout --------------------
    def _rollout(self, traj_grid: torch.Tensor):
        """Seed with the first ``n_seed_frames`` frames and roll ``rollout_steps`` forward.

        Returns ``(preds_grid [B, R, H, W], seq_node [B, n_seed+R, N])`` where ``seq_node`` is
        the ground-truth seed frames followed by the predicted frames, all zeroed on the
        Dirichlet boundary. ``preds_grid`` are the predicted frames (also projected)."""
        nx, ny = self.grid_size
        ns = self.n_seed_frames
        window = [self._project_zero_bc(traj_grid[:, i]) for i in range(ns)]   # each [B, H, W]
        seq_node = [grid_to_node(f, nx, ny) for f in window]
        preds_grid = []
        for _ in range(self.rollout_steps):
            inp = torch.stack(window, dim=1)                             # [B, ns, H, W]
            nxt = self._project_zero_bc(self.model(inp).squeeze(1))      # [B, H, W]
            preds_grid.append(nxt)
            seq_node.append(grid_to_node(nxt, nx, ny))
            window = window[1:] + [nxt]                                  # slide the window
        return torch.stack(preds_grid, dim=1), torch.stack(seq_node, dim=1)

    def train_epoch(self) -> float:
        self.model.train()
        total, nb = 0.0, 0
        ns, R = self.n_seed_frames, self.rollout_steps
        for trajs_grid, _ in self.train_loader:
            self.optimizer.zero_grad()
            trajs_grid = trajs_grid.to(self.device)
            preds_grid, seq_node = self._rollout(trajs_grid)
            loss = self.lambda_galerkin * self.criterion(seq_node)
            if self.lambda_data > 0.0:
                ref = trajs_grid[:, ns:ns + R]                           # [B, R, H, W]
                loss = loss + self.lambda_data * ((preds_grid - ref) ** 2).mean()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            nb += 1
        return total / nb

    @torch.no_grad()
    def _eval_loader(self, loader) -> float:
        """Rollout MSE vs the reference trajectory over the ``rollout_steps`` window."""
        self.model.eval()
        ns, R = self.n_seed_frames, self.rollout_steps
        total_mse, n = 0.0, 0
        for trajs_grid, _ in loader:
            trajs_grid = trajs_grid.to(self.device)
            preds_grid, _ = self._rollout(trajs_grid)                    # [B, R, H, W]
            ref = trajs_grid[:, ns:ns + R]
            mse = ((preds_grid - ref) ** 2).mean().item()
            total_mse += mse * trajs_grid.shape[0]
            n += trajs_grid.shape[0]
        return total_mse / n

    def validate(self):
        return self._eval_loader(self.val_loader)

    def test(self):
        return self._eval_loader(self.test_loader)

    # -------------------- viz --------------------
    def visualize_sample(self, dataset, split: str, sample_idx: int = 0):
        suffix = "" if sample_idx == 0 else f"_sample{sample_idx}"
        path = f"{self.output_dir}/visualization/{self._file_prefix()}_{split}{suffix}.png"
        viz.visualize_rollout_sample(self.model, dataset, split, sample_idx, self.device,
                                     self._project_zero_bc, self.rollout_steps, self.n_seed_frames,
                                     self.loss_type, self.K, path)

    def compute_error_distribution(self):
        path = f"{self.output_dir}/error/{self._file_prefix()}_error_dist.png"
        return viz.compute_rollout_error_distribution(
            self.model, self.test_dataset, self.device, self._project_zero_bc,
            self.rollout_steps, self.n_seed_frames, self.loss_type, self.K, path)

    def _after_train_viz(self):
        self.visualize_sample(self.train_dataset, "train", 0)
        self.visualize_sample(self.test_dataset, "test", 0)
        self.compute_error_distribution()


class WaveTrainer(RolloutTrainer):
    r"""Wave equation stepper: FNO maps ``[u^{n-1}, u^n]`` → ``u^{n+1}`` (2 seed frames),
    trained on the central-difference weak-form residual (label-free) + optional data MSE."""

    def __init__(self, model, train_dataset, val_dataset, test_dataset,
                 loss_type="galerkin", optimizer_name="adam", lr=1e-3, lr_min=1e-6,
                 weight_decay=0.0, batch_size=32, epochs=500, device="cuda",
                 output_dir="output", lambda_galerkin=1.0, lambda_data=0.0,
                 rollout_steps=4, discount_factor=1.0):
        super().__init__(model, train_dataset, val_dataset, test_dataset, loss_type,
                         n_seed_frames=2, optimizer_name=optimizer_name, lr=lr, lr_min=lr_min,
                         weight_decay=weight_decay, batch_size=batch_size, epochs=epochs,
                         device=device, output_dir=output_dir, lambda_galerkin=lambda_galerkin,
                         lambda_data=lambda_data, rollout_steps=rollout_steps)
        self.c = train_dataset.c
        self.discount_factor = discount_factor
        self.criterion = build_wave_loss(self.problem, c=self.c, dt=self.dt,
                                         discount=discount_factor)

    def _file_prefix(self) -> str:
        return (f"fno_wave_{self.loss_type}_c{self.c:g}_dt{self.dt:g}_"
                f"T{self.n_steps}_R{self.rollout_steps}_K{self.K}_"
                f"samples-{len(self.train_dataset)}-{len(self.val_dataset)}-{len(self.test_dataset)}")


class ACTrainer(RolloutTrainer):
    r"""Allen–Cahn stepper: FNO maps ``u^n`` → ``u^{n+1}`` (1 seed frame), trained on the
    backward-Euler weak-form residual (label-free) + optional data MSE against the FEM reference."""

    def __init__(self, model, train_dataset, val_dataset, test_dataset,
                 loss_type="galerkin", optimizer_name="adam", lr=1e-3, lr_min=1e-6,
                 weight_decay=0.0, batch_size=32, epochs=500, device="cuda",
                 output_dir="output", lambda_galerkin=1.0, lambda_data=0.0,
                 rollout_steps=4, discount_factor=1.0):
        super().__init__(model, train_dataset, val_dataset, test_dataset, loss_type,
                         n_seed_frames=1, optimizer_name=optimizer_name, lr=lr, lr_min=lr_min,
                         weight_decay=weight_decay, batch_size=batch_size, epochs=epochs,
                         device=device, output_dir=output_dir, lambda_galerkin=lambda_galerkin,
                         lambda_data=lambda_data, rollout_steps=rollout_steps)
        self.a = train_dataset.a
        self.eps = train_dataset.eps
        self.discount_factor = discount_factor
        self.criterion = build_ac_loss(self.problem, a=self.a, eps=self.eps, dt=self.dt,
                                       discount=discount_factor)

    def _file_prefix(self) -> str:
        return (f"fno_ac_{self.loss_type}_a{self.a:g}_eps{self.eps:g}_dt{self.dt:g}_"
                f"T{self.n_steps}_R{self.rollout_steps}_K{self.K}_"
                f"samples-{len(self.train_dataset)}-{len(self.val_dataset)}-{len(self.test_dataset)}")
