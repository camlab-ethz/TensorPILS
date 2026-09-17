"""Training loops, evaluation, checkpointing, and plotting orchestration.

``BaseTrainer`` holds the machinery shared by every PDE: optimizer/scheduler,
the epoch loop, best-model tracking, checkpointing (keyed by ``_file_prefix``),
and the boundary projection / loss-curve helpers. Two subclasses specialize it:

* :class:`PoissonTrainer` — static Poisson, one of four losses (grid-space MSE eval).
* :class:`WaveTrainer`    — autoregressive wave time-stepper (trajectory-MSE eval).

Evaluation is **always** grid-space MSE against the analytical solution, regardless of
the training loss, so model selection and the reported error stay comparable.
"""

import math
import os
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, RandomSampler
from tqdm import tqdm

from .meshing import node_to_grid, grid_to_node
from .physics import apply_zero_boundary
from .preconditioners import (Preconditioner, GeometricMultigrid,
                              StokesBlockPreconditioner, StokesBlendPreconditioner,
                              StokesMonolithicMultigrid, build_preconditioner)
from .losses import build_loss, build_wave_loss, build_ac_loss, build_stokes_loss
from .optim import build_optimizer
from . import viz

__all__ = ["BaseTrainer", "Trainer", "PoissonTrainer", "RolloutTrainer",
           "WaveTrainer", "ACTrainer", "StokesTrainer", "TrainingStats",
           "arch_tag", "ArchPrefixMixin",
           "GAOTPoissonTrainer", "GAOTWaveTrainer", "GAOTACTrainer", "GAOTStokesTrainer",
           "UnstructuredPoissonTrainer", "GAOTUnstructuredPoissonTrainer",
           "UnstructuredStokesTrainer", "GAOTUnstructuredStokesTrainer"]


#: Model class name -> the tag that goes in a run's file names. ``fno`` is the default so that
#: every filename written before a second architecture existed stays byte-identical.
_ARCH_TAGS = {"DeepONetModel": "deeponet", "GAOTModel": "gaot"}


def arch_tag(model) -> str:
    """``fno`` / ``deeponet`` / ``gaot`` for the run file name, seen through any wrapper.

    The hard-BC wrappers (``MollifiedModel``, ``ZeroBoundaryModel``) hold the real model in
    ``.model``, so unwrap one level before looking at the type.
    """
    inner = getattr(model, "model", model)
    return _ARCH_TAGS.get(type(inner).__name__, "fno")


class ArchPrefixMixin:
    """Put the architecture in the run file name.

    ``PoissonTrainer._file_prefix`` and friends hard-code ``fno_``, which was unambiguous while
    the FNO was the only architecture. Now that any loss can be paired with a DeepONet or with
    GAOT, a ``--model gaot --loss data`` run would land on the *same* checkpoint / results path
    as the FNO run of the same config and silently overwrite it. Rewriting the prefix here keeps
    the base trainers untouched and leaves every existing FNO filename byte-identical.
    """

    def _file_prefix(self) -> str:
        prefix = super()._file_prefix()
        tag = arch_tag(self.model)
        if tag == "fno":
            return prefix
        return tag + prefix[3:] if prefix.startswith("fno") else f"{tag}_{prefix}"


@dataclass
class TrainingStats:
    train_losses: List[float] = field(default_factory=list)
    val_errors: List[float] = field(default_factory=list)
    val_l2_errors: List[float] = field(default_factory=list)
    val_rel_l2_errors: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    # Rollout (time-dependent) FEM-L2 diagnostics, per epoch. ``val_rel_l2_steps`` is the
    # per-timestep relative L2 series over the rollout (len = rollout_steps); the two scalars are
    # the space-time aggregate and the final-time value derived from it.
    val_rel_l2_steps: List[List[float]] = field(default_factory=list)
    val_st_rel_l2: List[float] = field(default_factory=list)
    val_final_rel_l2: List[float] = field(default_factory=list)
    # Per-epoch relative-L2 on extra (out-of-distribution) eval sets, keyed by label.
    ood_rel_l2: dict = field(default_factory=dict)
    best_epoch: int = 0
    best_val_error: float = float("inf")
    # Epoch at which patience-based early stopping fired (-1: ran the full budget).
    stopped_epoch: int = -1
    # Stokes: the two fields are scored separately (different spaces and grids).
    val_rel_l2_u: List[float] = field(default_factory=list)
    val_rel_l2_p: List[float] = field(default_factory=list)
    # Wall-clock seconds of each training epoch (train_epoch only, no validation) -- the source
    # for the paper's time-per-epoch column. Missing in results written before it existed.
    epoch_times: List[float] = field(default_factory=list)
    # Preconditioner identity + conditioning (for the sweep / collapse plot).
    precond_kind: str = ""
    precond_strength: float = float("nan")
    precond_method: str = "dense"
    precond_cond_pa: float = float("nan")
    precond_cond_h: float = float("nan")


class BaseTrainer:
    #: Write the best-model ``.pth``? Set to False for sweeps, where the checkpoints are never
    #: reloaded and each one carries the optimizer state as well (Adam's two moment buffers),
    #: roughly 3x the parameter count. Model selection is UNAFFECTED either way: ``best_state``
    #: lives in memory and is restored after training regardless. A class attribute rather than a
    #: constructor argument, so it does not have to be threaded through five subclass signatures.
    save_checkpoints = True

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
        # Label for the end-of-training test print. Most trainers report a grid MSE; Stokes
        # selects on a combined relative error, so it says so rather than mislabelling it.
        self._test_label = "Test MSE"

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

    def _save_results(self):
        """Hook run right after the test evaluation and BEFORE any plotting: persist the run's
        results (the results JSON). Plots are diagnostics and must never cost a finished run its
        results -- a failing loss-curve plot used to abort ``train()`` before the JSON was written."""

    def _run_viz(self, fn, *args):
        """Run one diagnostic plotting step; a failure is reported with its traceback, not raised."""
        try:
            fn(*args)
        except Exception:
            print(f"WARNING: {getattr(fn, '__name__', fn)} failed -- results are already saved:")
            traceback.print_exc()

    def train(self) -> float:
        print(f"Training {arch_tag(self.model).upper()} with {self.loss_type} loss  (device: {self.device})\n")
        self._before_train()

        with tqdm(range(self.epochs), desc="Training", unit="epoch", colour="green") as bar:
            for epoch in bar:
                t0 = time.perf_counter()
                tr = self.train_epoch()
                self.stats.epoch_times.append(time.perf_counter() - t0)
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
        print(f"{self._test_label}: {test_err:.2e}")
        self._save_results()
        self._run_viz(self.plot_loss_curve)
        self._run_viz(self._after_train_viz)
        return test_err

    # -------------------- checkpointing --------------------
    def _file_prefix(self) -> str:
        raise NotImplementedError

    def _save_checkpoint(self, epoch, val_error):
        if not self.save_checkpoints:
            return
        path = f"{self.output_dir}/checkpoints/{self._file_prefix()}_best.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "model_config": getattr(self.model, "build_config", None),
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
        precond_method: str = "dense",
        mg_levels: int = 4,
        mg_pre_smooth: int = 2,
        mg_post_smooth: int = 2,
        mg_omega: float = 2.0 / 3.0,
        steps_per_epoch: Optional[int] = None,
        patience: Optional[int] = None,
        eval_datasets: Optional[dict] = None,
    ):
        super().__init__(model, loss_type, train_dataset.K, optimizer_name,
                         lr, lr_min, weight_decay, epochs, device, output_dir)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.grid_size = train_dataset.grid_size
        self.stream = isinstance(train_dataset, IterableDataset)
        self.steps_per_epoch = steps_per_epoch
        self.patience = patience

        # All splits share one PoissonProblem; move it (and its A, M) to the device once.
        self.problem = train_dataset.problem.to(device)

        if self.stream:
            # Infinite stream: the dataset yields its samples_per_epoch fresh samples per pass.
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                           collate_fn=self._collate)
        elif steps_per_epoch is not None:
            # Fixed-budget "virtual epoch": steps_per_epoch batches drawn i.i.d. WITH replacement
            # from the finite dataset — the empirical-measure analogue of the fresh stream, so a
            # dataset-size sweep varies only the sampled measure, not the algorithm.
            sampler = RandomSampler(train_dataset, replacement=True,
                                    num_samples=steps_per_epoch * batch_size)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                           sampler=sampler, collate_fn=self._collate)
        else:
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
        self.precond_method = precond_method
        self.mg_settings = (mg_levels, mg_pre_smooth, mg_post_smooth, mg_omega)
        needs_precond = (loss_type == "pls") or (loss_type == "deepritz" and precondition)
        if needs_precond:
            self.precond = build_preconditioner(
                kind=precond_kind, problem=self.problem, grid_size=self.grid_size,
                mg_levels=mg_levels, mg_pre_smooth=mg_pre_smooth,
                mg_post_smooth=mg_post_smooth, mg_omega=mg_omega,
                strength=precond_strength, method=precond_method, device=device,
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
            self.stats.precond_method = self.precond_method
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
        print(f"Training {arch_tag(self.model).upper()} with {self.loss_type} loss")
        ntr = (f"stream ({len(self.train_dataset)}/epoch)" if self.stream
               else len(self.train_dataset))
        print(f"  Train: {ntr}  Val: {len(self.val_dataset)}  "
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

                if (self.patience is not None
                        and epoch - self.stats.best_epoch >= self.patience):
                    self.stats.stopped_epoch = epoch
                    print(f"\nEarly stop at epoch {epoch}: no val improvement "
                          f"for {self.patience} epochs (best: epoch {self.stats.best_epoch}).")
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state, strict=False)
            print(f"\nRestored best model from epoch {self.stats.best_epoch} "
                  f"(val={self.stats.best_val_error:.2e})")

        test_mse, test_l2, test_rl2 = self.test()
        print(f"Test MSE: {test_mse:.2e}  Test FEM-L2: {test_l2:.2e}  Test rel-L2: {test_rl2:.2%}")
        self._save_results_json((test_mse, test_l2, test_rl2))      # before any plotting
        self._run_viz(self.plot_loss_curve)
        self._run_viz(self.visualize_sample, self.train_dataset, "train", 0)
        self._run_viz(self.visualize_sample, self.test_dataset, "test", 0)
        self._run_viz(self.compute_error_distribution)
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
            "precond_method": self.precond_method,
            "K": self.K,
            # Which labels this run was trained and scored against. Without it, an analytic-label
            # and a fem-label run are indistinguishable on disk -- same prefix, same filename --
            # and the only signal is the file mtime.
            "dataset_solution": getattr(self.train_dataset, "solution", None),
            "n_train": (None if self.stream else len(self.train_dataset)),
            "stream": self.stream,
            "steps_per_epoch": self.steps_per_epoch,
            "patience": self.patience,
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
        if self.precond_kind == "amg":
            # No level count: AmgX picks its own hierarchy depth from the matrix.
            _, pre, _, _ = self.mg_settings
            return f"amg-s{pre}"
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
        ntr = "inf" if self.stream else len(self.train_dataset)
        return (f"fno_{self.loss_type}{tag}_K{self.K}_"
                f"samples-{ntr}-{len(self.val_dataset)}-{len(self.test_dataset)}")

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
        bptt_mode=None,
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
        # Autodiff/BPTT mode. None -> current behaviour (full BPTT rollout). "pushforward" detaches
        # each step's rollout input so gradients are one-step; the loss-side "coupling" detach is
        # resolved separately (see ACTrainer). full_bptt/detach_prev keep the full rollout graph.
        self.bptt_mode = bptt_mode
        self.detach_rollout = (bptt_mode == "pushforward")

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
            # Slide the window. detach_rollout (pushforward) feeds the next step a detached input,
            # so each stored frame is a one-step function of its input (no backprop through time).
            window = window[1:] + [nxt.detach() if self.detach_rollout else nxt]
        return torch.stack(preds_grid, dim=1), torch.stack(seq_node, dim=1)

    def train_epoch(self) -> float:
        self.model.train()
        total, nb = 0.0, 0
        ns, R = self.n_seed_frames, self.rollout_steps
        for trajs_grid, _ in self.train_loader:
            self.optimizer.zero_grad()
            trajs_grid = trajs_grid.to(self.device)
            preds_grid, seq_node = self._rollout(trajs_grid)
            loss = preds_grid.new_zeros(())
            if self.lambda_galerkin != 0.0:          # never evaluate an unused residual: 0 * NaN = NaN
                loss = loss + self.lambda_galerkin * self.criterion(seq_node)
            if self.lambda_data > 0.0:
                ref = trajs_grid[:, ns:ns + R]                           # [B, R, H, W]
                loss = loss + self.lambda_data * ((preds_grid - ref) ** 2).mean()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            nb += 1
        return total / nb

    @torch.no_grad()
    def _eval_full(self, loader):
        """One rollout pass over ``loader`` returning ``(mse, rel_l2_steps, st_rel_l2)``:

        * ``mse``          — grid rollout MSE (mean over batch, rollout steps, grid);
        * ``rel_l2_steps`` — per-timestep **relative FEM-L2** over the rollout window, a list of
          ``rollout_steps`` values ``√(Σ‖ê^k−u^k‖²_M / Σ‖u^k‖²_M)`` (dataset-level, so robust to
          late frames whose reference norm decays toward zero);
        * ``st_rel_l2``    — the space-time aggregate ``√(Σ_k Σ‖ê^k−u^k‖²_M / Σ_k Σ‖u^k‖²_M)``.

        The final-time relative L2 is simply ``rel_l2_steps[-1]``."""
        self.model.eval()
        ns, R = self.n_seed_frames, self.rollout_steps
        M = self.problem.M
        total_mse, n = 0.0, 0
        num = torch.zeros(R)          # Σ_samples ‖ê^k − u^k‖²_M   per step
        den = torch.zeros(R)          # Σ_samples ‖u^k‖²_M          per step
        for trajs_grid, trajs_node in loader:
            trajs_grid = trajs_grid.to(self.device)
            preds_grid, seq_node = self._rollout(trajs_grid)
            ref_grid = trajs_grid[:, ns:ns + R]
            total_mse += ((preds_grid - ref_grid) ** 2).mean().item() * trajs_grid.shape[0]
            preds_node = seq_node[:, ns:ns + R]                          # [B, R, N]
            ref_node = trajs_node[:, ns:ns + R].to(self.device)          # [B, R, N]
            for k in range(R):
                e = preds_node[:, k] - ref_node[:, k]                    # [B, N]
                num[k] += (e * self.problem._spmm(M, e)).sum(-1).clamp_min(0).sum().item()
                u = ref_node[:, k]
                den[k] += (u * self.problem._spmm(M, u)).sum(-1).clamp_min(0).sum().item()
            n += trajs_grid.shape[0]
        rel_l2_steps = (num / den.clamp_min(1e-30)).sqrt().tolist()
        st_rel_l2 = float((num.sum() / den.sum().clamp_min(1e-30)).sqrt())
        return total_mse / n, rel_l2_steps, st_rel_l2

    def validate(self):
        mse, rel_l2_steps, st_rel_l2 = self._eval_full(self.val_loader)
        self.stats.val_rel_l2_steps.append(rel_l2_steps)
        self.stats.val_st_rel_l2.append(st_rel_l2)
        self.stats.val_final_rel_l2.append(rel_l2_steps[-1])
        # Model selection AND the per-epoch `val_errors` series track the space-time relative
        # FEM-L2 (the reported, scale-independent metric), not the scale-dependent rollout MSE.
        # (mse is still computed in the same pass but no longer drives checkpointing.)
        return st_rel_l2

    def test(self):
        return self._eval_full(self.test_loader)[0]

    def _save_results(self):
        """Persist per-epoch stats + config to a results JSON (the plottable record; the ``.pth``
        checkpoint also holds stats but is excluded from cross-run sync)."""
        self._save_results_json()

    def _save_results_json(self) -> None:
        """Dump stats + config to ``results/{prefix}.json`` for cross-run aggregation. The per-epoch
        validation metric (``stats.val_errors``, and the selection metric ``best_val_st_rel_l2``) is
        the space-time relative FEM-L2; ``stats.train_losses`` is the optimization-health signal."""
        import json
        from dataclasses import asdict
        test_mse, test_rel_l2_steps, test_st_rel_l2 = self._eval_full(self.test_loader)
        record = {
            "prefix": self._file_prefix(),
            "loss_type": self.loss_type,
            "ac_loss_form": getattr(self, "ac_loss_form", None),
            "ac_integrator": getattr(self, "ac_integrator", None),
            "ac_precond": getattr(self, "ac_precond", ""),
            "bptt_mode": self.bptt_mode,
            "lambda_galerkin": self.lambda_galerkin,
            "lambda_data": self.lambda_data,
            "a": getattr(self, "a", None), "eps": getattr(self, "eps", None),
            "dt": self.dt, "n_steps": self.n_steps, "rollout_steps": self.rollout_steps,
            "K": self.K, "n_train": len(self.train_dataset), "epochs": self.epochs,
            "best_val_st_rel_l2": self.stats.best_val_error, "best_epoch": self.stats.best_epoch,
            # test-set (best model): rollout MSE + FEM-L2 space-time / final / per-step series
            "test_mse": test_mse,
            "test_st_rel_l2": test_st_rel_l2,
            "test_final_rel_l2": test_rel_l2_steps[-1],
            "test_rel_l2_steps": test_rel_l2_steps,
            "stats": asdict(self.stats),
        }
        os.makedirs(f"{self.output_dir}/results", exist_ok=True)
        path = f"{self.output_dir}/results/{self._file_prefix()}.json"
        with open(path, "w") as fh:
            json.dump(record, fh)
        print(f"Results -> {path}")

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
                 rollout_steps=4, discount_factor=1.0, ac_integrator=None,
                 ac_loss_form="galerkin", bptt_mode=None, ac_precond="",
                 mg_levels=4, mg_pre_smooth=2, mg_post_smooth=2, mg_omega=2.0 / 3.0):
        super().__init__(model, train_dataset, val_dataset, test_dataset, loss_type,
                         n_seed_frames=1, optimizer_name=optimizer_name, lr=lr, lr_min=lr_min,
                         weight_decay=weight_decay, batch_size=batch_size, epochs=epochs,
                         device=device, output_dir=output_dir, lambda_galerkin=lambda_galerkin,
                         lambda_data=lambda_data, rollout_steps=rollout_steps, bptt_mode=bptt_mode)
        self.a = train_dataset.a
        self.eps = train_dataset.eps
        self.discount_factor = discount_factor
        # Default the physics-loss residual to the same integrator the reference data was built
        # with (kept coherent), unless explicitly overridden.
        self.ac_integrator = ac_integrator or getattr(train_dataset, "integrator", "backward_euler")
        self.ac_loss_form = ac_loss_form
        # bptt_mode resolves to the loss-side "coupling" detach (proximal centre / previous frame);
        # None keeps each loss's own default (MM detaches, Galerkin does not). detach_rollout (the
        # pushforward) is handled on the base trainer.
        detach_coupling = None if bptt_mode is None else (bptt_mode in ("detach_prev", "pushforward"))

        # Preconditioned least-squares: P ≈ J_0^{-1} with J_0 = a²A + cM the frozen (u²=1) Newton
        # Jacobian (c = 1/dt + 3ε²), realized as a multigrid V-cycle on a²A + cM. See
        # notes/ac_autoregressive/ §"Preconditioning the least-squares loss". Galerkin form only.
        self.ac_precond = ac_precond or ""
        self.precond: Optional[Preconditioner] = None
        if self.ac_precond:
            if ac_loss_form != "galerkin":
                raise ValueError("--ac_precond applies to the least-squares residual only "
                                 f"(--ac_loss_form galerkin), got {ac_loss_form!r}")
            if self.ac_precond != "multigrid":
                raise ValueError(f"unknown ac_precond {self.ac_precond!r}; expected 'multigrid'")
            a2 = self.a * self.a
            c = 1.0 / self.dt + 3.0 * self.eps * self.eps
            self.precond = build_preconditioner(
                kind="multigrid", problem=self.problem, grid_size=self.grid_size,
                mg_levels=mg_levels, mg_pre_smooth=mg_pre_smooth, mg_post_smooth=mg_post_smooth,
                mg_omega=mg_omega, mg_a2=a2, mg_c=c, device=device)
            print(f"[ac precond] multigrid V-cycle on a²A+cM  (a²={a2:g}, c=1/dt+3ε²={c:g})")

        self.criterion = build_ac_loss(self.problem, a=self.a, eps=self.eps, dt=self.dt,
                                       discount=discount_factor, integrator=self.ac_integrator,
                                       form=ac_loss_form, detach_coupling=detach_coupling,
                                       precond=self.precond)

    def _file_prefix(self) -> str:
        itag = {"convex_concave": "cc", "backward_euler": "be"}.get(self.ac_integrator, self.ac_integrator)
        ftag = {"galerkin": "ls", "min_movement": "mm"}.get(self.ac_loss_form, self.ac_loss_form)
        # Tag preconditioned LS so it never collides with the bare-LS run at the same config.
        ptag = "_precmg" if getattr(self, "ac_precond", "") else ""
        # Tag the bptt mode only when set, so existing (mode=None) run filenames are unchanged.
        btag = "" if self.bptt_mode is None else \
            "_bptt-" + {"full_bptt": "full", "detach_prev": "detach", "pushforward": "push"}.get(
                self.bptt_mode, self.bptt_mode)
        return (f"fno_ac_{self.loss_type}_{ftag}{ptag}_{itag}{btag}_a{self.a:g}_eps{self.eps:g}_dt{self.dt:g}_"
                f"T{self.n_steps}_R{self.rollout_steps}_K{self.K}_"
                f"samples-{len(self.train_dataset)}-{len(self.val_dataset)}-{len(self.test_dataset)}")


class StokesTrainer(BaseTrainer):
    r"""Trainer for the stationary Stokes saddle-point problem.

    The FNO maps the body force to the *pair* of fields: ``[B, 2, Ny, Nx]`` → ``[B, 3, Ny, Nx]``
    with channels ``(u_x, u_y, p)``. :meth:`~tensorpils.physics.StokesProblem.from_grid`
    reads velocity on the full fine grid and pressure on the ``[::2, ::2]`` Q1 subgrid.

    Every prediction — for *every* loss, at train and eval time — is passed through the two
    projections that encode known structure rather than learned structure:

    * zero Dirichlet velocity on ``∂Ω``;
    * the zero-mean pressure gauge, since the residual is blind to the constant mode
      (:math:`B^\top\mathbf 1 = 0`) and could never determine it.

    Evaluation reports FE :math:`L^2` errors against the discrete Taylor-Hood reference,
    separately for velocity and pressure — they live in different spaces on different grids,
    so a single grid MSE would be meaningless. Best-model selection uses the combined
    squared FE error.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        test_dataset,
        loss_type: str = "pls",
        optimizer_name: str = "adam",
        lr: float = 1e-3,
        lr_min: float = 1e-6,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        epochs: int = 500,
        device: str = "cuda",
        output_dir: str = "output",
        mg_levels: int = 4,
        mg_pre_smooth: int = 2,
        mg_post_smooth: int = 2,
        mg_omega: float = 2.0 / 3.0,
        schur_omega: float = 0.5,
        precond_strength: float = 1.0,
        precond_kind: str = "block",
        pls_form: str = "auto",
        uzawa_pre: int = 4,
        uzawa_post: int = 4,
        cheb_degree: int = 8,
        cheb_ratio: float = 30.0,
    ):
        super().__init__(model, loss_type, train_dataset.K, optimizer_name,
                         lr, lr_min, weight_decay, epochs, device, output_dir)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.problem = train_dataset.problem.to(device)
        self.grid_size = self.problem.grid_size
        self.mu = self.problem.mu
        self.f_scale = float(getattr(train_dataset, "f_scale", 1.0))
        self.p_scale = float(getattr(train_dataset, "p_scale", 1.0))
        self.u_l2_scale = float(getattr(train_dataset, "u_l2_scale", 1.0))
        self.p_l2_scale = float(getattr(train_dataset, "p_l2_scale", 1.0))
        self._test_label = "Test combined rel-L2"

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.precond: Optional[Preconditioner] = None
        self.schur_omega = schur_omega
        self.precond_strength = float(precond_strength)
        self.precond_kind = precond_kind
        self.mg_settings = (mg_levels, mg_pre_smooth, mg_post_smooth, mg_omega)
        self.uzawa_settings = (uzawa_pre, uzawa_post, cheb_degree, cheb_ratio)
        if loss_type == "pls":
            if precond_kind == "monolithic":
                # A genuine P ~ K^-1, which is what makes the *applied* form viable at O(1)
                # conditioning; the block preconditioner below is only norm-equivalent.
                self.precond = StokesMonolithicMultigrid(
                    self.problem, n_levels=mg_levels, n_pre=uzawa_pre, n_post=uzawa_post,
                    cheb_degree=cheb_degree, cheb_ratio=cheb_ratio,
                    schur_omega=schur_omega).to(device)
            elif precond_kind == "block":
                base = StokesBlockPreconditioner(
                    self.problem, mg_levels=mg_levels, mg_pre_smooth=mg_pre_smooth,
                    mg_post_smooth=mg_post_smooth, mg_omega=mg_omega,
                    schur_omega=schur_omega,
                    velocity_precond=self._velocity_precond(mg_pre_smooth, device)).to(device)
                # strength < 1 blends toward the identity: P_t = (1-t)*alpha*I + t*P_block, so t=0
                # reproduces the bare least-squares loss and t=1 the block preconditioner.
                self.precond = (base if self.precond_strength >= 1.0
                                else StokesBlendPreconditioner(
                                    base, self.precond_strength).to(device))
            else:
                raise ValueError(f"precond_kind must be 'block' or 'monolithic', "
                                 f"got {precond_kind!r}")

        # 'auto': the weighted norm for a norm-equivalent block P; for a monolithic approximate
        # inverse the applied form, in the FE metric — the plain nodal one is ~350x
        # pressure-dominated for this dataset (see StokesAppliedPLSLoss).
        self.pls_form = (("applied_fe" if precond_kind == "monolithic" else "weighted")
                         if pls_form == "auto" else pls_form)
        self.criterion = (None if loss_type == "data"
                          else build_stokes_loss(loss_type, self.problem, precond=self.precond,
                                                 form=self.pls_form,
                                                 u_scale=self.u_l2_scale, p_scale=self.p_l2_scale))
        os.makedirs(f"{self.output_dir}/results", exist_ok=True)

    def _velocity_precond(self, sweeps: int, device: str):
        """Operator for the block preconditioner's velocity half.

        ``None`` means "build the geometric V-cycle from the grid", which is the structured
        case. The unstructured subclass returns an algebraic V-cycle instead — that one
        substitution is the only part of the block preconditioner that was ever tied to a grid.
        """
        return None

    # -------------------- prediction --------------------
    def _predict(self, f_grid: torch.Tensor):
        """FNO forward → physical ``(u_node, p_node)``.

        Four fixed, non-trainable steps. Two are *scaling* (see ``StokesDataset``): the
        input is divided by ``f_scale`` and the pressure channel multiplied by ``p_scale``,
        so the network sees O(1) on every channel even though the physics ties the three
        magnitudes orders of magnitude apart. Two are *structural*: the velocity is
        projected to the zero Dirichlet boundary and the pressure to zero mean.
        """
        out = self.model(f_grid / self.f_scale)                    # [B, 3, Ny, Nx]
        u_node, p_node = self.problem.from_grid(out)
        return (self.problem.project_velocity_bc(u_node),
                self.problem.project_pressure_gauge(p_node * self.p_scale))

    def _data_loss(self, u_pred, p_pred, u_true, p_true) -> torch.Tensor:
        r"""Supervised FE-norm loss ``½(‖u−u*‖²_{M_u}/‖u‖² + ‖p−p*‖²_{M_p}/‖p‖²)``.

        Mass-weighted rather than a flat grid MSE because velocity and pressure live on
        different spaces and grids. Each term is divided by that field's mean squared FE
        norm over the dataset: the physics fixes ``‖p‖ ≈ 50‖u‖`` here, so an *unweighted*
        sum is ~2500x dominated by pressure and the velocity is effectively not optimized
        at all (measured: 85% velocity error while pressure reaches 9%).
        """
        eu = self.problem.velocity_l2(u_pred - u_true) ** 2 / self.u_l2_scale ** 2
        ep = self.problem.pressure_l2(p_pred - p_true) ** 2 / self.p_l2_scale ** 2
        return 0.5 * (eu + ep).mean()

    # -------------------- train / eval --------------------
    def train_epoch(self) -> float:
        self.model.train()
        total, nb = 0.0, 0
        for f_grid, f_node, u_true, p_true in self.train_loader:
            f_grid, f_node = f_grid.to(self.device), f_node.to(self.device)
            u_true, p_true = u_true.to(self.device), p_true.to(self.device)
            u_pred, p_pred = self._predict(f_grid)
            if self.loss_type == "data":
                loss = self._data_loss(u_pred, p_pred, u_true, p_true)
            else:
                loss = self.criterion(u_pred, p_pred, f_node)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            nb += 1
        return total / nb

    @torch.no_grad()
    def _eval_loader(self, loader):
        """Returns ``(sq_err, rel_l2_u, rel_l2_p)`` — combined squared FE error and the two
        dataset-level relative FE ``L2`` errors ``sqrt(sum ‖e‖² / sum ‖ref‖²)``."""
        self.model.eval()
        sq, n = 0.0, 0
        nu = du = np_ = dp = 0.0
        for f_grid, f_node, u_true, p_true in loader:
            f_grid = f_grid.to(self.device)
            u_true, p_true = u_true.to(self.device), p_true.to(self.device)
            u_pred, p_pred = self._predict(f_grid)
            eu = self.problem.velocity_l2(u_pred - u_true) ** 2          # [B]
            ep = self.problem.pressure_l2(p_pred - p_true) ** 2          # [B]
            sq += (eu + ep).sum().item()
            nu += eu.sum().item()
            du += (self.problem.velocity_l2(u_true) ** 2).sum().item()
            np_ += ep.sum().item()
            dp += (self.problem.pressure_l2(p_true) ** 2).sum().item()
            n += f_grid.shape[0]
        return (sq / n, math.sqrt(nu / max(du, 1e-30)), math.sqrt(np_ / max(dp, 1e-30)))

    def validate(self):
        sq, ru, rp = self._eval_loader(self.val_loader)
        self.stats.val_rel_l2_u.append(ru)
        self.stats.val_rel_l2_p.append(rp)
        self.stats.val_l2_errors.append(sq)
        self.stats.val_rel_l2_errors.append(0.5 * (ru + rp))
        return 0.5 * (ru + rp)

    def test(self):
        _, ru, rp = self._eval_loader(self.test_loader)
        return 0.5 * (ru + rp)

    def _before_train(self):
        if self.precond is not None:
            self.precond.report()

    def _save_results(self):
        sq, ru, rp = self._eval_loader(self.test_loader)
        print(f"Test rel-L2 — velocity: {ru:.2%}   pressure: {rp:.2%}   "
              f"(combined FE-L2 squared error {sq:.3e})")
        self._save_results_json((sq, ru, rp))

    def _save_results_json(self, test_metrics) -> None:
        import json
        from dataclasses import asdict
        sq, ru, rp = test_metrics
        record = {
            "prefix": self._file_prefix(),
            "pde": "stokes",
            "loss_type": self.loss_type,
            "mu": self.mu,
            "schur_omega": self.schur_omega,
            "K": self.K,
            # None on an unstructured mesh; the node counts below are what identify it.
            "grid": list(self.grid_size) if self.grid_size else None,
            "pgrid": (list(self.problem.pgrid_size)
                      if self.problem.pgrid_size else None),
            "n_u": self.problem.n_u, "n_p": self.problem.n_p,
            "mesh": getattr(self, "mesh_tag", "structured"),
            "n_train": len(self.train_dataset),
            "epochs": self.epochs,
            "precond_strength": self.precond_strength if self.loss_type == "pls" else None,
            "precond_alpha": getattr(self.precond, "alpha", None),
            "precond_kind": self.precond_kind if self.loss_type == "pls" else None,
            "pls_form": self.pls_form if self.loss_type == "pls" else None,
            "uzawa": list(self.uzawa_settings) if self.precond_kind == "monolithic" else None,
            "test_sq_fe_l2": sq, "test_rel_l2_u": ru, "test_rel_l2_p": rp,
            "best_val_error": self.stats.best_val_error, "best_epoch": self.stats.best_epoch,
            "stats": asdict(self.stats),
        }
        path = f"{self.output_dir}/results/{self._file_prefix()}.json"
        with open(path, "w") as fh:
            json.dump(record, fh)
        print(f"Results -> {path}")

    # -------------------- naming / viz --------------------
    def _file_prefix(self) -> str:
        if self.loss_type == "pls" and self.precond_kind == "monolithic":
            L, _, _, _ = self.mg_settings
            nu1, nu2, cd, _ = self.uzawa_settings
            tag = f"_mono-L{L}-u{nu1}{nu2}-c{cd}-w{self.schur_omega:g}"
            if self.pls_form != "applied":
                tag += f"-{self.pls_form}"
        elif self.loss_type == "pls":
            L, pre, post, _ = self.mg_settings
            tag = f"_mg-L{L}-s{pre}{post}-w{self.schur_omega:g}"
            if self.precond_strength < 1.0:
                tag += f"-t{self.precond_strength:g}"
            if self.pls_form != "weighted":
                tag += f"-{self.pls_form}"
        else:
            tag = ""
        nx, _ = self.grid_size
        return (f"fno_stokes_{self.loss_type}{tag}_mu{self.mu:g}_gr{nx}_K{self.K}_"
                f"samples-{len(self.train_dataset)}-{len(self.val_dataset)}-{len(self.test_dataset)}")

    @torch.no_grad()
    def _predict_eval(self, f_grid: torch.Tensor):
        """``_predict`` in eval mode — the single prediction path handed to ``viz``."""
        self.model.eval()
        return self._predict(f_grid)

    def visualize_sample(self, dataset, split: str, sample_idx: int = 0):
        suffix = "" if sample_idx == 0 else f"_sample{sample_idx}"
        viz.visualize_stokes_sample(
            self._predict_eval, dataset, self.problem, self.device, sample_idx=sample_idx,
            save_path=f"{self.output_dir}/visualization/{self._file_prefix()}_{split}{suffix}.png")

    def compute_error_distribution(self):
        return viz.compute_stokes_error_distribution(
            self._predict_eval, self.test_dataset, self.problem, self.device,
            save_path=f"{self.output_dir}/error/{self._file_prefix()}_error_dist.png")

    def _after_train_viz(self):
        self.visualize_sample(self.test_dataset, "test", 0)
        self.compute_error_distribution()


# ================================================================= architecture variants
# GAOT is a second *production* architecture, not a baseline: it runs the same losses, the same
# validation and the same test metric as the FNO, and differs only in what maps f to u. These
# subclasses exist solely so its runs get their own file prefix -- see ``ArchPrefixMixin``.


class GAOTPoissonTrainer(ArchPrefixMixin, PoissonTrainer):
    """``PoissonTrainer`` on a :class:`~tensorpils.gaot.GAOTModel`.

    Nothing else changes, and that is the point of the wrapper's grid signature: the FEM losses
    receive node values obtained from the model output by the same ``grid_to_node`` call as for
    the FNO, so a GAOT-vs-FNO comparison isolates the architecture.
    """


class GAOTWaveTrainer(ArchPrefixMixin, WaveTrainer):
    """``WaveTrainer`` counterpart (autoregressive wave time-stepper)."""


class GAOTACTrainer(ArchPrefixMixin, ACTrainer):
    """``ACTrainer`` counterpart of :class:`GAOTPoissonTrainer` (autoregressive Allen-Cahn)."""


class GAOTStokesTrainer(ArchPrefixMixin, StokesTrainer):
    """``StokesTrainer`` counterpart: GAOT emits the 3-channel ``(u_x, u_y, p)`` grid the
    Taylor-Hood plumbing already expects, so ``_predict`` and the evaluation are inherited."""


# ============================================================== unstructured (no grid at all)


class UnstructuredPoissonTrainer(PoissonTrainer):
    r"""Poisson on an unstructured mesh: everything in node space, no image anywhere.

    Subclasses :class:`PoissonTrainer` rather than replacing it, so the optimizer, the cosine
    schedule, model selection, checkpointing, early stopping and the results JSON are the
    *same code* as the structured runs — an unstructured number is therefore produced by the
    same evaluation as a structured one, and the two are comparable.

    Four things are overridden, and each is exactly the place the grid was assumed:

    * ``_collate`` / ``train_epoch`` / ``_forward`` — items are ``(f_node, u_node)`` and the
      model is called through ``forward_nodes``; there is no ``[B, C, H, W]`` to build.
    * ``_project_zero_bc`` — the boundary is the mesh's ``boundary_mask``, not an outer frame.
    * ``_eval_loader`` — model selection is node MSE, which on a uniform grid is *numerically*
      the grid MSE the structured runs select on. The FEM L2 (``sqrt(e^T M e)``) and relative
      L2 alongside it were already mesh-agnostic and are untouched.
    * ``_file_prefix`` — carries a mesh tag, so an unstructured run cannot land on a structured
      run's checkpoint.

    The preconditioner is the other half of this. ``GeometricMultigrid`` re-discretises a
    structured hierarchy and has no grid to do it on here — it builds without complaint and
    then dies on a shape mismatch at the first apply — so the ``pls`` arm needs
    ``--precond_kind amg``, which is assembled from the matrix. :mod:`tensorpils.cli` enforces
    that rather than letting the job discover it after the queue wait.
    """

    #: Tag that goes in the run file name, so structured/unstructured runs never collide.
    mesh_tag = "circle"

    def __init__(self, *args, mesh_tag: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if mesh_tag is not None:
            self.mesh_tag = mesh_tag
        self.n_nodes = self.train_dataset.n_nodes
        # PoissonTrainer builds the criterion from build_loss(); `data` is the one loss defined
        # on the image, so swap in its node counterpart.
        if self.loss_type == "data":
            self.criterion = build_loss(self.loss_type, self.problem, lambda_bc=0.0,
                                        bc_mode=self.bc_mode, node_form=True)

    # -------------------- node-space plumbing --------------------
    @staticmethod
    def _collate(batch):
        fs = torch.stack([item[0] for item in batch], dim=0)        # [B, N]
        us = torch.stack([item[1] for item in batch], dim=0)        # [B, N]
        return fs, us

    def _project_zero_bc(self, u_node: torch.Tensor) -> torch.Tensor:
        """Zero the mesh's boundary nodes. ``[B, N]`` in, ``[B, N]`` out."""
        return apply_zero_boundary(u_node, self.problem.boundary_mask.to(u_node.device))

    def _predict(self, fs_node: torch.Tensor) -> torch.Tensor:
        """``[B, N]`` source -> ``[B, N]`` prediction, through the point-cloud entry point."""
        return self.model.forward_nodes(fs_node.unsqueeze(-1))[..., 0]

    def _forward(self, fs_node, us_node):
        u_pred = self._predict(fs_node)
        if self.loss_type == "data":
            return self.criterion(u_pred, us_node)
        return self.criterion(u_pred, fs_node, us_node)

    def train_epoch(self) -> float:
        self.model.train()
        total, nb = 0.0, 0
        for fs, us in self.train_loader:
            self.optimizer.zero_grad()
            fs, us = fs.to(self.device), us.to(self.device)
            loss = self._forward(fs, us)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            nb += 1
        return total / nb

    @torch.no_grad()
    def _eval_loader(self, loader):
        """``(node MSE, FEM L2, relative L2)`` — the structured tuple, read off nodes."""
        self.model.eval()
        total_mse, total_l2, total_rel_l2, n = 0.0, 0.0, 0.0, 0
        for fs, us in loader:
            fs, us = fs.to(self.device), us.to(self.device)
            u_pred = self._apply_eval_bc(self._predict(fs))
            b = fs.shape[0]
            total_mse += ((u_pred - us) ** 2).mean().item() * b
            e = u_pred - us
            Me = self.problem._spmm(self.problem.M, e)
            l2 = (e * Me).sum(dim=1).clamp(min=0).sqrt()
            total_l2 += l2.sum().item()
            Mu = self.problem._spmm(self.problem.M, us)
            u_norm = (us * Mu).sum(dim=1).clamp(min=0).sqrt()
            total_rel_l2 += (l2 / u_norm.clamp(min=1e-12)).sum().item()
            n += b
        return total_mse / n, total_l2 / n, total_rel_l2 / n

    # -------------------- bookkeeping --------------------
    def _file_prefix(self) -> str:
        return f"{super()._file_prefix()}_{self.mesh_tag}-n{self.n_nodes}"

    def visualize_sample(self, dataset, split: str, sample_idx: int = 0):
        suffix = "" if sample_idx == 0 else f"{sample_idx}"
        viz.visualize_unstructured_sample(
            self._predict, dataset, self.device, self._apply_eval_bc, sample_idx,
            save_path=f"{self.output_dir}/visualization/{self._file_prefix()}_{split}{suffix}.png")

    def compute_error_distribution(self):
        return viz.compute_unstructured_error_distribution(
            self._predict, self.test_dataset, self.problem, self.device, self._apply_eval_bc,
            save_path=f"{self.output_dir}/error/{self._file_prefix()}_error_dist.png")


class GAOTUnstructuredPoissonTrainer(ArchPrefixMixin, UnstructuredPoissonTrainer):
    """The unstructured arm as it is actually run: GAOT is the only architecture here that can
    consume a point cloud, but the prefix tag is kept general so a future one drops in."""


class UnstructuredStokesTrainer(StokesTrainer):
    r"""Taylor-Hood Stokes on an unstructured mesh — no image on either side.

    Subclasses :class:`StokesTrainer` for the same reason the Poisson one does: the optimizer,
    the schedule, model selection, checkpointing, the results JSON and — decisively — the
    **evaluation** (per-field relative FE ``L²``, selected on their mean) are the same code as
    the structured runs, so an unstructured Stokes number is produced by exactly the evaluation
    that produced the structured table.

    What had to change is small and all of it was grid:

    * ``_predict`` — the model is called through ``forward_nodes`` on the P2 node set and emits
      three channels there; pressure is gathered at the corner nodes by
      :meth:`~tensorpils.physics.UnstructuredStokesProblem.from_nodes` instead of read off the
      ``[::2, ::2]`` subgrid. The two scalings and the two projections are untouched.
    * ``_velocity_precond`` — an **algebraic** V-cycle on the P2 scalar stiffness replaces the
      geometric one. This is the only part of the block preconditioner
      ``P = diag(Â⁻¹/μ, ω μ/diag(M_p))`` that was ever tied to a grid: the pressure half is a
      lumped diagonal and was always mesh-agnostic.
    * the loaders' item is ``(f, u, p)`` rather than ``(f_grid, f, u, p)``.

    ``--stokes_precond monolithic`` is **not** available here. That preconditioner's transfer
    operators are geometric (nested grids for both fields simultaneously), and an algebraic
    monolithic saddle-point preconditioner is a research question rather than a port. The block
    form is the arm the structured table's label-free number came from, and it is the one that
    carries over.
    """

    mesh_tag = "obstacle"

    def __init__(self, *args, mesh_tag: Optional[str] = None, **kwargs):
        if kwargs.get("precond_kind") == "monolithic":
            raise SystemExit(
                "--stokes_precond monolithic has no unstructured form: its transfer operators "
                "are geometric (both fields nesting by two on a grid). Use --stokes_precond "
                "block, whose velocity half becomes an algebraic V-cycle.")
        super().__init__(*args, **kwargs)
        if mesh_tag is not None:
            self.mesh_tag = mesh_tag
        self.n_nodes = self.problem.n_u

    def _velocity_precond(self, sweeps: int, device: str):
        """Algebraic V-cycle on the P2 scalar stiffness — the velocity block up to ``mu``."""
        from .preconditioners import AMGXPreconditioner
        return AMGXPreconditioner(self.problem.A, self.problem.boundary_mask,
                                  sweeps=sweeps, device=(device or "cuda:0"))

    # -------------------- prediction --------------------
    def _predict(self, f_node: torch.Tensor):
        """``[B, n_u, 2]`` body force → physical ``(u_node [B, n_u, 2], p_node [B, n_p])``.

        The same four fixed steps as the structured trainer — input scaling, pressure scaling,
        the velocity boundary projection and the pressure gauge — with the grid reshape
        replaced by a gather. Nothing here is trainable.
        """
        out = self.model.forward_nodes(f_node / self.f_scale)          # [B, n_u, 3]
        u_node, p_node = self.problem.from_nodes(out)
        return (self.problem.project_velocity_bc(u_node),
                self.problem.project_pressure_gauge(p_node * self.p_scale))

    # -------------------- train / eval --------------------
    def train_epoch(self) -> float:
        self.model.train()
        total, nb = 0.0, 0
        for f_node, u_true, p_true in self.train_loader:
            f_node = f_node.to(self.device)
            u_true, p_true = u_true.to(self.device), p_true.to(self.device)
            u_pred, p_pred = self._predict(f_node)
            loss = (self._data_loss(u_pred, p_pred, u_true, p_true) if self.loss_type == "data"
                    else self.criterion(u_pred, p_pred, f_node))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            nb += 1
        return total / nb

    @torch.no_grad()
    def _eval_loader(self, loader):
        """``(sq_err, rel_l2_u, rel_l2_p)`` — the structured tuple, read off nodes."""
        self.model.eval()
        sq, n = 0.0, 0
        nu = du = np_ = dp = 0.0
        for f_node, u_true, p_true in loader:
            f_node = f_node.to(self.device)
            u_true, p_true = u_true.to(self.device), p_true.to(self.device)
            u_pred, p_pred = self._predict(f_node)
            eu = self.problem.velocity_l2(u_pred - u_true) ** 2
            ep = self.problem.pressure_l2(p_pred - p_true) ** 2
            sq += (eu + ep).sum().item()
            nu += eu.sum().item()
            du += (self.problem.velocity_l2(u_true) ** 2).sum().item()
            np_ += ep.sum().item()
            dp += (self.problem.pressure_l2(p_true) ** 2).sum().item()
            n += f_node.shape[0]
        return (sq / n, math.sqrt(nu / max(du, 1e-30)), math.sqrt(np_ / max(dp, 1e-30)))

    # -------------------- bookkeeping --------------------
    def _file_prefix(self) -> str:
        tag = ""
        if self.loss_type == "pls":
            _, pre, _, _ = self.mg_settings
            tag = f"_amg-s{pre}-w{self.schur_omega:g}"
            if self.pls_form != "weighted":
                tag += f"-{self.pls_form}"
        return (f"fno_stokes_{self.loss_type}{tag}_mu{self.mu:g}_"
                f"{self.mesh_tag}-n{self.n_nodes}_K{self.K}_"
                f"samples-{len(self.train_dataset)}-{len(self.val_dataset)}-{len(self.test_dataset)}")

    def visualize_sample(self, dataset, split: str, sample_idx: int = 0):
        suffix = "" if sample_idx == 0 else f"{sample_idx}"
        viz.visualize_unstructured_stokes_sample(
            self._predict, dataset, self.problem, self.device, sample_idx,
            save_path=f"{self.output_dir}/visualization/{self._file_prefix()}_{split}{suffix}.png")

    def compute_error_distribution(self):
        return viz.compute_unstructured_stokes_error_distribution(
            self._predict, self.test_dataset, self.problem, self.device,
            save_path=f"{self.output_dir}/error/{self._file_prefix()}_error_dist.png")


class GAOTUnstructuredStokesTrainer(ArchPrefixMixin, UnstructuredStokesTrainer):
    """The unstructured Stokes arm as it is run: GAOT is the only architecture here that can
    consume a point cloud and emit three fields on it."""
