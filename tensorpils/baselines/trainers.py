r"""Trainers for the two baselines.

Each one **subclasses the production trainer** rather than modifying it, so ``trainer.py``,
``losses.py`` and ``physics.py`` are untouched by this branch. The consequence that matters
scientifically: validation, model selection, checkpointing and the final test metric all run
through the inherited code paths, so every baseline number is produced by *exactly* the same
evaluation as the numbers already in the paper — FEM relative ``L²`` with the eval-time
boundary projection.

Only the training objective is overridden. Where a base ``__init__`` insists on building one
of the registered losses, it is given a benign ``loss_type`` and the criterion is replaced
immediately afterwards; ``self.loss_type`` is then set to the baseline's own name so file
prefixes and the results JSON record what actually ran.
"""

from typing import Optional

import torch

from ..meshing import grid_to_node, node_to_grid
from ..trainer import PoissonTrainer, ACTrainer
from .pino import PINOPoissonLoss, PINOACLoss
from .pi_deeponet import PIDeepONetPoissonLoss, PIDeepONetACLoss, autodiff_laplacian

__all__ = ["PINOPoissonTrainer", "PINOACTrainer",
           "PIDeepONetPoissonTrainer", "PIDeepONetACTrainer",
           "DeepONetPoissonTrainer", "DeepONetACTrainer"]


def _arch_tag(model) -> str:
    """``fno`` / ``deeponet`` for the run file name, seen through any wrapper."""
    inner = getattr(model, "model", model)
    return "deeponet" if type(inner).__name__ == "DeepONetModel" else "fno"


def _detach_coupling(bptt_mode):
    """The loss-side coupling detach, resolved exactly as ``ACTrainer`` resolves it.

    Copied deliberately rather than approximated: the FEM Allen–Cahn arm detaches the previous
    frame inside the residual under ``detach_prev``/``pushforward``. A baseline that did not
    would differ from it in what the gradient flows through *as well as* in the residual, and a
    win or loss could no longer be attributed to the residual alone.
    """
    if bptt_mode is None:
        return None
    return bptt_mode in ("detach_prev", "pushforward")


class _ArchPrefixMixin:
    """Put the architecture in the run file name.

    The production ``_file_prefix`` hard-codes ``fno_``, which was unambiguous while the FNO
    was the only architecture. Now that any loss can be paired with a DeepONet, a
    ``--model deeponet --loss data`` run would land on the *same* checkpoint / results path as
    the FNO run of the same config and silently overwrite it. Rewriting the prefix here keeps
    ``trainer.py`` untouched and leaves every existing FNO filename byte-identical.
    """

    def _file_prefix(self) -> str:
        prefix = super()._file_prefix()
        tag = _arch_tag(self.model)
        if tag == "fno":
            return prefix
        return tag + prefix[3:] if prefix.startswith("fno") else f"{tag}_{prefix}"


class DeepONetPoissonTrainer(_ArchPrefixMixin, PoissonTrainer):
    """``PoissonTrainer`` with an architecture-tagged file prefix — the "our loss, other
    architecture" cells (supervised ``data`` and preconditioned ``pls`` on a DeepONet)."""


class DeepONetACTrainer(_ArchPrefixMixin, ACTrainer):
    """``ACTrainer`` counterpart of :class:`DeepONetPoissonTrainer`."""


# =============================================================================== PINO

class PINOPoissonTrainer(PoissonTrainer):
    """Poisson with the PINO finite-difference strong-form residual.

    The model is expected to be wrapped so that the zero Dirichlet BC is imposed by the *model*
    (the CLI does this): :class:`~tensorpils.baselines.pino.ZeroBoundaryModel` by default, which
    zeroes the boundary nodes exactly as ``losses.py`` does to our own arms, or
    :class:`~tensorpils.baselines.pino.MollifiedModel` under ``--pino_bc mollifier`` to reproduce
    the reference. Either way ``eval_project_bc`` has nothing left to do and the prediction fed to
    the loss is the same one that is evaluated.
    """

    def __init__(self, *args, pino_reduction: str = "rel", **kwargs):
        # PoissonTrainer.__init__ builds a registered loss; 'data' is inert and costs nothing,
        # and it also keeps needs_precond False. The real criterion is installed below.
        kwargs["loss_type"] = "data"
        super().__init__(*args, **kwargs)
        self.loss_type = "pino"
        self.pino_reduction = pino_reduction
        self.criterion = PINOPoissonLoss(self.grid_size, reduction=pino_reduction)

    def _forward(self, fs_grid, us_grid, fs_node, us_node):
        u_pred_grid = self.model(fs_grid).squeeze(1)                     # [B, H, W]
        return self.criterion(u_pred_grid, fs_grid)                      # labels never touched

    def _file_prefix(self) -> str:
        return (f"{_arch_tag(self.model)}_pino-{self.pino_reduction}_K{self.K}_"
                f"samples-{len(self.train_dataset)}-{len(self.val_dataset)}-"
                f"{len(self.test_dataset)}")


class PINOACTrainer(ACTrainer):
    """Allen–Cahn with the PINO finite-difference strong-form step residual.

    Reuses the inherited autoregressive rollout unchanged — only the residual differs — so
    the rollout length, the boundary projection between steps and the BPTT mode all behave
    exactly as in the FEM arms.
    """

    def __init__(self, *args, **kwargs):
        kwargs["loss_type"] = "galerkin"
        super().__init__(*args, **kwargs)
        self.loss_type = "pino"
        self.criterion = PINOACLoss(
            self.grid_size, a=self.a, eps=self.eps, dt=self.dt,
            discount=self.discount_factor, integrator=self.ac_integrator,
            detach_coupling=_detach_coupling(self.bptt_mode),
        )

    def train_epoch(self) -> float:
        self.model.train()
        total, nb = 0.0, 0
        ns, R = self.n_seed_frames, self.rollout_steps
        for trajs_grid, _ in self.train_loader:
            self.optimizer.zero_grad()
            trajs_grid = trajs_grid.to(self.device)
            preds_grid, _ = self._rollout(trajs_grid)                    # [B, R, H, W]
            # The FD residual needs the grid sequence, where the FEM one needed node order.
            # Project frame by frame: apply_zero_boundary is defined for [N] / [B, N] only.
            seed = torch.stack([self._project_zero_bc(trajs_grid[:, i])
                                for i in range(ns)], dim=1)              # [B, ns, H, W]
            seq_grid = torch.cat([seed, preds_grid], dim=1)              # [B, ns+R, H, W]
            loss = preds_grid.new_zeros(())
            if self.lambda_galerkin != 0.0:          # never evaluate an unused residual: 0 * NaN = NaN
                loss = loss + self.lambda_galerkin * self.criterion(seq_grid)
            if self.lambda_data > 0.0:
                ref = trajs_grid[:, ns:ns + R]
                loss = loss + self.lambda_data * ((preds_grid - ref) ** 2).mean()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            nb += 1
        return total / nb

    def _file_prefix(self) -> str:
        itag = {"convex_concave": "cc", "backward_euler": "be"}.get(
            self.ac_integrator, self.ac_integrator)
        btag = "" if self.bptt_mode is None else "_bptt-" + {
            "full_bptt": "full", "detach_prev": "detach", "pushforward": "push",
        }.get(self.bptt_mode, self.bptt_mode)
        return (f"{_arch_tag(self.model)}_ac_pino_{itag}{btag}_a{self.a:g}_eps{self.eps:g}_"
                f"dt{self.dt:g}_T{self.n_steps}_R{self.rollout_steps}_K{self.K}_"
                f"samples-{len(self.train_dataset)}-{len(self.val_dataset)}-"
                f"{len(self.test_dataset)}")


# ========================================================================= PI-DeepONet

class PIDeepONetPoissonTrainer(PoissonTrainer):
    r"""Poisson with the strong-form residual differentiated by autodiff through the trunk.

    Collocation points are the grid nodes, optionally subsampled to ``n_colloc`` fresh
    random nodes per step (the stochastic-collocation regime PI-DeepONet is usually trained
    in, at a fraction of the cost of all ``N``). Node order is row-major, matching both
    ``DeepONetModel.grid_coords`` and the dataset's ``f_node``, so ``f`` at the collocation
    points is a plain index into the batch — no interpolation anywhere.
    """

    def __init__(self, *args, n_colloc: int = 0, pi_reduction: str = "rel",
                 lambda_bc_pi: float = 0.0, **kwargs):
        kwargs["loss_type"] = "data"
        super().__init__(*args, **kwargs)
        self.loss_type = "pideeponet"
        self.n_colloc = n_colloc
        self.pi_reduction = pi_reduction
        self.criterion = PIDeepONetPoissonLoss(reduction=pi_reduction, p=2,
                                               lambda_bc=lambda_bc_pi)
        if not hasattr(self.model, "forward_at"):
            raise TypeError("--loss pideeponet needs a coordinate-queryable model "
                            "(--model deeponet)")
        self._interior_idx = (~self.problem.boundary_mask).nonzero(as_tuple=False).squeeze(1)
        self._interior_idx = self._interior_idx.to(self.model.grid_coords.device)

    def _colloc(self, batch_size: int):
        """``(coords [B, Q, 2] requiring grad, node index [Q])``.

        Restricted to **interior** nodes, matching ``PINOPoissonLoss``'s interior-only slice
        (and the reference ``FDM_Darcy``). Averaging over different node sets would make the
        two baselines' losses differ by the ratio of node counts alone, so their learning
        rates would not be comparable.
        """
        idx = self._interior_idx
        if self.n_colloc and self.n_colloc < idx.numel():
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:self.n_colloc]]
        coords = self.model.grid_coords[idx]                              # [Q, 2]
        c = coords.unsqueeze(0).expand(batch_size, -1, -1).clone().requires_grad_(True)
        return c, idx

    def train_epoch(self) -> float:
        self.model.train()
        total, nb = 0.0, 0
        for batch_data, _ in self.train_loader:
            self.optimizer.zero_grad()
            fs_grid, _, fs_node, _ = batch_data
            fs_grid, fs_node = fs_grid.to(self.device), fs_node.to(self.device)
            coords, idx = self._colloc(fs_grid.shape[0])
            f_at = fs_node[:, idx]                                         # [B, Q]
            loss = self.criterion(self.model, fs_grid, coords, f_at)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            nb += 1
        return total / nb

    def _file_prefix(self) -> str:
        ctag = "" if not self.n_colloc else f"-c{self.n_colloc}"
        return (f"deeponet_pi-{self.pi_reduction}{ctag}_K{self.K}_"
                f"samples-{len(self.train_dataset)}-{len(self.val_dataset)}-"
                f"{len(self.test_dataset)}")


class PIDeepONetACTrainer(ACTrainer):
    r"""Allen–Cahn with the autodiff strong-form step residual.

    The rollout is re-implemented (rather than inherited) for one reason: a single
    coordinate-space model evaluation yields *both* the next frame and its Laplacian, so the
    residual costs no extra forward pass. Evaluation still goes through the inherited
    ``_eval_full``/``_rollout``, which call the grid ``forward`` — so the reported metric is
    produced identically to every other arm.

    The model must be built with ``mollify=True`` (the CLI default here): the predicted node
    values then already satisfy the zero Dirichlet BC, so the values entering the residual
    are the same ones that are fed back and scored, with no projection step in between.
    """

    def __init__(self, *args, n_colloc: int = 1024, **kwargs):
        kwargs["loss_type"] = "galerkin"
        super().__init__(*args, **kwargs)
        self.loss_type = "pideeponet"
        self.n_colloc = n_colloc
        # No interior_mask here: the collocation indices are already restricted to interior
        # nodes (see _rollout_autodiff), so the residual is averaged over the same node set
        # PINOACLoss's [1:-1, 1:-1] slice uses and the two losses stay on one scale.
        self.criterion = PIDeepONetACLoss(
            a=self.a, eps=self.eps, dt=self.dt,
            discount=self.discount_factor, integrator=self.ac_integrator,
            detach_coupling=_detach_coupling(self.bptt_mode),
        )
        self._interior_idx = (~self.problem.boundary_mask).nonzero(
            as_tuple=False).squeeze(1).to(self.model.grid_coords.device)
        if not hasattr(self.model, "forward_at"):
            raise TypeError("--loss pideeponet needs a coordinate-queryable model "
                            "(--model deeponet)")
        if not getattr(self.model, "mollify", False):
            raise ValueError("PI-DeepONet Allen–Cahn requires mollify=True so the rollout "
                             "values and the residual values coincide on the boundary")

    def _rollout_autodiff(self, traj_grid: torch.Tensor):
        r"""``(seq_sub [B, 1+R, Q], laps [B, R, Q], preds_grid [B, R, H, W])``.

        Two evaluations per step, deliberately:

        * the **full grid** without coordinate-grad, whose values are fed back into the rollout
          and scored;
        * a **subsample of ``n_colloc`` interior nodes** with coordinate-grad, which is where
          the Laplacian is taken and the residual is formed.

        Splitting them is what makes this arm fit in memory. Taking the Laplacian over all
        ``N`` nodes retains a double-backward graph per rollout step -- four of them at
        ``rollout_steps=4`` -- and at ``64²`` that reliably exhausts a 24 GB card (measured:
        OOM on all three seeds). The values are identical either way, since both passes
        evaluate the same model at the same coordinates; only the retained graph shrinks.
        Fresh indices each step make this the stochastic-collocation regime PI-DeepONet is
        normally trained in.
        """
        nx, ny = self.grid_size
        b = traj_grid.shape[0]
        idx = self._interior_idx
        if self.n_colloc and self.n_colloc < idx.numel():
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:self.n_colloc]]
        coords0 = self.model.grid_coords[idx]                              # [Q, 2]

        frame = self._project_zero_bc(traj_grid[:, 0])                     # [B, H, W]
        seq_sub = [grid_to_node(frame, nx, ny)[:, idx]]                    # [B, Q]
        laps, preds_grid = [], []
        for _ in range(self.rollout_steps):
            inp = frame.unsqueeze(1)                                       # [B, 1, H, W]
            u_full = self.model.forward_at(inp, self.model.grid_coords)    # [B, N], no coord grad
            coords = coords0.unsqueeze(0).expand(b, -1, -1).clone().requires_grad_(True)
            u_sub, lap = autodiff_laplacian(self.model, inp, coords)       # [B, Q] each
            seq_sub.append(u_sub)
            laps.append(lap)
            nxt = node_to_grid(u_full, nx, ny)                             # [B, H, W]
            preds_grid.append(nxt)
            frame = nxt.detach() if self.detach_rollout else nxt
        return (torch.stack(seq_sub, dim=1), torch.stack(laps, dim=1),
                torch.stack(preds_grid, dim=1))

    def train_epoch(self) -> float:
        self.model.train()
        total, nb = 0.0, 0
        ns, R = self.n_seed_frames, self.rollout_steps
        for trajs_grid, _ in self.train_loader:
            self.optimizer.zero_grad()
            trajs_grid = trajs_grid.to(self.device)
            seq_node, laps, preds_grid = self._rollout_autodiff(trajs_grid)
            loss = preds_grid.new_zeros(())
            if self.lambda_galerkin != 0.0:          # never evaluate an unused residual: 0 * NaN = NaN
                loss = loss + self.lambda_galerkin * self.criterion(seq_node, laps)
            if self.lambda_data > 0.0:
                ref = trajs_grid[:, ns:ns + R]
                loss = loss + self.lambda_data * ((preds_grid - ref) ** 2).mean()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            nb += 1
        return total / nb

    def _file_prefix(self) -> str:
        itag = {"convex_concave": "cc", "backward_euler": "be"}.get(
            self.ac_integrator, self.ac_integrator)
        btag = "" if self.bptt_mode is None else "_bptt-" + {
            "full_bptt": "full", "detach_prev": "detach", "pushforward": "push",
        }.get(self.bptt_mode, self.bptt_mode)
        return (f"deeponet_ac_pi_{itag}{btag}_a{self.a:g}_eps{self.eps:g}_dt{self.dt:g}_"
                f"T{self.n_steps}_R{self.rollout_steps}_K{self.K}_"
                f"samples-{len(self.train_dataset)}-{len(self.val_dataset)}-"
                f"{len(self.test_dataset)}")
