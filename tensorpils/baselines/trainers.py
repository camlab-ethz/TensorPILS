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
from ..trainer import (PoissonTrainer, ACTrainer, StokesTrainer,
                       arch_tag as _arch_tag, ArchPrefixMixin as _ArchPrefixMixin)
from .pino import PINOPoissonLoss, PINOACLoss, PINOStokesLoss, boundary_mask_grid
from .pi_deeponet import (PIDeepONetPoissonLoss, PIDeepONetACLoss, PIDeepONetStokesLoss,
                          autodiff_laplacian)

__all__ = ["PINOPoissonTrainer", "PINOACTrainer", "PINOStokesTrainer",
           "PIDeepONetPoissonTrainer", "PIDeepONetACTrainer", "PIDeepONetStokesTrainer",
           "DeepONetPoissonTrainer", "DeepONetACTrainer", "DeepONetStokesTrainer"]


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


def _interior_colloc(model, interior_idx: torch.Tensor, n_colloc: int, batch_size: int):
    """``(coords [B, Q, 2] requiring grad, node index [Q])`` over the interior grid nodes,
    optionally a fresh random subsample of ``n_colloc`` of them.

    Restricted to **interior** nodes, matching the PINO losses' interior-only slice (and the
    reference ``FDM_Darcy``). Averaging over different node sets would make the two baselines'
    losses differ by the ratio of node counts alone, so their learning rates would not be
    comparable.
    """
    idx = interior_idx
    if n_colloc and n_colloc < idx.numel():
        idx = idx[torch.randperm(idx.numel(), device=idx.device)[:n_colloc]]
    coords = model.grid_coords[idx]                                          # [Q, 2]
    c = coords.unsqueeze(0).expand(batch_size, -1, -1).clone().requires_grad_(True)
    return c, idx


class DeepONetPoissonTrainer(_ArchPrefixMixin, PoissonTrainer):
    """``PoissonTrainer`` with an architecture-tagged file prefix — the "our loss, other
    architecture" cells (supervised ``data`` and preconditioned ``pls`` on a DeepONet)."""


class DeepONetACTrainer(_ArchPrefixMixin, ACTrainer):
    """``ACTrainer`` counterpart of :class:`DeepONetPoissonTrainer`."""


class DeepONetStokesTrainer(_ArchPrefixMixin, StokesTrainer):
    """``StokesTrainer`` counterpart: our Stokes losses on a 3-channel DeepONet. The model's
    grid ``forward`` emits ``(u_x, u_y, p)`` like the FNO, so ``_predict`` and the whole
    evaluation run unchanged."""


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
                 lambda_bc_pi: float = 0.0, pi_bc: str = "mollifier", **kwargs):
        kwargs["loss_type"] = "data"
        super().__init__(*args, **kwargs)
        self.loss_type = "pideeponet"
        self.n_colloc = n_colloc
        self.pi_reduction = pi_reduction
        # 'mollifier': hard BC inside the model, no penalty (the reference). 'zero': boundary
        # nodes of the grid output zeroed as our arms do, plus the soft penalty weighted by
        # lambda_bc_pi -- see PIDeepONetPoissonLoss for why the penalty is then mandatory.
        self.pi_bc = pi_bc
        self.lambda_bc = float(lambda_bc_pi)
        self.criterion = PIDeepONetPoissonLoss(reduction=pi_reduction, p=2,
                                               lambda_bc=self.lambda_bc)
        if not hasattr(self.model, "forward_at"):
            raise TypeError("--loss pideeponet needs a coordinate-queryable model "
                            "(--model deeponet)")
        dev = self.model.grid_coords.device
        bmask = self.problem.boundary_mask
        self._interior_idx = (~bmask).nonzero(as_tuple=False).squeeze(1).to(dev)
        # Boundary collocation for the penalty: every boundary node, shared across the batch
        # (no coordinate gradient is needed there, so the [Qb, 2] form is fine).
        self._bc_coords = self.model.grid_coords[bmask.nonzero(as_tuple=False).squeeze(1).to(dev)]

    def _colloc(self, batch_size: int):
        """``(coords [B, Q, 2] requiring grad, node index [Q])``.

        Restricted to **interior** nodes, matching ``PINOPoissonLoss``'s interior-only slice
        (and the reference ``FDM_Darcy``). Averaging over different node sets would make the
        two baselines' losses differ by the ratio of node counts alone, so their learning
        rates would not be comparable.
        """
        return _interior_colloc(self.model, self._interior_idx, self.n_colloc, batch_size)

    def train_epoch(self) -> float:
        self.model.train()
        total, nb = 0.0, 0
        for batch_data, _ in self.train_loader:
            self.optimizer.zero_grad()
            fs_grid, _, fs_node, _ = batch_data
            fs_grid, fs_node = fs_grid.to(self.device), fs_node.to(self.device)
            coords, idx = self._colloc(fs_grid.shape[0])
            f_at = fs_node[:, idx]                                         # [B, Q]
            bc = self._bc_coords if self.lambda_bc > 0.0 else None
            loss = self.criterion(self.model, fs_grid, coords, f_at, bc_coords=bc)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            nb += 1
        return total / nb

    def _file_prefix(self) -> str:
        ctag = "" if not self.n_colloc else f"-c{self.n_colloc}"
        # Mollified runs keep their historical file names byte-identical; the zero-BC arm is
        # tagged (with its penalty weight) so it never lands on a mollified run's files.
        btag = "" if self.pi_bc != "zero" else f"-zbc{self.lambda_bc:g}"
        return (f"deeponet_pi-{self.pi_reduction}{ctag}{btag}_K{self.K}_"
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
                             "values and the residual values coincide on the boundary "
                             "(--pideeponet_bc zero is implemented for Poisson only)")

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



# ================================================================================ Stokes

class PINOStokesTrainer(StokesTrainer):
    r"""Stokes with the PINO finite-difference strong-form residual.

    Evaluation, model selection and the reported velocity / pressure errors are the inherited
    ones (``_predict``: input scaling, pressure scaling, velocity BC projection, pressure gauge,
    pressure read on the Q1 subgrid). Only the training objective differs: the residual is
    formed on the **full** fine grid from the physical fields, with the velocity boundary ring
    zeroed exactly as ``_predict`` projects it — the Stokes form of ``--pino_bc zero`` — and the
    FNO's pressure channel used everywhere rather than strided (a strong-form residual needs
    ``∇p`` at every interior node). No mollifier is involved.
    """

    def __init__(self, *args, pino_reduction: str = "rel", div_weight: float = 1.0, **kwargs):
        kwargs["loss_type"] = "data"           # inert in StokesTrainer.__init__; replaced below
        super().__init__(*args, **kwargs)
        self.loss_type = "pino"
        self.pino_reduction = pino_reduction
        self.div_weight = float(div_weight)
        self.criterion = PINOStokesLoss(self.grid_size, mu=self.mu, reduction=pino_reduction,
                                        div_weight=div_weight)
        nx, ny = self.grid_size
        self._vel_mask = boundary_mask_grid(nx, ny).to(self.device)          # [ny, nx]

    def _physical_grid(self, f_grid: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W]`` physical fields: velocity zeroed on the boundary ring, pressure
        times ``p_scale`` on the full grid."""
        out = self.model(f_grid / self.f_scale)
        u = out[:, :2] * self._vel_mask
        p = out[:, 2:3] * self.p_scale
        return torch.cat([u, p], dim=1)

    def train_epoch(self) -> float:
        self.model.train()
        total, nb = 0.0, 0
        for f_grid, _, _, _ in self.train_loader:
            f_grid = f_grid.to(self.device)
            loss = self.criterion(self._physical_grid(f_grid), f_grid)   # labels never touched
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            nb += 1
        return total / nb

    def _file_prefix(self) -> str:
        nx, _ = self.grid_size
        dtag = "" if self.div_weight == 1.0 else f"-dw{self.div_weight:g}"
        return (f"{_arch_tag(self.model)}_stokes_pino-{self.pino_reduction}{dtag}_mu{self.mu:g}_"
                f"gr{nx}_K{self.K}_samples-{len(self.train_dataset)}-{len(self.val_dataset)}-"
                f"{len(self.test_dataset)}")


class PIDeepONetStokesTrainer(StokesTrainer):
    r"""Stokes with the strong-form residual differentiated by autodiff through the trunk.

    Collocation points are the interior grid nodes (optionally a fresh ``n_colloc`` subsample
    per step), ``f`` at those points is a plain index into ``f_node``. The branch sees
    ``f / f_scale`` exactly as ``_predict`` feeds every arm, and the loss scales the pressure
    channel by ``p_scale``. BC as for Poisson: ``pi_bc='mollifier'`` multiplies the velocity
    channels by ``sin(πx)sin(πy)`` inside the model (pressure untouched), ``'zero'`` zeroes the
    velocity boundary nodes of the grid output and adds the relative boundary penalty
    ``lambda_bc_pi``.
    """

    def __init__(self, *args, n_colloc: int = 0, pi_reduction: str = "rel",
                 pi_bc: str = "mollifier", lambda_bc_pi: float = 0.0,
                 div_weight: float = 1.0, **kwargs):
        kwargs["loss_type"] = "data"
        super().__init__(*args, **kwargs)
        self.loss_type = "pideeponet"
        self.n_colloc = n_colloc
        self.pi_reduction = pi_reduction
        self.pi_bc = pi_bc
        self.lambda_bc = float(lambda_bc_pi)
        self.div_weight = float(div_weight)
        self.criterion = PIDeepONetStokesLoss(mu=self.mu, reduction=pi_reduction, p=2,
                                              div_weight=div_weight, lambda_bc=self.lambda_bc,
                                              p_scale=self.p_scale)
        if not hasattr(self.model, "forward_at"):
            raise TypeError("--loss pideeponet needs a coordinate-queryable model "
                            "(--model deeponet)")
        dev = self.model.grid_coords.device
        bmask = self.problem.boundary_mask
        self._interior_idx = (~bmask).nonzero(as_tuple=False).squeeze(1).to(dev)
        self._bc_coords = self.model.grid_coords[bmask.nonzero(as_tuple=False).squeeze(1).to(dev)]

    def train_epoch(self) -> float:
        self.model.train()
        total, nb = 0.0, 0
        for f_grid, f_node, _, _ in self.train_loader:
            f_grid, f_node = f_grid.to(self.device), f_node.to(self.device)
            coords, idx = _interior_colloc(self.model, self._interior_idx, self.n_colloc,
                                           f_grid.shape[0])
            f_at = f_node[:, idx]                                              # [B, Q, 2]
            bc = self._bc_coords if self.lambda_bc > 0.0 else None
            loss = self.criterion(self.model, f_grid / self.f_scale, coords, f_at, bc_coords=bc)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            nb += 1
        return total / nb

    def _file_prefix(self) -> str:
        nx, _ = self.grid_size
        ctag = "" if not self.n_colloc else f"-c{self.n_colloc}"
        btag = "" if self.pi_bc != "zero" else f"-zbc{self.lambda_bc:g}"
        dtag = "" if self.div_weight == 1.0 else f"-dw{self.div_weight:g}"
        return (f"deeponet_stokes_pi-{self.pi_reduction}{ctag}{btag}{dtag}_mu{self.mu:g}_"
                f"gr{nx}_K{self.K}_samples-{len(self.train_dataset)}-{len(self.val_dataset)}-"
                f"{len(self.test_dataset)}")

