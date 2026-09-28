r"""Trainers for the two baselines.

Each one **subclasses the production trainer** rather than modifying it. The consequence that
matters scientifically: validation, model selection, checkpointing and the final test metric
all run through the inherited code paths, so every baseline number is produced by *exactly* the
same evaluation as our own arms — FEM relative ``L²`` with the eval-time boundary projection.

Only the training objective is overridden. Where a base ``__init__`` insists on building one
of the registered losses, it is given a benign ``loss_type`` and the criterion is replaced
immediately afterwards; ``self.loss_type`` is then set to the baseline's own name so file
prefixes and the results JSON record what actually ran.
"""

from typing import Optional

import torch

from ..meshing import grid_to_node, node_to_grid
from ..trainer import PoissonTrainer, ACTrainer, StokesTrainer, arch_tag as _arch_tag
from .pino import PINOPoissonLoss, PINOACLoss
from .pi_deeponet import (PIDeepONetPoissonLoss, PIDeepONetACLoss, PIDeepONetStokesLoss,
                          autodiff_laplacian)

__all__ = ["PINOPoissonTrainer", "PINOACTrainer",
           "PIDeepONetPoissonTrainer", "PIDeepONetACTrainer", "PIDeepONetStokesTrainer"]


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
    the rollout length, the boundary projection between steps and the detached (pushforward)
    rollout inputs all behave exactly as in the FEM arms.
    """

    def __init__(self, *args, **kwargs):
        kwargs["loss_type"] = "galerkin"
        super().__init__(*args, **kwargs)
        self.loss_type = "pino"
        self.criterion = PINOACLoss(self.grid_size, a=self.a, eps=self.eps, dt=self.dt)

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
        return (f"{_arch_tag(self.model)}_ac_pino_cc_bptt-push_a{self.a:g}_eps{self.eps:g}_"
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
        self.criterion = PIDeepONetACLoss(a=self.a, eps=self.eps, dt=self.dt)
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
        ``N`` nodes retains a double-backward graph per rollout step and exhausts a 24 GB card.
        The values are identical either way, since both passes evaluate the same model at the
        same coordinates; only the retained graph shrinks.
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
            frame = nxt.detach()                                           # pushforward
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
        return (f"deeponet_ac_pi_cc_bptt-push_a{self.a:g}_eps{self.eps:g}_dt{self.dt:g}_"
                f"T{self.n_steps}_R{self.rollout_steps}_K{self.K}_"
                f"samples-{len(self.train_dataset)}-{len(self.val_dataset)}-"
                f"{len(self.test_dataset)}")



# ================================================================================ Stokes

class PIDeepONetStokesTrainer(StokesTrainer):
    r"""PI-DeepONet on the obstacle mesh — the physics-informed baseline that *can* follow.

    PINO's residual is a finite-difference stencil and does not exist off a grid. An **autodiff**
    residual does: the trunk consumes a coordinate, so the strong-form Stokes operator can be
    differentiated at any point of any mesh. The collocation points are the mesh's own interior
    nodes (optionally a fresh ``n_colloc`` subsample per step) and the model is queried through
    ``forward_at``.

    The boundary condition is a **relative penalty** on the velocity at the mesh's boundary nodes
    (weight ``lambda_bc_pi``). It cannot be a mask, because an autodiff Laplacian at an interior
    collocation point is blind to one; and there is no mollifier, because on a domain with a hole
    no closed-form ansatz vanishes on both boundary components.
    """

    def __init__(self, *args, n_colloc: int = 0, pi_reduction: str = "rel",
                 lambda_bc_pi: float = 0.0, div_weight: float = 1.0, **kwargs):
        kwargs["loss_type"] = "data"
        super().__init__(*args, **kwargs)
        self.loss_type = "pideeponet"
        self.n_colloc = n_colloc
        self.pi_reduction = pi_reduction
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

    def _predict(self, f_node: torch.Tensor):
        """``[B, n_u, 2]`` -> physical ``(u, p)``, queried at the mesh's own nodes.

        The same four fixed steps as every other Stokes arm — input scaling, pressure scaling,
        the velocity boundary projection and the pressure gauge — so evaluation is identical to
        the arms this is compared against.
        """
        coords = self.model.grid_coords
        out = self.model.forward_at(f_node / self.f_scale, coords)        # [B, n_u, 3]
        u_node, p_node = self.problem.from_nodes(out)
        return (self.problem.project_velocity_bc(u_node),
                self.problem.project_pressure_gauge(p_node * self.p_scale))

    def train_epoch(self) -> float:
        self.model.train()
        total, nb = 0.0, 0
        for f_node, _, _ in self.train_loader:
            f_node = f_node.to(self.device)
            coords, idx = _interior_colloc(self.model, self._interior_idx, self.n_colloc,
                                           f_node.shape[0])
            f_at = f_node[:, idx]                                          # [B, Q, 2]
            bc = self._bc_coords if self.lambda_bc > 0.0 else None
            loss = self.criterion(self.model, f_node / self.f_scale, coords, f_at, bc_coords=bc)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            nb += 1
        return total / nb

    def _file_prefix(self) -> str:
        ctag = "" if not self.n_colloc else f"-c{self.n_colloc}"
        btag = f"-zbc{self.lambda_bc:g}" if self.lambda_bc > 0.0 else ""
        dtag = "" if self.div_weight == 1.0 else f"-dw{self.div_weight:g}"
        return (f"deeponet_stokes_pi-{self.pi_reduction}{ctag}{btag}{dtag}_mu{self.mu:g}_"
                f"{self.mesh_tag}-n{self.n_nodes}_K{self.K}_"
                f"samples-{len(self.train_dataset)}-{len(self.val_dataset)}-{len(self.test_dataset)}")
