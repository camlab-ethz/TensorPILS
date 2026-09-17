"""Physics-informed loss functions for neural-operator training.

All FEM-based losses act on **node** values ``[B, N]`` and route through
:class:`PoissonProblem` (which wraps TensorMesh's assembled matrices):

* :class:`DataLoss`               — supervised MSE on the grid (labels required)
* :class:`GalerkinLoss`           — ``‖A u − b‖²`` weak-form residual
* :class:`DeepRitzLoss`           — energy ``∫(½|∇u|² − f u)`` (+ optional BC penalty)
* :class:`PreconditionedLSLoss`   — ``½‖P(A u − b)‖²`` with ``P ≈ A⁻¹`` (multigrid V-cycle)
* :class:`PreconditionedDeepRitzLoss` — Deep Ritz with the gradient preconditioned by ``P``

Register a new loss by adding a class and a branch in :func:`build_loss`.
"""

from typing import Optional

import torch
import torch.nn as nn

from .physics import (PoissonProblem, WaveProblem, ACProblem, StokesProblem,
                      apply_zero_boundary)
from .preconditioners import Preconditioner

__all__ = [
    "DataLoss", "NodeDataLoss", "DataL2Loss", "DataH1Loss", "GalerkinLoss", "DeepRitzLoss",
    "PreconditionedLSLoss", "PreconditionedDeepRitzLoss", "build_loss",
    "WaveGalerkinLoss", "build_wave_loss",
    "ACGalerkinLoss", "build_ac_loss",
    "StokesGalerkinLoss", "StokesPLSLoss", "StokesAppliedPLSLoss", "build_stokes_loss",
]


class DataLoss(nn.Module):
    """Supervised MSE on the regular grid. ``u_pred``, ``u_true``: ``[B, H, W]``.

    ``bc_mode='penalty'`` (default): MSE over the full grid (boundary included).
    ``bc_mode='hard'``: MSE over interior nodes only (the outer frame is dropped);
    boundary outputs are left unconstrained at train time and projected to 0 at eval.
    """

    def __init__(self, bc_mode: str = "penalty"):
        super().__init__()
        if bc_mode not in ("penalty", "hard"):
            raise ValueError(f"bc_mode must be 'penalty' or 'hard', got {bc_mode!r}")
        self.bc_mode = bc_mode

    def forward(self, u_pred: torch.Tensor, u_true: torch.Tensor,
                f_node: torch.Tensor = None) -> torch.Tensor:
        if self.bc_mode == "hard":
            return ((u_pred[..., 1:-1, 1:-1] - u_true[..., 1:-1, 1:-1]) ** 2).mean(dim=0).sum()
        return ((u_pred - u_true) ** 2).mean(dim=0).sum()


class NodeDataLoss(nn.Module):
    """Supervised MSE on node values ``[B, N]`` — :class:`DataLoss` off the grid.

    On a uniform grid this is *numerically identical* to ``DataLoss``: both average over the
    batch and sum over spatial points, and the grid nodes carry equal weight. That is what makes
    an unstructured ``data`` arm comparable to the structured one. On a non-uniform mesh the
    equal weighting is no longer an integral — ``DataL2Loss`` (mass-weighted) is the principled
    norm there, and this stays the plain analogue of the grid MSE.

    ``bc_mode='hard'`` drops the boundary nodes, which is what ``DataLoss`` does by slicing the
    outer frame; off the grid the frame is the mesh's ``boundary_mask``.
    """

    def __init__(self, problem: PoissonProblem, bc_mode: str = "penalty"):
        super().__init__()
        if bc_mode not in ("penalty", "hard"):
            raise ValueError(f"bc_mode must be 'penalty' or 'hard', got {bc_mode!r}")
        self.problem = problem
        self.bc_mode = bc_mode

    def forward(self, u_pred_node: torch.Tensor, u_true_node: torch.Tensor,
                f_node: torch.Tensor = None) -> torch.Tensor:
        e2 = (u_pred_node - u_true_node) ** 2
        if self.bc_mode == "hard":
            interior = ~self.problem.boundary_mask.to(e2.device).bool()
            e2 = e2[..., interior]
        return e2.mean(dim=0).sum()


class DataL2Loss(nn.Module):
    r"""Supervised true-``L²`` loss ``½‖u−u★‖²_{L²} = ½ eᵀ M e`` on node values ``[B, N]``.

    Unlike :class:`DataLoss` (flat grid MSE), the error ``e = u_pred − u_true`` is measured in
    the finite-element ``L²`` norm via the mass matrix ``M`` — the same metric used to report
    validation error. Requires labels ``u_true``.
    """

    def __init__(self, problem: PoissonProblem):
        super().__init__()
        self.problem = problem

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        e = u_pred_node - u_true_node
        Me = self.problem._spmm(self.problem.M, e)      # M e
        return 0.5 * (e * Me).sum(dim=-1).mean()        # ½ mean_B(eᵀ M e)


class DataH1Loss(nn.Module):
    r"""Supervised ``H¹₀``-norm loss ``½‖u−u★‖²_{H¹₀} = ½ eᵀ A e`` on node values ``[B, N]``.

    Like :class:`DataL2Loss` but with the stiffness matrix ``A`` (``∫∇φ·∇φ``) in place of the
    mass matrix, measuring the error in the energy / ``H¹₀`` seminorm ``½∫|∇e|²``. The
    prediction is first projected to zero on the Dirichlet boundary, so ``e = Πu_pred − u★``
    lies in ``H₀¹`` where the seminorm is a genuine norm (Poincaré) — without this the
    constant/boundary mode is in the loss nullspace. Requires labels ``u_true``.
    """

    def __init__(self, problem: PoissonProblem):
        super().__init__()
        self.problem = problem

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        u_bc = apply_zero_boundary(u_pred_node, self.problem.boundary_mask)  # -> H¹₀
        e = u_bc - u_true_node
        Ae = self.problem._spmm(self.problem.A, e)      # A e
        return 0.5 * (e * Ae).sum(dim=-1).mean()        # ½ mean_B(eᵀ A e)


class GalerkinLoss(nn.Module):
    """``0.5 ‖A u − b‖²`` on node values ``[B, N]`` (no labels needed)."""

    def __init__(self, problem: PoissonProblem):
        super().__init__()
        self.problem = problem

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        residual = self.problem.residual(u_pred_node, f_node)
        return 0.5 * (residual ** 2).sum(dim=-1).mean()


class DeepRitzLoss(nn.Module):
    """Deep Ritz energy on node values ``[B, N]``.

    ``bc_mode='penalty'`` (soft): ``E(u) + λ_bc · mean(u[∂Ω]²)``.
    ``bc_mode='hard'``: project ``u → 0`` on the boundary, then ``E(u_proj)`` (no penalty).
    """

    def __init__(self, problem: PoissonProblem, lambda_bc: float = 100.0,
                 bc_mode: str = "penalty"):
        super().__init__()
        if bc_mode not in ("penalty", "hard"):
            raise ValueError(f"bc_mode must be 'penalty' or 'hard', got {bc_mode!r}")
        self.problem = problem
        self.lambda_bc = lambda_bc
        self.bc_mode = bc_mode

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        mask = self.problem.boundary_mask
        if self.bc_mode == "hard":
            u_for_energy = apply_zero_boundary(u_pred_node, mask)
            return self.problem.energy(u_for_energy, f_node, reduce="mean")

        energy = self.problem.energy(u_pred_node, f_node, reduce="mean")
        if u_pred_node.dim() == 1:
            bc_loss = (u_pred_node[mask] ** 2).mean()
        else:
            bc_loss = (u_pred_node[:, mask] ** 2).mean()
        return energy + self.lambda_bc * bc_loss


class PreconditionedLSLoss(nn.Module):
    r"""Preconditioned least-squares loss :math:`L = \tfrac12\|P(Au-b)\|^2`.

    ``P`` is a preconditioner ``P ≈ A⁻¹`` (a multigrid V-cycle, or the exact spectral
    blend/power operator). The squared-norm form has c-space Hessian ``AᵀPᵀPA ≈ I``
    (mesh-independent conditioning) and gradient ``≈ J_θᵀ(u − u⋆)``, recovering supervised
    training dynamics without labels.
    """

    def __init__(self, problem: PoissonProblem, precond: Preconditioner):
        super().__init__()
        self.problem = problem
        self.precond = precond

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        r = self.problem.residual(u_pred_node, f_node)
        if r.dim() == 1:
            r = r.unsqueeze(0)
        Pr = self.precond(r)                    # ≈ A⁻¹ r
        return 0.5 * (Pr * Pr).sum(dim=-1).mean()    # changed to sum, not mean.


class PreconditionedDeepRitzLoss(nn.Module):
    r"""Deep Ritz (hard-BC) with the energy gradient preconditioned by ``P ≈ A⁻¹``.

    Surrogate-gradient trick: ``L = ⟨u, (P r).detach()⟩`` so that ``∂L/∂u = P r``,
    i.e. the descent follows ``M r ≈ A⁻¹(Au − b) = u − u⋆`` (Newton/supervised direction).
    The reported value is the surrogate inner product, not the energy; model selection
    uses validation MSE, so this is harmless.
    """

    def __init__(self, problem: PoissonProblem, precond: Preconditioner):
        super().__init__()
        self.problem = problem
        self.precond = precond

    def forward(self, u_pred_node: torch.Tensor, f_node: torch.Tensor,
                u_true_node: torch.Tensor = None) -> torch.Tensor:
        r = self.problem.residual(u_pred_node, f_node)
        u = u_pred_node
        if r.dim() == 1:
            r = r.unsqueeze(0)
            u = u.unsqueeze(0)
        Mr = self.precond(r).detach()           # preconditioned descent direction
        return (u * Mr).sum(dim=1).mean()       # surrogate: ∂/∂u = M r


def build_loss(loss_type: str, problem: PoissonProblem,
               lambda_bc: float, bc_mode: str = "penalty",
               precond: Optional[Preconditioner] = None,
               precondition: bool = False, node_form: bool = False):
    """Factory for the Poisson loss criterion. Add a branch here to register a new loss.

    ``node_form=True`` is the unstructured-mesh path: every other loss here already operates on
    node vectors ``[B, N]`` and is mesh-agnostic, so only ``data`` — the one loss defined on the
    image — needs a substitute (:class:`NodeDataLoss`).
    """
    if loss_type == "data":
        return NodeDataLoss(problem, bc_mode=bc_mode) if node_form else DataLoss(bc_mode=bc_mode)
    if loss_type == "data_l2":
        return DataL2Loss(problem)
    if loss_type == "data_h1":
        return DataH1Loss(problem)
    if loss_type == "galerkin":
        return GalerkinLoss(problem)
    if loss_type == "deepritz":
        if precondition:
            if precond is None:
                raise ValueError("preconditioned deepritz requires a preconditioner")
            return PreconditionedDeepRitzLoss(problem, precond)
        return DeepRitzLoss(problem, lambda_bc=lambda_bc, bc_mode=bc_mode)
    if loss_type == "pls":
        if precond is None:
            raise ValueError("loss_type='pls' requires a preconditioner")
        return PreconditionedLSLoss(problem, precond)
    raise ValueError(f"Unknown loss type {loss_type!r}")


# ============================ wave equation losses ============================

class WaveGalerkinLoss(nn.Module):
    r"""Central-difference weak-form residual loss for the wave equation (no labels).

    Given a node-space trajectory ``seq`` of shape ``[B, L, N]`` — the two ground-truth
    seed frames followed by the model's autoregressive predictions — this sums the squared
    boundary-masked residual over every consecutive triple
    :math:`(u^{k-1}, u^k, u^{k+1})`, weighted by ``discount**(k-1)`` so later (more error-prone)
    rollout steps can be down-weighted. See :meth:`WaveProblem.residual`.
    """

    def __init__(self, problem: WaveProblem, c: float, dt: float, discount: float = 1.0):
        super().__init__()
        self.problem = problem
        self.c = c
        self.dt = dt
        self.discount = discount

    def forward(self, seq_node: torch.Tensor) -> torch.Tensor:
        L = seq_node.shape[1]
        if L < 3:
            raise ValueError(f"wave residual needs at least 3 frames, got L={L}")
        total = seq_node.new_zeros(())
        wsum = 0.0
        for k in range(1, L - 1):
            r = self.problem.residual(seq_node[:, k - 1], seq_node[:, k],
                                      seq_node[:, k + 1], self.c, self.dt)
            w = self.discount ** (k - 1)
            total = total + w * (r ** 2).mean()
            wsum += w
        return total / wsum


def build_wave_loss(problem: WaveProblem, c: float, dt: float, discount: float = 1.0):
    """Factory for the wave physics (Galerkin) criterion. The supervised data term is a
    plain trajectory MSE handled by the trainer; only the residual loss is assembled here."""
    return WaveGalerkinLoss(problem, c=c, dt=dt, discount=discount)


# ============================ Allen–Cahn losses ============================

class ACGalerkinLoss(nn.Module):
    r"""Least-squares weak-form residual loss for Allen–Cahn (no labels).

    Given a node-space trajectory ``seq`` of shape ``[B, L, N]`` — the ground-truth seed frame
    followed by the model's autoregressive predictions — this penalises the boundary-masked step
    residual over every consecutive pair :math:`(u^k, u^{k+1})` as ``½‖R‖²`` (sum over nodes,
    mean over batch), weighted by ``discount**k`` so later (more error-prone) rollout steps can be
    down-weighted. Summing over nodes (rather than averaging) keeps the gradient magnitude
    :math:`O(1)` instead of :math:`O(1/N)`.

    ``integrator`` selects the residual form (see :meth:`ACProblem.residual`): ``"backward_euler"``
    (fully implicit) or ``"convex_concave"`` (Eyre split). Both use the ``/dt`` scaling, so the
    loss is ``dt``-independent in gradient scale.

    If ``precond`` is given, the loss is the **preconditioned** least-squares residual
    :math:`\tfrac12\|P R\|^2` with :math:`P\approx J_0^{-1}`, :math:`J_0=a^2A+cM` the frozen
    (``u²=1``) Newton Jacobian (see ``notes/ac_autoregressive/`` §"Preconditioning the least-squares
    loss"). The preconditioned Gauss–Newton Hessian :math:`J^\top P^\top P J\approx I` removes the
    :math:`\kappa^2` squaring that stalls the bare residual at stiff ``eps``.
    """

    def __init__(self, problem: ACProblem, a: float, eps: float, dt: float, discount: float = 1.0,
                 integrator: str = "backward_euler", detach_coupling=None,
                 precond: Optional[Preconditioner] = None):
        super().__init__()
        self.problem = problem
        self.a = a
        self.eps = eps
        self.dt = dt
        self.discount = discount
        self.integrator = integrator
        self.precond = precond
        # Whether to detach the previous frame u^k in R(u^k, u^{k+1}). ½‖R‖² is self-balancing
        # (∇ vanishes in both args at R=0), so the default is False (no detach) — a valid variant
        # for the bptt-mode study. None -> class default.
        self.detach_coupling = False if detach_coupling is None else detach_coupling

    def forward(self, seq_node: torch.Tensor) -> torch.Tensor:
        L = seq_node.shape[1]
        if L < 2:
            raise ValueError(f"AC residual needs at least 2 frames, got L={L}")
        total = seq_node.new_zeros(())
        wsum = 0.0
        for k in range(L - 1):
            prev = seq_node[:, k].detach() if self.detach_coupling else seq_node[:, k]
            r = self.problem.residual(prev, seq_node[:, k + 1],
                                      self.a, self.eps, self.dt, integrator=self.integrator)
            if self.precond is not None:
                r = self.precond(r)                                   # ½‖P R‖², P ≈ (a²A+cM)⁻¹
            w = self.discount ** k
            total = total + w * (0.5 * (r ** 2).sum(dim=-1).mean())   # sum nodes, mean batch
            wsum += w
        return total / wsum


class ACMinMovementLoss(nn.Module):
    r"""Convex–concave minimizing-movement (JKO) objective loss for Allen–Cahn (no labels).

    The Deep-Ritz analogue of :class:`ACGalerkinLoss` for time-stepping: rather than penalising
    the squared residual ``‖R‖²``, it minimises the convex incremental objective
    ``J(u^k, u^{k+1})`` (see :meth:`ACProblem.mm_objective`) whose unique minimiser is the
    convex–concave step. Over a trajectory ``seq`` of shape ``[B, L, N]`` it sums
    ``discount**k · J(u^k, u^{k+1})`` across consecutive pairs (mean over batch). ``J`` is already
    a full spatial functional, so no per-node reduction is applied. This is intrinsically the
    convex–concave scheme (the split is what makes ``J`` convex), so there is no integrator flag.
    """

    def __init__(self, problem: ACProblem, a: float, eps: float, dt: float, discount: float = 1.0,
                 detach_coupling=None):
        super().__init__()
        self.problem = problem
        self.a = a
        self.eps = eps
        self.dt = dt
        self.discount = discount
        # Whether to detach the proximal centre u^k inside J. Default True: ∇_{u^k}J does NOT
        # vanish at the correct step, so backpropagating through it distorts the trajectory (see
        # notes §sec:mm-pf). None -> class default. This is the *coupling* detach only; the
        # *rollout*-input detach (the full pushforward) is a separate knob on the trainer.
        self.detach_coupling = True if detach_coupling is None else detach_coupling

    def forward(self, seq_node: torch.Tensor) -> torch.Tensor:
        L = seq_node.shape[1]
        if L < 2:
            raise ValueError(f"AC min-movement loss needs at least 2 frames, got L={L}")
        total = seq_node.new_zeros(())
        wsum = 0.0
        for k in range(L - 1):
            prev = seq_node[:, k].detach() if self.detach_coupling else seq_node[:, k]
            Jk = self.problem.mm_objective(prev, seq_node[:, k + 1],
                                           self.a, self.eps, self.dt)   # [B]
            w = self.discount ** k
            total = total + w * Jk.mean()                               # mean over batch
            wsum += w
        return total / wsum


def build_ac_loss(problem: ACProblem, a: float, eps: float, dt: float, discount: float = 1.0,
                  integrator: str = "backward_euler", form: str = "galerkin",
                  detach_coupling=None, precond: Optional[Preconditioner] = None):
    """Factory for the Allen–Cahn physics criterion. The supervised data term is a plain
    trajectory MSE handled by the trainer; only the label-free physics loss is assembled here.

    ``form`` picks the physics loss:
      * ``"galerkin"`` — least-squares residual ``½‖R‖²`` (:class:`ACGalerkinLoss`); ``integrator``
        selects the residual scheme (``backward_euler`` | ``convex_concave``). If ``precond`` is
        given, this becomes the preconditioned residual ``½‖P R‖²``.
      * ``"min_movement"`` — convex–concave minimizing-movement objective ``J``
        (:class:`ACMinMovementLoss`); intrinsically convex–concave, ``integrator`` is ignored.
    """
    if form == "galerkin":
        return ACGalerkinLoss(problem, a=a, eps=eps, dt=dt, discount=discount, integrator=integrator,
                              detach_coupling=detach_coupling, precond=precond)
    if form == "min_movement":
        if precond is not None:
            raise ValueError("preconditioning applies to the least-squares residual only "
                             "(form='galerkin'), not the minimizing-movement objective")
        return ACMinMovementLoss(problem, a=a, eps=eps, dt=dt, discount=discount,
                                 detach_coupling=detach_coupling)
    raise ValueError(f"unknown AC loss form {form!r}; expected 'galerkin' or 'min_movement'")


# ============================== Stokes losses ==============================

class StokesGalerkinLoss(nn.Module):
    r"""Least-squares saddle-point residual ``½‖(r_mom, w·r_cont)‖²`` (no labels).

    The **negative control**, and with ``div_weight`` also the *fair* control for a weighted
    strong-form baseline.

    With ``div_weight=1`` this is the bare ``½‖Kc − b‖²``. For the indefinite Stokes operator
    :math:`\mathcal K` the Gauss--Newton matrix is :math:`J^\top\mathcal K^2 J`, and since
    :math:`\mathcal K` is symmetric its squared eigenvalues give :math:`\kappa = O(h^{-4})`, so
    it is expected *not* to train — exactly as the bare ``galerkin`` loss stalls for Poisson.

    **Why the weight exists.** The strong-form baseline (:func:`~tensorpils.baselines.pino.
    stokes_reduce`) carries a continuity weight ``--pi_div_weight`` and tunes it on validation,
    because momentum and continuity have different units (``div u ~ k u`` against
    ``f ~ mu k^2 u``). That knob is worth a factor of ~27 to it on the structured benchmark
    (46.05 % velocity at ``w=1``, 1.73 % at ``w=100``), and an unweighted FEM control is
    therefore not the comparison a weighted PINO should be read against. Measured here, the two
    objectives are very nearly the *same function* at ``w=1`` (gradient cosine 0.993 on an
    untrained FNO) and diverge as ``w`` grows (0.83 at ``w=300``) — so the weight, not the
    residual's form, is what separates them.

    Note what this weight is: ``½ r^T W r`` with ``W = diag(I, w^2 I)``, a block-diagonal norm
    weight. It is :class:`StokesPLSLoss`'s ``P`` with the velocity block's ``A^-1`` replaced by
    the identity and the pressure block's ``diag(M_p)^-1`` by a constant. Sweeping ``w`` here
    therefore separates what a two-block rescaling buys from what the velocity preconditioner
    adds on top.
    """

    def __init__(self, problem: StokesProblem, div_weight: float = 1.0):
        super().__init__()
        self.problem = problem
        self.div_weight = float(div_weight)

    def forward(self, u_node: torch.Tensor, p_node: torch.Tensor,
                f_node: torch.Tensor) -> torch.Tensor:
        r = self.problem.residual(u_node, p_node, f_node)
        if self.div_weight != 1.0:
            off = self.problem.off_p
            r = torch.cat([r[..., :off], self.div_weight * r[..., off:]], dim=-1)
        return 0.5 * (r ** 2).sum(dim=-1).mean()          # sum dofs, mean batch


class StokesPLSLoss(nn.Module):
    r"""Preconditioned least-squares loss ``½ rᵀ P r`` with ``r = K c − b``.

    ``P = diag(Â⁻¹, Ŝ⁻¹)`` (see :class:`~tensorpils.preconditioners.stokes.
    StokesBlockPreconditioner`) is used as a **norm weight**, giving

    .. math::
        \nabla L = J^\top[\mathcal K\mathcal P\mathcal K c - \mathcal K\mathcal P b],
        \qquad G(\theta) = J^\top \mathcal K\mathcal P\mathcal K J,

    whose conditioning is governed by :math:`\kappa(\mathcal P)`, i.e. :math:`O(h^{-2})`
    — versus :math:`O(h^{-4})` for :class:`StokesGalerkinLoss`.

    The ``½‖P r‖²`` variant is deliberately *not* offered: for a saddle-point system it
    squares the conditioning back to :math:`O(h^{-4})` and is a known dead end (it works
    for Poisson only because there ``P ≈ A⁻¹`` genuinely approximates the inverse, which
    no block-diagonal preconditioner does for :math:`\mathcal K`).
    """

    def __init__(self, problem: StokesProblem, precond: Preconditioner):
        super().__init__()
        self.problem = problem
        self.precond = precond

    def forward(self, u_node: torch.Tensor, p_node: torch.Tensor,
                f_node: torch.Tensor) -> torch.Tensor:
        r = self.problem.residual(u_node, p_node, f_node)
        Pr = self.precond(r)
        return 0.5 * (r * Pr).sum(dim=-1).mean()          # ½ rᵀPr, mean over batch


class StokesAppliedPLSLoss(nn.Module):
    r"""Applied preconditioned least squares ``½‖P r‖²`` with ``r = K c − b``.

    The form that is only viable when :math:`\mathcal P` is a genuine **approximate inverse** of
    the saddle-point operator, i.e. :math:`\mathcal{PK}\approx I`:

    .. math::
        \nabla L = J^\top(\mathcal{PK})^\top(\mathcal{PK}c - \mathcal Pb),
        \qquad
        G(\theta) = J^\top(\mathcal{PK})^\top(\mathcal{PK})J,

    whose conditioning is :math:`\kappa_2(\mathcal{PK})^2` — mesh-independent, :math:`O(1)`, for a
    monolithic multigrid V-cycle
    (:class:`~tensorpils.preconditioners.stokes_monolithic.StokesMonolithicMultigrid`).

    With a **block-diagonal** :math:`\mathcal P` this is the note's documented dead end: no such
    operator approximates :math:`\mathcal K^{-1}`, and the conditioning then grows like
    :math:`O(h^{-3.4})` (measured) rather than staying bounded — worse than the
    :math:`O(h^{-2})` of the weighted :class:`StokesPLSLoss`. It is not blocked, because measuring
    that crossover is itself an experiment (``experiments/stokes/conditioning``), but the pairing is
    warned about at construction.

    ``metric`` selects the norm the squared residual is measured in, and it matters more than one
    would expect. Since :math:`\mathcal Pr \approx c - c^*`, the default ``'euclidean'`` form
    :math:`\tfrac12\|\mathcal Pr\|_2^2` measures the error in **nodal** units, where velocity and
    pressure DOFs count equally — but the physics ties their magnitudes together, and for this
    dataset :math:`\|p\|\sim50\|u\|`, so the loss comes out pressure-dominated by a measured factor
    :math:`\sim\!350` and the velocity is barely optimized. That is the same failure the hand-written
    supervised loss had, reappearing for the same reason. ``'fe'`` divides each block by its own
    dataset FE norm instead, which turns this loss into a **label-free surrogate for the supervised
    objective**: with :math:`\mathcal P=\mathcal K^{-1}` exactly it *is*
    ``StokesTrainer._data_loss``, computed from the residual and no labels at all.
    """

    def __init__(self, problem: StokesProblem, precond: Preconditioner,
                 metric: str = "euclidean", u_scale: float = 1.0, p_scale: float = 1.0):
        super().__init__()
        if metric not in ("euclidean", "fe"):
            raise ValueError(f"metric must be 'euclidean' or 'fe', got {metric!r}")
        self.problem = problem
        self.precond = precond
        self.metric = metric
        self.u_scale = float(u_scale)
        self.p_scale = float(p_scale)

    def forward(self, u_node: torch.Tensor, p_node: torch.Tensor,
                f_node: torch.Tensor) -> torch.Tensor:
        r = self.problem.residual(u_node, p_node, f_node)
        Pr = self.precond(r)
        if self.metric == "euclidean":
            return 0.5 * (Pr ** 2).sum(dim=-1).mean()     # ½‖Pr‖², mean over batch
        e_u, e_p = self.problem.unpack(Pr)                # Pr ≈ c − c*, so these are the errors
        lu = self.problem.velocity_l2(e_u) ** 2 / self.u_scale ** 2
        lp = self.problem.pressure_l2(e_p) ** 2 / self.p_scale ** 2
        return 0.5 * (lu + lp).mean()


def build_stokes_loss(loss_type: str, problem: StokesProblem,
                      precond: Optional[Preconditioner] = None,
                      form: str = "weighted", u_scale: float = 1.0, p_scale: float = 1.0,
                      div_weight: float = 1.0):
    """Factory for the label-free Stokes criterion.

    ``galerkin`` → ``½‖(r_mom, w·r_cont)‖²`` with ``w = div_weight`` (1 = the bare control). ``pls`` needs a ``precond`` and comes in three forms:

    * ``'weighted'`` → ``½ rᵀPr``, for a norm-equivalent block ``P`` (requires ``P`` SPD);
    * ``'applied'`` → ``½‖Pr‖²`` in nodal units, for a monolithic ``P ≈ K⁻¹``;
    * ``'applied_fe'`` → the same with each field block divided by its dataset FE norm
      (``u_scale``, ``p_scale``), which removes the ~350x pressure domination the nodal norm
      carries for this dataset.

    The supervised ``data`` loss is a plain field MSE handled by the trainer, so it has no entry
    here.
    """
    if loss_type == "galerkin":
        return StokesGalerkinLoss(problem, div_weight=div_weight)
    if loss_type == "pls":
        if precond is None:
            raise ValueError("Stokes loss_type='pls' requires a preconditioner")
        if form == "weighted":
            if _is_monolithic(precond):
                print("[warning] pls form='weighted' with a monolithic P: as a norm weight P must "
                      "be SPD, and an approximate inverse of an INDEFINITE operator is indefinite, "
                      "so this loss is unbounded below and will diverge. Use form='applied'.")
            return StokesPLSLoss(problem, precond)
        if form in ("applied", "applied_fe"):
            if not _is_monolithic(precond):
                print("[warning] pls form='applied' with a preconditioner that is not a monolithic "
                      "multigrid: no block-diagonal P approximates K^-1, so this is the note's "
                      "dead-end variant (kappa ~ O(h^-3.4) measured). Intended only as a control.")
            return StokesAppliedPLSLoss(
                problem, precond, metric="fe" if form == "applied_fe" else "euclidean",
                u_scale=u_scale, p_scale=p_scale)
        raise ValueError(f"unknown Stokes pls form {form!r}; expected 'weighted', 'applied' "
                         "or 'applied_fe'")
    raise ValueError(f"unknown Stokes loss {loss_type!r}; expected 'galerkin' or 'pls'")


def _is_monolithic(precond) -> bool:
    return type(precond).__name__ == "StokesMonolithicMultigrid"
