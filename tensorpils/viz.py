"""Plotting helpers: training curves, prediction panels, and error distribution.

These are pure functions (no Trainer dependency); the Trainer passes its model, its
evaluation-time boundary projection (``apply_eval_bc``), and output paths.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

__all__ = ["plot_loss_curve", "visualize_sample", "compute_error_distribution",
           "visualize_rollout_sample", "compute_rollout_error_distribution",
           "visualize_stokes_sample", "compute_stokes_error_distribution"]


def _log_yscale(ax, values):
    """Log y-axis if the series has a finite positive value, else stay linear and say so.
    matplotlib raises on a log axis with nothing positive to show -- e.g. a diverged, all-NaN run --
    and a diagnostic plot must not be what takes a run down."""
    v = np.asarray(values, dtype=float)
    if np.any(np.isfinite(v) & (v > 0)):
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "no finite positive values (diverged?)", transform=ax.transAxes,
                ha="center", va="center", color="red")


def _best_line(ax, value, label_fmt):
    """Horizontal best-value marker, skipped when no epoch ever improved (value stays inf)."""
    if np.isfinite(value):
        ax.axhline(value, color="g", linestyle=":", label=label_fmt.format(value))


def plot_loss_curve(stats, loss_type: str, K: int, save_path: str):
    """Training-diagnostics figure. Panel 0 is always the training loss (optimization health).
    For the rollout trainer (Allen–Cahn) the validation panels are the space-time and
    final-time relative FEM-L2 (the reported, scale-independent metric that also drives model
    selection); for the static Poisson trainer they stay MSE + FEM-L2 + relative FEM-L2."""
    is_rollout = bool(getattr(stats, "val_st_rel_l2", []))
    has_l2 = bool(getattr(stats, "val_l2_errors", []))
    has_rel_l2 = bool(getattr(stats, "val_rel_l2_errors", []))
    n_panels = 3 if is_rollout else 2 + has_l2 + has_rel_l2
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels))
    epochs = range(len(stats.train_losses))

    # Panel 0 — training loss. Log scale, falling back to symlog whenever any value ≤ 0.
    # Only finite values decide: NaN/inf epochs (divergence) are left out of the scale choice.
    axes[0].plot(epochs, stats.train_losses, label="Train Loss", linewidth=2)
    axes[0].axvline(stats.best_epoch, color="r", linestyle="--",
                    label=f"best epoch ({stats.best_epoch})")
    tl = np.asarray(stats.train_losses, dtype=float)
    tl = tl[np.isfinite(tl)]
    if tl.size and tl.min() <= 0:
        min_abs = min((abs(v) for v in tl if v != 0), default=1e-6)
        axes[0].set_yscale("symlog", linthresh=max(min_abs * 0.1, 1e-6))
    else:
        _log_yscale(axes[0], tl)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"FNO Training Loss ({loss_type}, K={K})")
    axes[0].legend(); axes[0].grid(alpha=0.4)

    if is_rollout:
        # Panel 1 — validation space-time relative L2 (the selection metric = best_val_error).
        pct = [v * 100 for v in stats.val_st_rel_l2]
        axes[1].plot(epochs, pct, label="Val rel-L2 space-time (%)", linewidth=2, color="teal")
        axes[1].axvline(stats.best_epoch, color="r", linestyle="--")
        _best_line(axes[1], stats.best_val_error * 100, "best={:.2f}%")
        _log_yscale(axes[1], pct)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("rel-L2 (%)")
        axes[1].set_title("Validation relative L2 — space-time  (√ΣΣ‖ê−u‖²_M / ΣΣ‖u‖²_M)")
        axes[1].legend(); axes[1].grid(alpha=0.4)
        # Panel 2 — validation final-rollout-step relative L2.
        pctf = [v * 100 for v in stats.val_final_rel_l2]
        axes[2].plot(epochs, pctf, label="Val rel-L2 final step (%)", linewidth=2, color="darkorange")
        axes[2].axvline(stats.best_epoch, color="r", linestyle="--")
        _log_yscale(axes[2], pctf)
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("rel-L2 (%)")
        axes[2].set_title("Validation relative L2 — final rollout step")
        axes[2].legend(); axes[2].grid(alpha=0.4)
    else:
        axes[1].plot(epochs, stats.val_errors, label="Val MSE", linewidth=2, color="orange")
        axes[1].axvline(stats.best_epoch, color="r", linestyle="--")
        _best_line(axes[1], stats.best_val_error, "best val={:.2e}")
        _log_yscale(axes[1], stats.val_errors)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE")
        axes[1].set_title("Validation MSE")
        axes[1].legend(); axes[1].grid(alpha=0.4)

        panel = 2
        if has_l2:
            axes[panel].plot(epochs, stats.val_l2_errors, label="Val FEM-L2", linewidth=2, color="purple")
            axes[panel].axvline(stats.best_epoch, color="r", linestyle="--")
            _log_yscale(axes[panel], stats.val_l2_errors)
            axes[panel].set_xlabel("Epoch"); axes[panel].set_ylabel("FEM-L2")
            axes[panel].set_title("Validation FEM-L2  (√eᵀMe)")
            axes[panel].legend(); axes[panel].grid(alpha=0.4)
            panel += 1
        if has_rel_l2:
            pct = [v * 100 for v in stats.val_rel_l2_errors]
            axes[panel].plot(epochs, pct, label="Val rel-L2 (%)", linewidth=2, color="teal")
            axes[panel].axvline(stats.best_epoch, color="r", linestyle="--")
            _log_yscale(axes[panel], pct)
            axes[panel].set_xlabel("Epoch"); axes[panel].set_ylabel("rel-L2 (%)")
            axes[panel].set_title("Validation relative FEM-L2  (√eᵀMe / √uᵀMu)")
            axes[panel].legend(); axes[panel].grid(alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Loss curve -> {save_path}")


@torch.no_grad()
def visualize_sample(model, dataset, split: str, sample_idx: int,
                     device, apply_eval_bc, loss_type: str, K: int, save_path: str):
    """Four-panel figure: source, ground truth, prediction, absolute error."""
    model.eval()
    (f_grid, _, _, _), u_true = dataset[sample_idx]
    f_grid = f_grid.to(device)
    u_true = u_true.cpu().numpy()
    u_pred_t = model(f_grid.unsqueeze(0)).squeeze(0).squeeze(0)   # [H, W]
    u_pred_t = apply_eval_bc(u_pred_t)
    u_pred = u_pred_t.cpu().numpy()
    f_np = f_grid.squeeze(0).cpu().numpy()

    u_vmin = float(min(u_true.min(), u_pred.min()))
    u_vmax = float(max(u_true.max(), u_pred.max()))
    err = np.abs(u_pred - u_true)
    rel_l2 = np.sqrt((err ** 2).sum() / (u_true ** 2).sum()) * 100.0

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    ax[0, 0].imshow(f_np, cmap="RdBu_r", origin="lower"); ax[0, 0].set_title("Source $f$")
    ax[0, 1].imshow(u_true, cmap="RdBu_r", origin="lower", vmin=u_vmin, vmax=u_vmax)
    ax[0, 1].set_title("Ground-truth $u$")
    ax[1, 0].imshow(u_pred, cmap="RdBu_r", origin="lower", vmin=u_vmin, vmax=u_vmax)
    ax[1, 0].set_title(f"Predicted $u$ ({loss_type})")
    ax[1, 1].imshow(err, cmap="hot", origin="lower")
    ax[1, 1].set_title(f"|err|  (rel L2 = {rel_l2:.2f}%)")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    plt.suptitle(f"FNO {split} sample #{sample_idx}  (K={K})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Visualization -> {save_path}")


@torch.no_grad()
def compute_error_distribution(model, test_dataset, device, apply_eval_bc,
                               loss_type: str, K: int, save_path: str):
    """Histogram of per-sample relative L2 error over the test set. Returns (median, errs)."""
    model.eval()
    errs = []
    for i in range(len(test_dataset)):
        (f_grid, _, _, _), u_true = test_dataset[i]
        f_grid = f_grid.to(device)
        u_true = u_true.to(device)
        u_pred = model(f_grid.unsqueeze(0)).squeeze(0).squeeze(0)
        u_pred = apply_eval_bc(u_pred)
        errs.append(torch.sqrt(((u_pred - u_true) ** 2).sum()
                               / (u_true ** 2).sum()).item() * 100.0)
    errs = np.array(errs)
    median, mean, std = np.median(errs), np.mean(errs), np.std(errs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errs, bins=30, color="#3498db", edgecolor="white", alpha=0.85)
    ax.axvline(median, color="#e74c3c", linestyle="--", linewidth=2, label=f"median {median:.2f}%")
    ax.axvline(mean, color="#2ecc71", linestyle="-.", linewidth=2, label=f"mean {mean:.2f}%")
    ax.set_xlabel("Relative $L^2$ error (%)"); ax.set_ylabel("Count")
    ax.set_title(f"FNO ({loss_type}) test error distribution (K={K})")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()

    print("\n" + "=" * 50)
    print("Test set error distribution")
    print(f"  n     : {len(errs)}")
    print(f"  median: {median:.4f}%")
    print(f"  mean  : {mean:.4f}%   std: {std:.4f}%")
    print(f"  min   : {errs.min():.4f}%   max: {errs.max():.4f}%")
    print(f"  saved -> {save_path}")
    print("=" * 50 + "\n")
    return median, errs


# ==================== autoregressive rollout visualization (Allen–Cahn) ====================

@torch.no_grad()
def _rollout_grid(model, traj_grid, device, project_bc, rollout_steps, n_seed_frames):
    """Seed with the first ``n_seed_frames`` frames of a single trajectory ``[T+1, H, W]`` and
    roll forward. Returns ``(preds [R, H, W], refs [R, H, W])`` on ``device`` (zero-BC projected)."""
    traj_grid = traj_grid.to(device)
    window = [project_bc(traj_grid[i]) for i in range(n_seed_frames)]     # each [H, W]
    preds = []
    for _ in range(rollout_steps):
        inp = torch.stack(window, dim=0).unsqueeze(0)         # [1, ns, H, W]
        nxt = project_bc(model(inp).squeeze(0).squeeze(0))    # [H, W]
        preds.append(nxt)
        window = window[1:] + [nxt]
    preds = torch.stack(preds, dim=0)                         # [R, H, W]
    refs = traj_grid[n_seed_frames:n_seed_frames + rollout_steps]         # [R, H, W]
    return preds, refs


@torch.no_grad()
def visualize_rollout_sample(model, dataset, split: str, sample_idx: int,
                             device, project_bc, rollout_steps: int, n_seed_frames: int,
                             loss_type: str, K: int, save_path: str):
    """Snapshot panels (reference / prediction / |error|) at a few rollout times."""
    model.eval()
    traj_grid, _ = dataset[sample_idx]
    preds, refs = _rollout_grid(model, traj_grid, device, project_bc, rollout_steps, n_seed_frames)
    preds, refs = preds.cpu().numpy(), refs.cpu().numpy()

    R = preds.shape[0]
    snaps = sorted(set([0, R // 2, R - 1]))                   # up to 3 distinct frames
    fig, ax = plt.subplots(len(snaps), 3, figsize=(11, 3.4 * len(snaps)), squeeze=False)
    for row, k in enumerate(snaps):
        ref, pred = refs[k], preds[k]
        vmin, vmax = float(min(ref.min(), pred.min())), float(max(ref.max(), pred.max()))
        err = np.abs(pred - ref)
        rel_l2 = np.sqrt((err ** 2).sum() / max((ref ** 2).sum(), 1e-30)) * 100.0
        ax[row, 0].imshow(ref, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
        ax[row, 0].set_ylabel(f"step {k + n_seed_frames}")
        ax[row, 0].set_title("Reference $u$" if row == 0 else "")
        ax[row, 1].imshow(pred, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
        ax[row, 1].set_title(f"Predicted $u$ ({loss_type})" if row == 0 else "")
        ax[row, 2].imshow(err, cmap="hot", origin="lower")
        ax[row, 2].set_title("|err|" if row == 0 else "")
        ax[row, 2].text(0.98, 0.04, f"relL2={rel_l2:.2f}%", color="w",
                        ha="right", va="bottom", transform=ax[row, 2].transAxes, fontsize=9)
        for c in range(3):
            ax[row, c].set_xticks([]); ax[row, c].set_yticks([])
    plt.suptitle(f"FNO {split} sample #{sample_idx}  (K={K}, rollout={R})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Visualization -> {save_path}")


@torch.no_grad()
def compute_rollout_error_distribution(model, test_dataset, device, project_bc,
                                       rollout_steps: int, n_seed_frames: int,
                                       loss_type: str, K: int, save_path: str):
    """Histogram of per-sample rollout relative L2 error (over the whole window) on the test set."""
    model.eval()
    errs = []
    for i in range(len(test_dataset)):
        traj_grid, _ = test_dataset[i]
        preds, refs = _rollout_grid(model, traj_grid, device, project_bc, rollout_steps, n_seed_frames)
        rel = torch.sqrt(((preds - refs) ** 2).sum() / (refs ** 2).sum().clamp_min(1e-30))
        errs.append(rel.item() * 100.0)
    errs = np.array(errs)
    median, mean, std = np.median(errs), np.mean(errs), np.std(errs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errs, bins=30, color="#9b59b6", edgecolor="white", alpha=0.85)
    ax.axvline(median, color="#e74c3c", linestyle="--", linewidth=2, label=f"median {median:.2f}%")
    ax.axvline(mean, color="#2ecc71", linestyle="-.", linewidth=2, label=f"mean {mean:.2f}%")
    ax.set_xlabel("Rollout relative $L^2$ error (%)"); ax.set_ylabel("Count")
    ax.set_title(f"FNO ({loss_type}) test rollout error (K={K}, steps={rollout_steps})")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()

    print("\n" + "=" * 50)
    print("Test set rollout error distribution")
    print(f"  n     : {len(errs)}")
    print(f"  median: {median:.4f}%")
    print(f"  mean  : {mean:.4f}%   std: {std:.4f}%")
    print(f"  saved -> {save_path}")
    print("=" * 50 + "\n")
    return median, errs


# ------------------------------- Stokes (obstacle mesh) -------------------------------

def _tri_from_mesh(mesh, node_ids=None):
    """Triangulation of a ``triangle6`` mesh, optionally restricted to its corner (P1) nodes."""
    import matplotlib.tri as mtri
    import numpy as np
    pts = mesh.points.cpu().numpy()
    conn = mesh.cells["triangle6"].cpu().numpy()[:, :3]        # corners carry the P1 space
    if node_ids is None:
        return mtri.Triangulation(pts[:, 0], pts[:, 1], conn)
    ids = node_ids.cpu().numpy()
    local = np.searchsorted(ids, conn)
    return mtri.Triangulation(pts[ids, 0], pts[ids, 1], local)


@torch.no_grad()
def visualize_stokes_sample(predict, dataset, problem, device, sample_idx: int,
                            save_path: str):
    """Six panels on the mesh: velocity magnitude (reference / predicted / error) and pressure.

    Drawn on the real triangulation, so the obstacle is a hole in the picture rather than a
    region of small values — which is the difference this experiment exists to show. The
    velocity panels use the P2 triangulation and the pressure ones the P1 corner sub-mesh,
    because those are the spaces the two fields actually live in.
    """
    import numpy as np

    f_node, u_true, p_true = dataset[sample_idx]
    u_pred, p_pred = predict(f_node.unsqueeze(0).to(device))
    u_pred, p_pred = u_pred[0].cpu().numpy(), p_pred[0].cpu().numpy()
    u_true, p_true = u_true.cpu().numpy(), p_true.cpu().numpy()

    tri_u = _tri_from_mesh(dataset.mesh)
    tri_p = _tri_from_mesh(dataset.mesh, problem.p_node_ids)

    mag = lambda v: np.hypot(v[..., 0], v[..., 1])
    mu_t, mu_p = mag(u_true), mag(u_pred)
    eu, ep = np.abs(mu_p - mu_t), np.abs(p_pred - p_true)
    ru = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true) * 100
    rp = np.linalg.norm(p_pred - p_true) / np.linalg.norm(p_true) * 100
    vlo, vhi = float(min(mu_t.min(), mu_p.min())), float(max(mu_t.max(), mu_p.max()))
    plo, phi = float(min(p_true.min(), p_pred.min())), float(max(p_true.max(), p_pred.max()))

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    panels = [
        (ax[0, 0], tri_u, mu_t, "reference $|u|$", "viridis", vlo, vhi),
        (ax[0, 1], tri_u, mu_p, "predicted $|u|$", "viridis", vlo, vhi),
        (ax[0, 2], tri_u, eu, f"$||u|-|u^\\star||$   (rel $L^2$ = {ru:.2f}%)", "hot", None, None),
        (ax[1, 0], tri_p, p_true, "reference $p$", "RdBu_r", plo, phi),
        (ax[1, 1], tri_p, p_pred, "predicted $p$", "RdBu_r", plo, phi),
        (ax[1, 2], tri_p, ep, f"$|p-p^\\star|$   (rel $L^2$ = {rp:.2f}%)", "hot", None, None),
    ]
    for a, tri, v, title, cmap, lo, hi in panels:
        tpc = a.tripcolor(tri, v, shading="gouraud", cmap=cmap, vmin=lo, vmax=hi)
        fig.colorbar(tpc, ax=a, fraction=0.046)
        a.set_title(title, fontsize=11)
        a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
    plt.suptitle(f"Stokes past an obstacle — "
                 f"{problem.n_u} P2 / {problem.n_p} P1 nodes, sample #{sample_idx}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Visualization -> {save_path}")


@torch.no_grad()
def compute_stokes_error_distribution(predict, test_dataset, problem, device,
                                      save_path: str, batch_size: int = 32):
    """Per-sample relative FE ``L²`` for both fields. Returns ``(median_u, errs_u, errs_p)``."""
    import numpy as np
    eu, ep = [], []
    for start in range(0, len(test_dataset), batch_size):
        items = [test_dataset[i] for i in range(start, min(start + batch_size,
                                                           len(test_dataset)))]
        f = torch.stack([it[0] for it in items]).to(device)
        ut = torch.stack([it[1] for it in items]).to(device)
        pt = torch.stack([it[2] for it in items]).to(device)
        up, pp = predict(f)
        eu.extend((problem.velocity_l2(up - ut)
                   / problem.velocity_l2(ut).clamp_min(1e-30) * 100).cpu().tolist())
        ep.extend((problem.pressure_l2(pp - pt)
                   / problem.pressure_l2(pt).clamp_min(1e-30) * 100).cpu().tolist())
    eu, ep = np.array(eu), np.array(ep)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, e, name, color in ((ax[0], eu, "velocity", "#3498db"),
                              (ax[1], ep, "pressure", "#9b59b6")):
        a.hist(e, bins=30, color=color, edgecolor="white", alpha=0.85)
        a.axvline(np.median(e), color="#e74c3c", ls="--", lw=2,
                  label=f"median {np.median(e):.2f}%")
        a.axvline(e.mean(), color="#2ecc71", ls="-.", lw=2, label=f"mean {e.mean():.2f}%")
        a.set_xlabel(f"{name} relative FE $L^2$ (%)"); a.set_ylabel("count")
        a.set_title(f"{name} error distribution"); a.legend(); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()

    print("\n" + "=" * 50)
    print("Test set error distribution (Stokes, relative FE L2)")
    print(f"  n        : {len(eu)}")
    print(f"  velocity : median {np.median(eu):.4f}%   mean {eu.mean():.4f}% +- {eu.std():.4f}")
    print(f"  pressure : median {np.median(ep):.4f}%   mean {ep.mean():.4f}% +- {ep.std():.4f}")
    print(f"  saved -> {save_path}")
    print("=" * 50 + "\n")
    return float(np.median(eu)), eu, ep
