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
           "visualize_data_trajectory"]


def plot_loss_curve(stats, loss_type: str, K: int, save_path: str):
    """Three-panel figure: training loss, validation MSE, and validation FEM-L2."""
    has_l2 = bool(getattr(stats, "val_l2_errors", []))
    has_rel_l2 = bool(getattr(stats, "val_rel_l2_errors", []))
    n_panels = 2 + has_l2 + has_rel_l2
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels))
    epochs = range(len(stats.train_losses))

    axes[0].plot(epochs, stats.train_losses, label="Train Loss", linewidth=2)
    axes[0].axvline(stats.best_epoch, color="r", linestyle="--",
                    label=f"best epoch ({stats.best_epoch})")
    if loss_type == "deepritz":
        min_abs = min((abs(v) for v in stats.train_losses if v != 0), default=1e-6)
        axes[0].set_yscale("symlog", linthresh=max(min_abs * 0.1, 1e-6))
    else:
        axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"FNO Training Loss ({loss_type}, K={K})")
    axes[0].legend(); axes[0].grid(alpha=0.4)

    axes[1].plot(epochs, stats.val_errors, label="Val MSE", linewidth=2, color="orange")
    axes[1].axvline(stats.best_epoch, color="r", linestyle="--")
    axes[1].axhline(stats.best_val_error, color="g", linestyle=":",
                    label=f"best val={stats.best_val_error:.2e}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE")
    axes[1].set_title("Validation MSE")
    axes[1].legend(); axes[1].grid(alpha=0.4)

    panel = 2
    if has_l2:
        axes[panel].plot(epochs, stats.val_l2_errors, label="Val FEM-L2", linewidth=2, color="purple")
        axes[panel].axvline(stats.best_epoch, color="r", linestyle="--")
        axes[panel].set_yscale("log")
        axes[panel].set_xlabel("Epoch"); axes[panel].set_ylabel("FEM-L2")
        axes[panel].set_title("Validation FEM-L2  (√eᵀMe)")
        axes[panel].legend(); axes[panel].grid(alpha=0.4)
        panel += 1
    if has_rel_l2:
        pct = [v * 100 for v in stats.val_rel_l2_errors]
        axes[panel].plot(epochs, pct, label="Val rel-L2 (%)", linewidth=2, color="teal")
        axes[panel].axvline(stats.best_epoch, color="r", linestyle="--")
        axes[panel].set_yscale("log")
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


# ==================== autoregressive rollout visualization (wave / AC) ====================

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


# ==================== data trajectory visualization (no model) ====================

def visualize_data_trajectory(dataset, sample_idx: int = 0, n_frames: int = 5,
                              save_path: str = None, title: str = None,
                              dt: float = None) -> str:
    """Filmstrip + amplitude/boundary diagnostics of a *reference* trajectory, straight from a
    time-dependent dataset (Wave / Allen--Cahn) with **no model** involved.

    Purpose: eyeball the generated data and compare integrators (e.g. backward-Euler vs a
    convex--concave splitting) in both eyeball- and quantitative norm. ``dataset[sample_idx]``
    must yield ``(traj_grid [T+1, H, W], ...)`` (the AC/Wave datasets do).

    The top row is the trajectory at ``n_frames`` evenly-spaced times on a shared symmetric
    colour scale (so decay/coarsening is visible); the bottom panel tracks ``max|u|`` and
    ``||u||_2`` over time, and the title reports the largest boundary value (a zero-Dirichlet
    leak check).

    Saves to ``output/data_viz/traj_sample{idx}.png`` by default (git-ignored, next to all
    other run output); pass ``save_path`` to override. Returns the resolved path.
    """
    item = dataset[sample_idx]
    traj_grid = item[0] if isinstance(item, (tuple, list)) else item
    tg = traj_grid.detach().cpu().numpy()                    # [T+1, H, W]
    T1 = tg.shape[0]
    dt = dt if dt is not None else getattr(dataset, "dt", None)

    if save_path is None:
        save_path = os.path.join("output", "data_viz", f"traj_sample{sample_idx}.png")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    ks = np.unique(np.linspace(0, T1 - 1, min(n_frames, T1)).round().astype(int))
    vlim = max(float(np.abs(tg).max()), 1e-12)               # shared scale -> decay is legible

    frames = tg.reshape(T1, -1)
    maxabs = np.abs(frames).max(axis=1)
    l2 = np.sqrt((frames ** 2).sum(axis=1))
    bmax = float(max(np.abs(tg[:, 0, :]).max(), np.abs(tg[:, -1, :]).max(),
                     np.abs(tg[:, :, 0]).max(), np.abs(tg[:, :, -1]).max()))
    tvec = np.arange(T1) * dt if dt is not None else np.arange(T1)

    fig = plt.figure(figsize=(3.0 * len(ks), 5.6))
    gs = fig.add_gridspec(2, len(ks), height_ratios=[3.0, 1.5], hspace=0.35)
    im = None
    for j, k in enumerate(ks):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(tg[k], cmap="RdBu_r", origin="lower", vmin=-vlim, vmax=vlim)
        tlab = f"  t={k * dt:.3g}" if dt is not None else ""
        ax.set_title(f"k={k}{tlab}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=fig.axes[:len(ks)], fraction=0.02, pad=0.02)

    axd = fig.add_subplot(gs[1, :])
    axd.plot(tvec, maxabs, "o-", color="#c0392b", lw=1.6, ms=3)
    axd.set_xlabel("time $t$" if dt is not None else "frame $k$")
    axd.set_ylabel(r"$\max|u|$", color="#c0392b")
    axd.tick_params(axis="y", labelcolor="#c0392b"); axd.grid(alpha=0.3)
    axr = axd.twinx()
    axr.plot(tvec, l2, "s--", color="#2c3e50", lw=1.4, ms=3)
    axr.set_ylabel(r"$\|u\|_2$", color="#2c3e50"); axr.tick_params(axis="y", labelcolor="#2c3e50")

    integ = getattr(dataset, "integrator", None)
    itag = f", integrator={integ}" if integ else ""
    sup = title or f"Reference trajectory (sample #{sample_idx}, T={T1 - 1} steps{itag})"
    fig.suptitle(f"{sup}   |   boundary max$|u|$ = {bmax:.1e}", fontsize=12)
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Data trajectory -> {save_path}")
    return save_path
