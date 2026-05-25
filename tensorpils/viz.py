"""Plotting helpers: training curves, prediction panels, and error distribution.

These are pure functions (no Trainer dependency); the Trainer passes its model, its
evaluation-time boundary projection (``apply_eval_bc``), and output paths.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

__all__ = ["plot_loss_curve", "visualize_sample", "compute_error_distribution"]


def plot_loss_curve(stats, loss_type: str, K: int, save_path: str):
    """Two-panel figure: training loss (log / symlog) and validation MSE (log)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
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
    axes[1].set_title("Validation Error")
    axes[1].legend(); axes[1].grid(alpha=0.4)

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
