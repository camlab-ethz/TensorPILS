"""The appendix sample panel: input, reference and every arm's prediction on one test sample.

A grid of small maps, ``\\linewidth`` wide. The leading column is the problem -- the body force
that goes in and the reference velocity that should come out -- and each remaining column is one
arm, predicted velocity on top and the pointwise error below. Every prediction shares one colour
scale and every error shares another, so a reader can compare across columns by eye rather than
by reading axis labels; that is the whole reason to draw the figure instead of pointing at the
table.

The velocity magnitude is what is shown. Pressure is a gauge-fixed field whose reference is
visually featureless at this forcing, and adding it would double the height for nothing.

This needs *reloadable* models, which the main sweep does not keep (it passes ``--no_checkpoint``).
``experiments/stokes/unstructured/sweep_checkpointed.txt`` re-runs the four arms with the
checkpoint kept, writing to ``output/stokes/unstructured_ckpt/<arm>/``.

    python experiments/stokes/unstructured/plot_samples.py \
        --figures_dir ../../paper/precond-pino-paper/figures
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

# (directory, column heading). Order is the table's: ours, supervised, control, baseline.
ARMS = [
    ("pls",        r"$L_{\mathrm{PLS}}$"),
    ("data",       r"$L_{\mathrm{data}}$"),
    ("galerkin",   r"$L_{\mathrm{LS}}$"),
    ("pideeponet", "PI-DeepONet"),
]

FS_LAB, FS_TICK = 8.0, 6.5
MUTED, TEXT = "#6b6b6b", "#1a1a1a"


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": MUTED, "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.5,
    })


def speed(vec, n):
    """|.| of a node-major interleaved velocity vector -- see StokesProblem.unpack."""
    uv = np.asarray(vec).reshape(-1, 2)
    assert uv.shape[0] == n, f"{uv.shape[0]} velocity nodes against {n} mesh nodes"
    return np.hypot(uv[:, 0], uv[:, 1])


def load_arm(root, arm, problem, coords, f_node, f_scale, device):
    """Rebuild the model from its checkpoint and predict on one sample."""
    hits = sorted(glob.glob(os.path.join(root, arm, "checkpoints", "*_best.pth")))
    if not hits:
        return None
    ckpt = torch.load(hits[0], map_location=device, weights_only=False)
    cfg = dict(ckpt["model_config"])
    # The checkpoint records no architecture tag, so read it off the config's own keys: GAOT
    # carries a latent token grid, the DeepONet a trunk width. Defaulting to one of them would
    # load a DeepONet's weights into a transformer and fail deep inside load_state_dict.
    if "latent_grid" in cfg:
        kind = "gaot"
        from tensorpils.gaot.model import GAOTModel as M
    elif "trunk_fourier" in cfg or "depth" in cfg:
        kind = "deeponet"
        from tensorpils.baselines.deeponet import DeepONetModel as M
    else:
        raise KeyError(f"cannot tell the architecture from {sorted(cfg)}")
    model = M(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # StokesTrainer._predict feeds every model f / f_scale -- the physics ties |f| ~ 10^2 to a
    # unit-norm velocity, and an unscaled input simply does not train. Omitting it here produces
    # a prediction that is wrong by more than the error being plotted.
    with torch.no_grad():
        fs = (f_node / f_scale).unsqueeze(0)
        out = (model.forward_at(fs, coords) if kind == "deeponet"
               else model.forward_nodes(fs, coords=coords))
        u, p = problem.from_nodes(out)
        u = problem.project_velocity_bc(u)
    return u[0].cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/stokes/unstructured_ckpt")
    ap.add_argument("--out_dir", default="output/stokes/unstructured/figures")
    ap.add_argument("--figures_dir", default="../../paper/precond-pino-paper/figures")
    ap.add_argument("--mesh_h", type=float, default=0.035)
    ap.add_argument("--mesh_cache",
                    default="/cluster/scratch/shiwen/TensorPILS_cache/meshes/obst_h0.035.msh")
    ap.add_argument("-K", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample", type=int, default=-1,
                    help="test index to draw; -1 scans --n_scan samples and picks the most "
                         "representative one (see --dataset_err)")
    ap.add_argument("--n_scan", type=int, default=24)
    ap.add_argument("--dataset_err", type=float, nargs=4, default=(1.74, 3.22, 29.45, 41.75),
                    help="dataset-level velocity rel. L2 per arm, in ARMS order, for the models "
                         "in --root; the scan picks the sample closest to these")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    style()
    from tensorpils.data import create_unstructured_stokes_datasets

    # The figure must use a TEST sample: a training one would flatter every arm.
    _, _, test = create_unstructured_stokes_datasets(
        n_train=1024, n_val=128, n_test=256, K=args.K, chara_length=args.mesh_h,
        seed=args.seed, cache_path=args.mesh_cache)
    mesh, problem, _ = test.get_shared_resources()
    problem = problem.to(args.device)
    f_scale = float(getattr(test, "f_scale", 1.0))
    print(f"    f_scale = {f_scale:.4g}")

    pts = mesh.points[:, :2].numpy()
    coords_t = mesh.points[:, :2].to(args.device, torch.float32)
    n = pts.shape[0]
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], mesh.cells["triangle6"][:, :3].numpy())

    def predict_all(idx):
        f, u_ref, _ = test[idx]
        f = f.to(args.device, torch.float32)
        s_ref = speed(u_ref.numpy(), n)
        out = {}
        for arm, _ in ARMS:
            u = load_arm(args.root, arm, problem, coords_t, f, f_scale, args.device)
            if u is None:
                return None, None, None, arm
            out[arm] = speed(u.numpy(), n)
        return f, s_ref, out, None

    # Draw a REPRESENTATIVE sample, not the first one. A panel is only worth printing if the
    # errors it shows are the errors the table reports, so scan and pick the sample whose per-arm
    # errors sit closest to the dataset-level ones in log ratio -- an explicit, statable rule
    # rather than an eye-picked frame.
    if args.sample >= 0:
        idx = args.sample
        f_node, s_ref, preds, missing = predict_all(idx)
    else:
        target = np.asarray(args.dataset_err, dtype=float)
        best = (np.inf, None)
        for k in range(min(args.n_scan, len(test))):
            _, sr, pr, missing = predict_all(k)
            if missing:
                break
            e = np.array([100 * np.linalg.norm(pr[a] - sr) / np.linalg.norm(sr) for a, _ in ARMS])
            score = float(np.abs(np.log(e / target)).sum())
            if score < best[0]:
                best = (score, k)
        if best[1] is None:
            print(f"  !! no checkpoint yet for: {missing} -- run the sweep first")
            return
        idx = best[1]
        print(f"    scanned {min(args.n_scan, len(test))} test samples, drawing #{idx} "
              f"(log-ratio deviation {best[0]:.3f})")
        f_node, s_ref, preds, missing = predict_all(idx)
    if missing:
        print(f"  !! no checkpoint yet for: {missing} -- run the sweep first")
        return
    s_f = speed(f_node.cpu().numpy(), n)

    vmax_u = max([s_ref.max()] + [p.max() for p in preds.values()])
    errs = {a: np.abs(preds[a] - s_ref) for a, _ in ARMS}
    vmax_e = max(e.max() for e in errs.values())

    ncol = 1 + len(ARMS)
    fig, axes = plt.subplots(2, ncol, figsize=(5.5, 2.55),
                             gridspec_kw=dict(wspace=0.06, hspace=0.16))

    def draw(ax, val, vmax, cmap):
        m = ax.tripcolor(tri, val, shading="gouraud", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color(MUTED)
        return m

    # Row 0 is the reference and the four predictions, all on ONE viridis scale so the columns
    # are comparable by eye. Row 1 is the forcing (its own scale -- different units) and the four
    # error fields, all on one magma scale. Labelling the rows "prediction"/"error" would be wrong
    # for column 0, so each cell is titled instead.
    m_u = draw(axes[0, 0], s_ref, vmax_u, "viridis")
    axes[0, 0].set_title(r"reference $|u^\ast|$", fontsize=FS_LAB, pad=2.5)
    m_f = draw(axes[1, 0], s_f, float(s_f.max()), "cividis")
    axes[1, 0].set_xlabel(r"input $|f|$", fontsize=FS_LAB, labelpad=2)

    for j, (arm, head) in enumerate(ARMS, start=1):
        draw(axes[0, j], preds[arm], vmax_u, "viridis")
        axes[0, j].set_title(head, fontsize=FS_LAB, pad=2.5)
        m_e = draw(axes[1, j], errs[arm], vmax_e, "magma")
        rel = 100.0 * np.linalg.norm(preds[arm] - s_ref) / np.linalg.norm(s_ref)
        axes[1, j].set_xlabel(f"{rel:.1f}%", fontsize=FS_TICK, labelpad=2)

    # Side labels, not titles: a title above the lower colorbar lands on the upper one's last
    # tick. shrink leaves a gap between the two bars as well.
    for m, ax_row, lab in ((m_u, axes[0, :], r"$|u|$"), (m_e, axes[1, :], r"$|u - u^\ast|$")):
        cb = fig.colorbar(m, ax=list(ax_row), fraction=0.020, pad=0.014, shrink=0.88)
        cb.set_label(lab, fontsize=FS_LAB, labelpad=3, rotation=90)
        cb.ax.tick_params(labelsize=FS_TICK - 0.5, width=0.5, length=2, color=MUTED)
        cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor(MUTED)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = "stokes_samples"
    pdf = os.path.join(args.out_dir, stem + ".pdf")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(args.out_dir, stem + ".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  {pdf}")
    if args.figures_dir:
        os.makedirs(args.figures_dir, exist_ok=True)
        shutil.copy(pdf, os.path.join(args.figures_dir, stem + ".pdf"))
        print(f"  -> {os.path.join(args.figures_dir, stem + '.pdf')}")
    for arm, _ in ARMS:
        print(f"    {arm:<11} sample rel. |u| error "
              f"{100 * np.linalg.norm(preds[arm] - s_ref) / np.linalg.norm(s_ref):.2f} %")


if __name__ == "__main__":
    main()
