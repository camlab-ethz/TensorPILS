"""The appendix sample figure (``stokes_samples``): input, reference and every arm's prediction on
one test sample.

A grid of small maps, ``\\linewidth`` wide. The leading column is the problem -- the body force
that goes in and the reference velocity that should come out -- and each remaining column is one
arm, predicted velocity on top and the pointwise error below. Every prediction shares one colour
scale and every error shares another, so a reader can compare across columns by eye rather than
by reading axis labels; that is the whole reason to draw the figure instead of pointing at the
table.

Rows: velocity magnitude, its pointwise error, pressure, its pointwise error.

It reloads the best-validation checkpoints that ``run.sh`` keeps for seed 42:

    python experiments/stokes/benchmark/plot_samples.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

# (key, column heading, run-file prefix). Order is the table's: ours, supervised, control,
# baseline.
ARMS = [
    ("pls",        r"$L_{\mathrm{PLS}}$",  "gaot_stokes_pls_"),
    ("data",       r"$L_{\mathrm{data}}$", "gaot_stokes_data_"),
    ("galerkin",   r"$L_{\mathrm{LS}}$",   "gaot_stokes_galerkin_"),
    ("pideeponet", "PI-DeepONet",          "deeponet_stokes_pi-"),
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


def load_arm(run_dir, prefix, problem, coords, f_node, f_scale, p_scale, device):
    """Rebuild the model from its checkpoint and predict on one sample."""
    hits = sorted(glob.glob(os.path.join(run_dir, "checkpoints", f"{prefix}*_best.pth")))
    if not hits:
        return None, None
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
        # The same four fixed steps as StokesTrainer._predict: the pressure channel carries a
        # scale of its own (the physics ties |p| ~ 50|u|), and both fields are projected onto the
        # admissible space -- zero velocity on the two boundary components, zero-mean pressure.
        u = problem.project_velocity_bc(u)
        p = problem.project_pressure_gauge(p * p_scale)
    return u[0].cpu(), p[0].cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/stokes/benchmark/final/seed42",
                    help="run directory holding the four arms' checkpoints and results")
    ap.add_argument("--out_dir", default="output/stokes/benchmark/figures")
    ap.add_argument("--figures_dir", default=None, help="also copy the PDF here")
    ap.add_argument("--mesh_h", type=float, default=0.035)
    ap.add_argument("--mesh_cache_dir", default="output/meshes")
    ap.add_argument("-K", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample", type=int, default=-1,
                    help="test index to draw; -1 scans --n_scan samples and picks the most "
                         "representative one (see --dataset_err)")
    ap.add_argument("--n_scan", type=int, default=24)
    ap.add_argument("--dataset_err", type=float, nargs=4, default=None,
                    help="dataset-level velocity rel. L2 [%%] per arm, in ARMS order; the scan picks "
                         "the sample closest to these. Default: read from the results JSONs in "
                         "--root")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    style()
    if args.dataset_err is None:
        errs = []
        for _, _, prefix in ARMS:
            hits = sorted(glob.glob(os.path.join(args.root, "results", f"{prefix}*.json")))
            if not hits:
                print(f"  !! no results for {prefix} under {args.root} -- run run.sh first")
                return
            with open(hits[0]) as fh:
                errs.append(100.0 * json.load(fh)["test_rel_l2_u"])
        args.dataset_err = errs
    from tensorpils.data import create_stokes_datasets

    # The figure must use a TEST sample: a training one would flatter every arm.
    _, _, test = create_stokes_datasets(
        n_train=1024, n_val=128, n_test=256, K=args.K, chara_length=args.mesh_h,
        seed=args.seed, cache_dir=args.mesh_cache_dir)
    mesh, problem, _ = test.get_shared_resources()
    problem = problem.to(args.device)
    f_scale = float(getattr(test, "f_scale", 1.0))
    p_scale = float(getattr(test, "p_scale", 1.0))
    print(f"    f_scale = {f_scale:.4g}   p_scale = {p_scale:.4g}")

    pts = mesh.points[:, :2].numpy()
    coords_t = mesh.points[:, :2].to(args.device, torch.float32)
    n = pts.shape[0]
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], mesh.cells["triangle6"][:, :3].numpy())

    def predict_all(idx):
        f, u_ref, p_ref = test[idx]
        f = f.to(args.device, torch.float32)
        ref = (speed(u_ref.numpy(), n), p_ref.numpy().reshape(-1))
        out = {}
        for arm, _, prefix in ARMS:
            u, q = load_arm(args.root, prefix, problem, coords_t, f, f_scale, p_scale,
                            args.device)
            if u is None:
                return None, None, None, arm
            out[arm] = (speed(u.numpy(), n), q.numpy().reshape(-1))
        return f, ref, out, None

    # Draw a REPRESENTATIVE sample, not the first one. A panel is only worth printing if the
    # errors it shows are the errors the table reports, so scan and pick the sample whose per-arm
    # errors sit closest to the dataset-level ones in log ratio -- an explicit, statable rule
    # rather than an eye-picked frame.
    if args.sample >= 0:
        idx = args.sample
        f_node, ref, preds, missing = predict_all(idx)
    else:
        target = np.asarray(args.dataset_err, dtype=float)
        best = (np.inf, None)
        for k in range(min(args.n_scan, len(test))):
            _, sr, pr, missing = predict_all(k)
            if missing:
                break
            e = np.array([100 * np.linalg.norm(pr[a][0] - sr[0]) / np.linalg.norm(sr[0])
                          for a, _, _ in ARMS])
            score = float(np.abs(np.log(e / target)).sum())
            if score < best[0]:
                best = (score, k)
        if best[1] is None:
            print(f"  !! no checkpoint yet for: {missing} -- run the sweep first")
            return
        idx = best[1]
        print(f"    scanned {min(args.n_scan, len(test))} test samples, drawing #{idx} "
              f"(log-ratio deviation {best[0]:.3f})")
        f_node, ref, preds, missing = predict_all(idx)
    if missing:
        print(f"  !! no checkpoint yet for: {missing} -- run the sweep first")
        return
    s_f = speed(f_node.cpu().numpy(), n)

    s_ref, p_ref = ref
    s_f = speed(f_node.cpu().numpy(), n)

    # Pressure is P1: 1035 values on the corner vertices, which are exactly the P1 node set
    # (verified: p_node_ids equals the unique corners of triangle6[:, :3]). Scatter onto a
    # full-length array so one Triangulation serves both fields; the quadratic edge-node entries
    # are never read, since a corner triangulation interpolates from its three corners only.
    pid = problem.p_node_ids.cpu().numpy()

    def on_mesh(vals_p1):
        full = np.zeros(n, dtype=float)
        full[pid] = vals_p1
        return full

    u_pred = {a: preds[a][0] for a, _, _ in ARMS}
    q_pred = {a: preds[a][1] for a, _, _ in ARMS}
    u_err = {a: np.abs(u_pred[a] - s_ref) for a, _, _ in ARMS}
    q_err = {a: np.abs(q_pred[a] - p_ref) for a, _, _ in ARMS}

    vmax_u = max([s_ref.max()] + [v.max() for v in u_pred.values()])
    vmax_ue = max(v.max() for v in u_err.values())
    vabs_p = max([np.abs(p_ref).max()] + [np.abs(v).max() for v in q_pred.values()])
    vmax_pe = max(v.max() for v in q_err.values())

    ncol = 1 + len(ARMS)
    fig, axes = plt.subplots(4, ncol, figsize=(5.5, 4.55),
                             gridspec_kw=dict(wspace=0.06, hspace=0.16))

    def draw(ax, val, cmap, vmin, vmax):
        m = ax.tripcolor(tri, val, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color(MUTED)
        return m

    # Column 0 is the problem: the reference fields, plus the forcing that produced them.
    m_u = draw(axes[0, 0], s_ref, "viridis", 0.0, vmax_u)
    axes[0, 0].set_title(r"reference", fontsize=FS_LAB, pad=2.5)
    draw(axes[1, 0], s_f, "cividis", 0.0, float(s_f.max()))
    # Title, not xlabel: below the panel it sits against the pressure row and reads as that row's
    # heading. Above it, it pairs with "reference" in the cell overhead.
    axes[1, 0].set_title(r"input $|f|$", fontsize=FS_LAB, pad=2.5)
    m_p = draw(axes[2, 0], on_mesh(p_ref), "RdBu_r", -vabs_p, vabs_p)
    axes[3, 0].set_axis_off()

    for j, (arm, head_lab, _) in enumerate(ARMS, start=1):
        draw(axes[0, j], u_pred[arm], "viridis", 0.0, vmax_u)
        axes[0, j].set_title(head_lab, fontsize=FS_LAB, pad=2.5)
        m_ue = draw(axes[1, j], u_err[arm], "magma", 0.0, vmax_ue)
        draw(axes[2, j], on_mesh(q_pred[arm]), "RdBu_r", -vabs_p, vabs_p)
        m_pe = draw(axes[3, j], on_mesh(q_err[arm]), "magma", 0.0, vmax_pe)
        eu = 100.0 * np.linalg.norm(u_pred[arm] - s_ref) / np.linalg.norm(s_ref)
        ep = 100.0 * np.linalg.norm(q_pred[arm] - p_ref) / np.linalg.norm(p_ref)
        axes[1, j].set_xlabel(f"{eu:.1f}%", fontsize=FS_TICK, labelpad=2)
        axes[3, j].set_xlabel(f"{ep:.1f}%", fontsize=FS_TICK, labelpad=2)

    # Side labels, not titles: a title above one colorbar lands on the one above it.
    bars = ((m_u, axes[0, :], r"$|u|$"), (m_ue, axes[1, :], r"$|u - u^\ast|$"),
            (m_p, axes[2, :], r"$p$"),   (m_pe, axes[3, :], r"$|p - p^\ast|$"))
    for m, ax_row, lab in bars:
        cb = fig.colorbar(m, ax=list(ax_row), fraction=0.020, pad=0.014, shrink=0.92)
        cb.set_label(lab, fontsize=FS_TICK + 0.5, labelpad=3, rotation=90)
        cb.ax.tick_params(labelsize=FS_TICK - 1.0, width=0.5, length=2, color=MUTED)
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
    for arm, _, _ in ARMS:
        eu = 100 * np.linalg.norm(u_pred[arm] - s_ref) / np.linalg.norm(s_ref)
        ep = 100 * np.linalg.norm(q_pred[arm] - p_ref) / np.linalg.norm(p_ref)
        print(f"    {arm:<11} sample rel. error   u {eu:6.2f} %   p {ep:6.2f} %")


if __name__ == "__main__":
    main()
