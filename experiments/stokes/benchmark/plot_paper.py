"""Figure 5: the two Stokes panels of the main text (``stokes_domain``, ``stokes_val_curves``).

Both panels are drawn on the canvas the manuscript gives them -- ``0.49\\linewidth`` of a 5.5 in
text block -- and reuse the style, colours and dash patterns of
``experiments/allen_cahn/benchmark/plot_paper.py``, so the result figures read as one set.

**Left: the domain.** Poisson and Allen--Cahn live on the unit square and need no picture; this
section's whole claim is that the method works on a geometry no grid resolves, and the reader
cannot otherwise see it. Velocity magnitude of one reference solution, over the actual
triangulation, with the hole cut out.

**Right: validation relative L2 against epoch**, three arms, mean over seeds 42/43/44 with a
min--max band. It carries two statements the table cannot: that ``L_PLS`` sits below ``L_data``
for essentially the whole run rather than at one lucky checkpoint, and that the unpreconditioned
control peaks early (epoch 64--77) and then drifts upwards.

PI-DeepONet is deliberately absent: at ~40 % it would flatten the three curves that matter.

Run after ``run.sh`` (the domain panel needs one reference solve, seconds on a CPU):

    python experiments/stokes/benchmark/plot_paper.py
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
from matplotlib.ticker import NullFormatter

# Same keys, colours and dashes as the Allen-Cahn figure, so a reader who has seen Figure 3
# recognizes the arms here without consulting the legend again.
ARMS = [
    ("pls",      r"$L_{\mathrm{PLS}}$",  "#bf3a77", "-"),
    ("galerkin", r"$L_{\mathrm{LS}}$",   "#e07a1f", (0, (1.2, 1.2))),
    ("data",     r"$L_{\mathrm{data}}$", "#5c167f", (0, (4, 2))),
]

FS_LAB, FS_TICK, FS_LEG = 9.0, 7.5, 6.5
MUTED, TEXT = "#6b6b6b", "#1a1a1a"
W_IN = 0.49 * 5.5                # 2.695 in -- exactly what \includegraphics receives
H_IN = 2.25


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": MUTED, "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    })


def smooth(y, window=15):
    """Centered moving average, edge-padded; NaN-tolerant."""
    y = np.asarray(y, dtype=float)
    w = min(int(window), y.size)
    if w % 2 == 0:
        w -= 1
    if w <= 1:
        return y
    pad = np.pad(y, w // 2, mode="edge")
    ok = np.isfinite(pad)
    num = np.convolve(np.where(ok, pad, 0.0), np.ones(w), mode="valid")
    cnt = np.convolve(ok.astype(float), np.ones(w), mode="valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, num / cnt, np.nan)


def save(fig, out_dir, stem, figures_dir, dpi=300):
    os.makedirs(out_dir, exist_ok=True)
    pdf, png = os.path.join(out_dir, stem + ".pdf"), os.path.join(out_dir, stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    print(f"  {pdf}\n  {png}")
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        shutil.copy(pdf, os.path.join(figures_dir, stem + ".pdf"))
        print(f"  -> {os.path.join(figures_dir, stem + '.pdf')}")


# --------------------------------------------------------------------------- left panel
def panel_domain(args, out_dir):
    """Velocity magnitude of one reference solution, on the mesh the experiment actually uses."""
    from tensorpils.data import create_stokes_datasets

    train, _, _ = create_stokes_datasets(
        n_train=1, n_val=0, n_test=0, K=args.K, chara_length=args.mesh_h,
        seed=args.seed, cache_dir=args.mesh_cache_dir)
    mesh, problem, _ = train.get_shared_resources()
    _, u_node, _ = train[0]

    pts = mesh.points[:, :2].numpy()
    n = pts.shape[0]
    # NODE-major and interleaved: StokesProblem.unpack does c[..., :off_p].reshape(n_u, 2), so the
    # vector is (ux0, uy0, ux1, uy1, ...). Reading it as two stacked component blocks produces a
    # plausible-looking field that is wrong everywhere; assert the shape rather than trust it.
    uv = u_node.reshape(-1, 2).numpy()
    assert uv.shape[0] == n, f"velocity has {uv.shape[0]} nodes, mesh has {n}"
    speed = np.hypot(uv[:, 0], uv[:, 1])

    # Draw on the P1 corner sub-triangulation: the quadratic edge nodes carry no extra vertices
    # for a filled contour, and a corner triangulation is what any FEM viewer would show.
    tri6 = mesh.cells["triangle6"].numpy()
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], tri6[:, :3])

    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    lev = np.linspace(0.0, float(speed.max()), 13)
    cf = ax.tricontourf(tri, speed, levels=lev, cmap="viridis")
    try:                                        # keeps the PDF free of hairline seams
        cf.set_edgecolor("face")                # matplotlib >= 3.8
    except AttributeError:                      # pragma: no cover
        for c in cf.collections:
            c.set_edgecolor("face")
    ax.triplot(tri, color="white", lw=0.12, alpha=0.45)

    th = np.linspace(0, 2 * np.pi, 256)
    ax.plot(args.cx + args.radius * np.cos(th), args.cy + args.radius * np.sin(th),
            color=TEXT, lw=0.7, zorder=5)

    ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"]); ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel(r"$x$", fontsize=FS_LAB, labelpad=1.5)
    ax.set_ylabel(r"$y$", fontsize=FS_LAB, labelpad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03,
                      ticks=np.linspace(0, lev[-1], 4))
    cb.ax.set_title(r"$|u|$", fontsize=FS_LAB, pad=3)      # horizontal: "|u|" rotated is unreadable
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    cb.ax.tick_params(labelsize=FS_TICK, width=0.6, length=2.5, color=MUTED)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor(MUTED)

    fig.tight_layout(pad=0.3)
    save(fig, out_dir, "stokes_domain", args.figures_dir)
    print(f"    mesh: {n} P2 nodes, {tri6.shape[0]} elements, |u|max = {speed.max():.3g}")


# --------------------------------------------------------------------------- right panel
def panel_curves(args, out_dir):
    """Validation velocity relative L2 vs epoch: mean over three seeds, min-max band."""
    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    for key, label, colour, dash in ARMS:
        runs = []
        for f in sorted(glob.glob(os.path.join(args.root, "seed4*", "results", "*.json"))):
            d = json.load(open(f))
            if d.get("loss_type") != key:
                continue
            runs.append(np.asarray(d["stats"]["val_rel_l2_u"], dtype=float) * 100.0)
        if not runs:
            print(f"    !! no runs for {key}")
            continue
        m = min(len(r) for r in runs)
        arr = np.stack([r[:m] for r in runs])
        ep = np.arange(1, m + 1)
        if arr.shape[0] > 1:
            ax.fill_between(ep, smooth(arr.min(0), args.window), smooth(arr.max(0), args.window),
                            color=colour, alpha=0.11, lw=0)
        ax.plot(ep, smooth(arr.mean(0), args.window), color=colour, lw=1.2, ls=dash, label=label)
        print(f"    {key:<9} {arr.shape[0]} seed(s), final {arr.mean(0)[-1]:.2f} %, "
              f"best {arr.mean(0).min():.2f} % at epoch {int(arr.mean(0).argmin()) + 1}")

    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("epoch", fontsize=FS_LAB, labelpad=1.5)
    ax.set_ylabel("val. velocity rel. $L^2$  [%]", fontsize=FS_LAB, labelpad=2)
    ax.grid(True, which="major", color=MUTED, alpha=0.22, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=FS_LEG, handlelength=2.4, labelspacing=0.25,
              borderaxespad=0.3, loc="lower left")
    fig.tight_layout(pad=0.3)
    save(fig, out_dir, "stokes_val_curves", args.figures_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="output/stokes/benchmark/final")
    ap.add_argument("--out_dir", default="output/stokes/benchmark/figures")
    ap.add_argument("--figures_dir", default=None, help="also copy the PDFs here")
    ap.add_argument("--mesh_h", type=float, default=0.035)
    ap.add_argument("--mesh_cache_dir", default="output/meshes")
    ap.add_argument("-K", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cx", type=float, default=0.40)
    ap.add_argument("--cy", type=float, default=0.50)
    ap.add_argument("--radius", type=float, default=0.14)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--skip_domain", action="store_true")
    args = ap.parse_args()

    style()
    if not args.skip_domain:
        print("domain panel:")
        panel_domain(args, args.out_dir)
    print("validation curves:")
    panel_curves(args, args.out_dir)


if __name__ == "__main__":
    main()
