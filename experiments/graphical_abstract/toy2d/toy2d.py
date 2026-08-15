"""How Adam and gradient descent navigate an ill-conditioned quadratic valley (2D toy).

Intuition-building only -- this is not the paper's experiment. It exists to explain the behaviour
observed in ``../run_landscape.py``, where Adam crossed the softest eigendirection in ~10 steps and
then wandered in an isotropic blob, while GD crept along the valley floor.

The model problem is the simplest ill-conditioned convex quadratic,

    L(x) = 1/2 (lam1 x1^2 + lam2 x2^2),    kappa = lam1 / lam2,

whose contours are ellipses of axis ratio ``sqrt(kappa)`` -- the eigenvalues enter the quadratic
form, so they enter the axis *lengths* under a square root. (In the paper's least-squares loss the
Hessian is ``A^2``, so its contour ratio is ``sqrt(kappa(A)^2) = kappa(A)``; the same square-root
rule, one level up.) Default ``kappa = 100`` gives a 10:1 valley, which draws well.

The two behaviours the figure is meant to make obvious:

* **GD is curvature-limited.** Stability caps the step at ``eta < 2/lam1``, set by the *stiff*
  direction, but progress along the soft direction then goes as ``(1 - eta*lam2)`` per step. At the
  optimal step the stiff coordinate flips sign every iteration -- the classic zig-zag -- and the
  soft one decays at rate ``(kappa-1)/(kappa+1)``, i.e. glacially for large kappa.

* **Adam is scale-free per coordinate.** The ``m / sqrt(v)`` normalisation cancels the gradient
  magnitude, so every coordinate advances by roughly ``lr`` per step regardless of its curvature.
  It therefore ignores the condition number entirely and reaches the minimum in ``~|x0| / lr``
  steps. **But only when the eigenbasis and the coordinate basis agree.** ``--rotate`` turns the
  eigenbasis away from the axes; GD is rotation-equivariant and is unaffected, while Adam loses
  its advantage, because per-coordinate normalisation is not per-eigenmode normalisation. That
  misalignment is the situation in the real problem, where Adam works in the nodal basis and the
  stiffness eigenvectors are sine modes.

**Adam does not converge to the minimum of a quadratic -- it converges to a limit cycle.** Run long
enough (``>1e4`` steps) and the iterates settle onto a period-4 orbit of radius

    r_inf ~ 0.011 * lr

measured constant to two significant figures over ``lr`` in ``[5e-3, 2e-1]``, and unchanged when
Adam's ``eps`` is varied from ``1e-8`` to ``1e-14`` -- so it is a property of the map, not a
floating-point floor. The coefficient depends on ``(beta1, beta2)``; rotation barely affects it
(0.014 aligned vs 0.011 rotated). The transient can pass arbitrarily close to the minimum first
(here down to ``1e-140``) before rising onto the orbit, so short runs can look convergent.

The practical reading: Adam carries an accuracy floor proportional to ``lr`` and independent of
conditioning. In the full 3844-dimensional run that floor is ~2 orders of magnitude below the
observed 5.6% plateau, so the plateau there is genuine slow convergence, not this cycle.

Usage:
    python experiments/graphical_abstract/toy2d/toy2d.py --out_dir output/graphical_abstract/toy2d
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display needed
import matplotlib.pyplot as plt

MUTED = "#6b6b6b"
TEXT = "#1a1a1a"


def hessian(lam, rotate_deg):
    """``H = Q diag(lam) Q^T``. Rotating decouples the eigenbasis from the *coordinate* basis --
    which is the whole point: Adam normalises per coordinate, not per eigenvector."""
    th = np.deg2rad(rotate_deg)
    Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return Q @ np.diag(lam) @ Q.T, Q


def gd(x0, H, eta, steps):
    x = np.array(x0, float)
    out = [x.copy()]
    for _ in range(steps):
        x = x - eta * (H @ x)
        out.append(x.copy())
    return np.array(out)


def adam(x0, H, lr, steps, b1=0.9, b2=0.999, eps=1e-8):
    x = np.array(x0, float)
    m = np.zeros(2)
    v = np.zeros(2)
    out = [x.copy()]
    for t in range(1, steps + 1):
        g = H @ x
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        out.append(x.copy())
    return np.array(out)


def steps_to(traj, x0, tol=0.01):
    """First iterate within ``tol`` of the minimum, relative to the start (None if never)."""
    d = np.linalg.norm(traj, axis=1) / np.linalg.norm(x0)
    hit = np.flatnonzero(d < tol)
    return int(hit[0]) if hit.size else None


def draw_contours(ax, H, R1, R2, n=400):
    X1, X2 = np.meshgrid(np.linspace(-R1, R1, n), np.linspace(-R2, R2, n))
    Z = 0.5 * (H[0, 0] * X1 ** 2 + 2 * H[0, 1] * X1 * X2 + H[1, 1] * X2 ** 2)
    ax.contour(X1, X2, Z, levels=np.geomspace(Z.max() * 1e-4, Z.max(), 11),
               colors=MUTED, linewidths=0.5, alpha=0.7)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kappa", type=float, default=100.0,
                    help="Hessian condition number; contours are sqrt(kappa):1 (default 100)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--rotate", type=float, default=0.0,
                    help="rotate the Hessian eigenbasis away from the coordinate axes, in degrees "
                         "(0 = aligned; 45 = maximally misaligned). Adam's per-coordinate "
                         "normalisation only helps when the two bases agree.")
    ap.add_argument("--out_dir", default="output/graphical_abstract/toy2d")
    args = ap.parse_args()

    lam = np.array([args.kappa, 1.0])         # stiff, soft eigenvalues
    H, Q = hessian(lam, args.rotate)
    # Start with equal error in the stiff and soft *eigen*-directions, then map into coordinates.
    # This keeps the optimisation problem identical under rotation (GD is rotation-equivariant, so
    # its step counts are unchanged); only Adam's coordinate alignment varies, which is the point.
    x0 = Q @ np.array([1.0, 1.0])
    eta_opt = 2.0 / (lam[0] + lam[1])         # the step that minimises the GD contraction factor

    runs_gd = [
        (gd(x0, H, eta_opt, args.steps), r"$\eta=2/(\lambda_1+\lambda_2)$", "#e07a1f", "-"),
        (gd(x0, H, 0.4 * eta_opt, args.steps), r"$\eta = 0.4\,\eta_{\rm opt}$", "#b35c00", "--"),
    ]
    runs_ad = [
        (adam(x0, H, 0.05, args.steps), "lr $=0.05$", "#1b6ec2", "-"),
        (adam(x0, H, 0.01, args.steps), "lr $=0.01$", "#0d3f73", "--"),
    ]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    })

    R1, R2 = 1.35, 1.35
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), sharex=True, sharey=True)
    for ax, runs, name in ((axes[0], runs_gd, "gradient descent"),
                           (axes[1], runs_ad, "Adam")):
        draw_contours(ax, H, R1, R2)
        for traj, lab, colr, ls in runs:
            k = steps_to(traj, x0)
            tag = f"{lab}  ({k} steps)" if k is not None else f"{lab}  (never)"
            ax.plot(traj[:, 0], traj[:, 1], color=colr, lw=1.0, ls=ls, zorder=4)
            ax.scatter(traj[::1][:60, 0], traj[::1][:60, 1], s=5, color=colr,
                       zorder=5, linewidths=0, label=tag)
        ax.scatter([x0[0]], [x0[1]], s=30, facecolor="white", edgecolor=TEXT,
                   linewidths=1.0, zorder=7)
        ax.scatter([0], [0], marker="*", s=80, color="#c0392b", zorder=8)
        ax.set_xlim(-R1, R1)
        ax.set_ylim(-R2, R2)
        ax.set_xlabel(r"$x_1$  (stiff, $\lambda_1$)")
        ax.set_title(name, fontsize=9, pad=5)
        ax.legend(loc="lower left", frameon=False, handlelength=1.6, borderaxespad=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"$x_2$  (soft, $\lambda_2$)")

    fig.suptitle(rf"$L=\frac{{1}}{{2}}(\lambda_1x_1^2+\lambda_2x_2^2)$,  "
                 rf"$\kappa={args.kappa:.0f}$,  contours ${np.sqrt(args.kappa):.0f}$:1"
                 rf"{', eigenbasis rotated ' + format(args.rotate, '.0f') + chr(176) if args.rotate else ''}",
                 fontsize=9, y=0.995)
    fig.tight_layout(pad=0.4)

    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, f"toy2d_adam_vs_gd{'_rot%g' % args.rotate if args.rotate else ''}")
    fig.savefig(p + ".png", dpi=200)
    fig.savefig(p + ".pdf")
    plt.close(fig)
    print(f"  {p}.png\n  {p}.pdf")

    # A few numbers to read alongside the picture.
    print(f"\nkappa = {args.kappa:g}   eta_opt = {eta_opt:.4f}   "
          f"GD contraction (soft dir) = {1 - eta_opt * lam[1]:.4f} / step")
    for traj, lab, _, _ in runs_gd + runs_ad:
        k = steps_to(traj, x0)
        d = np.linalg.norm(traj[-1]) / np.linalg.norm(x0)
        lab_clean = lab.replace("$", "").replace("\\", "")
        print(f"  {lab_clean:<32s} steps to 1%: {str(k):>6s}   final rel dist: {d:.2e}")


if __name__ == "__main__":
    main()
