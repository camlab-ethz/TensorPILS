"""Regenerate the per-run training-loss-curve figures (``curves/*_loss.png``) from the saved
``results/*.json`` stats, using the current ``viz.plot_loss_curve``.

The curve PNGs are written *during training*, so they freeze whatever ``viz.py`` existed then. After
a viz change (e.g. validation MSE -> relative-L2 panels, or the symlog fix for the minimizing-
movement energy) this re-renders them from the already-collected stats -- no retraining, no GPU.
For rollout runs the stats always carried ``val_st_rel_l2`` / ``val_final_rel_l2``, so the new
panels are available even for runs trained before the switch; the best-epoch marker is recomputed
as the argmin of the space-time relative-L2 series (the metric we now select on).

    python experiments/allen_cahn/realistic_ac/replot_curves.py \
        --results_dir output/allen_cahn/realistic_ac/results
"""

import argparse
import glob
import json
import os
import types

from tensorpils import viz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="output/allen_cahn/realistic_ac/results")
    ap.add_argument("--curves_dir", default=None,
                    help="Where to write curves (default: sibling 'curves' of results_dir).")
    args = ap.parse_args()
    curves_dir = args.curves_dir or os.path.join(os.path.dirname(os.path.abspath(args.results_dir)), "curves")
    os.makedirs(curves_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.results_dir, "*.json")))
    if not paths:
        raise SystemExit(f"No results JSONs in {args.results_dir}")
    for p in paths:
        rec = json.load(open(p))
        s = rec["stats"]
        st = s.get("val_st_rel_l2") or []
        if not st:
            print(f"  skip (not a rollout run): {os.path.basename(p)}")
            continue
        best = min(range(len(st)), key=lambda i: st[i])            # rel-L2-selected best epoch
        stats = types.SimpleNamespace(
            train_losses=s["train_losses"],
            val_st_rel_l2=st,
            val_final_rel_l2=s.get("val_final_rel_l2") or st,
            val_errors=s.get("val_errors") or [],
            val_l2_errors=[], val_rel_l2_errors=[],
            best_epoch=best, best_val_error=st[best],
        )
        out = os.path.join(curves_dir, f"{rec['prefix']}_loss.png")
        viz.plot_loss_curve(stats, rec.get("loss_type", "galerkin"), rec.get("K", 4), out)


if __name__ == "__main__":
    main()
