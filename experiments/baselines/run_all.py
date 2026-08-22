"""Run the whole baseline queue sequentially on one GPU.

The ``*.sbatch`` scripts are the SLURM path (submit from a login node). This is the
direct-execution path for a Euler compute node (``eu-a*`` / ``eu-g*``), where the job is
already inside an allocation and there is nothing to submit to.

**Why this is not just a shell loop.** Every ``tensorpils.cli`` invocation rebuilds its
dataset. For Poisson that costs 0.2 s and does not matter, so those stages simply shell out
to the CLI and inherit its tested argument handling. For Allen-Cahn the dataset is a batched
FEM Newton solve costing O(10 min), and the queue has 18 AC runs -- rebuilding each time
would burn hours on identical work. So the AC stage builds the dataset **once per seed** and
runs all six arms against it in-process, reusing ``cli._build_model`` and the same trainer
classes so the code path is otherwise identical.

Stages are ordered by value per hour, so that stopping early still leaves a coherent story:

    lr        learning-rate sweep for the new arms (everything downstream depends on it)
    ablation  PINO ingredient 2x2 -- also the cross-check that the port is faithful
    poisson   the 7-arm head-to-head table, 3 seeds        <- the deliverable
    href      PINO under mesh refinement (ties to the central conditioning claim)
    ac        the Allen-Cahn table, 3 seeds               <- longest, so it runs last

Usage:
    python experiments/baselines/run_all.py                 # everything, in order
    python experiments/baselines/run_all.py lr ablation     # only these stages
    python experiments/baselines/run_all.py --dry-run       # print the queue and exit
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "output", "baselines")
LOGS = os.path.join(REPO, "logs", "baselines")

SEEDS = [42, 43, 44]

# Learning-rate grids, per architecture. The FNO grid is centred where FNO training lives; the
# first sweep put every DeepONet arm's optimum at the *smallest* value tested (monotone in lr:
# 1.2e-7 at 3e-4 rising to 3.2e-2 at 3e-2), i.e. the grid was mis-centred and the true optimum
# lies below it. Reporting a baseline tuned at the edge of its own grid is exactly the
# under-tuning this stage exists to prevent, so the DeepONet grid is shifted a decade down.
LRS = ["3e-4", "1e-3", "3e-3", "1e-2", "3e-2"]
LRS_DEEPONET = ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3"]


def lr_grid(model):
    return LRS_DEEPONET if model == "deeponet" else LRS

# Poisson: (model, loss, extra flags). The four existing arms keep their published lr; the
# three new ones get whatever the lr stage selects (patched in by _best_lrs at run time).
POISSON_ARMS = [
    ("fno", "data", []),
    ("fno", "galerkin", []),
    ("fno", "pino", []),
    ("fno", "pls", []),
    ("deeponet", "data", []),
    ("deeponet", "pideeponet", []),
    ("deeponet", "pls", []),
]
NEW_ARMS = [("fno", "pino", []), ("deeponet", "pideeponet", []),
            ("deeponet", "data", []), ("deeponet", "pls", [])]

POISSON_BASE = ["--pde", "poisson", "--grid_resolution", "64",
                "--n_train", "1024", "--n_val", "128", "--n_test", "256",
                "-k", "4", "--batch_size", "32", "--lr_min", "1e-6",
                "--optimizer", "adam", "--device", "cuda"]

AC_BASE = dict(grid_resolution=64, n_train=512, n_val=64, n_test=128, K=4,
               dt=0.01, n_steps=20, a=1.0, eps=32.0, rollout_steps=4, epochs=300,
               batch_size=32, bptt_mode="pushforward")

# The Allen-Cahn arms. 'pino'/'pideeponet' are the baselines; the rest are existing arms,
# reproduced at identical settings so the table is self-contained. `glob` distinguishes an
# arm's finished run on disk -- several arms share loss_type='galerkin' and differ only in the
# prefix tags the trainer adds (mm / precmg), so matching on loss_type alone is not enough.
#
# min_movement is here because omitting it understates our own side: notes/paper_story reports
# it as the best label-free Allen-Cahn objective by a wide margin (0.008 vs 0.047 for
# preconditioned LS), so a baseline table without it compares PINO against our weaker arm.
AC_ARMS = [
    dict(model="fno", loss="data", precond="none", form="galerkin", glob="fno_ac_data_*"),
    dict(model="fno", loss="galerkin", precond="none", form="galerkin",
         glob="fno_ac_galerkin_ls_c*"),
    dict(model="fno", loss="pino", precond="none", form="galerkin", glob="fno_ac_pino_*"),
    dict(model="fno", loss="galerkin", precond="multigrid", form="galerkin",
         glob="fno_ac_galerkin_ls_precmg_*"),
    dict(model="fno", loss="galerkin", precond="none", form="min_movement",
         glob="fno_ac_galerkin_mm_*"),
    dict(model="deeponet", loss="data", precond="none", form="galerkin",
         glob="deeponet_ac_data_*"),
    dict(model="deeponet", loss="pideeponet", precond="none", form="galerkin",
         glob="deeponet_ac_pi_*"),
]


def _run_cli(name, argv):
    """Shell out to the CLI, logging to logs/baselines/<name>.log. Never aborts the queue."""
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, f"{name}.log")
    print(f"  [{time.strftime('%H:%M:%S')}] {name} ...", end="", flush=True)
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call([sys.executable, "-m", "tensorpils.cli"] + argv,
                             cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60
    print(f" {'ok' if rc == 0 else f'FAILED rc={rc}'}  ({dt:.1f} min)  -> {log}", flush=True)
    return rc == 0


# ------------------------------------------------------------------ stages

def _have_run(out_dir, model, loss):
    """True if `out_dir` already holds a finished run of this (architecture, loss).

    Lets the lr stage be re-entered after the grid is changed without redoing the cells that
    are already on disk.
    """
    for path in glob.glob(os.path.join(out_dir, "results", "*.json")):
        try:
            with open(path) as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        arch = "deeponet" if rec.get("prefix", "").startswith("deeponet") else "fno"
        if arch == model and rec.get("loss_type") == loss:
            return True
    return False


def stage_lr(dry=False):
    """Learning-rate sweep for the new arms, at 1/5 the budget."""
    jobs, skipped = [], 0
    for model, loss, extra in NEW_ARMS:
        for lr in lr_grid(model):
            # separate dir per lr: the run prefix does not carry the lr, so a shared dir would
            # have the cells overwrite each other
            out_dir = os.path.join(OUT, "poisson", "lr", f"lr{lr}")
            if _have_run(out_dir, model, loss):
                skipped += 1
                continue
            name = f"lr_{model}_{loss}_lr{lr}"
            argv = POISSON_BASE + ["--model", model, "--loss", loss, "--epochs", "100",
                                   "--lr", lr, "--seed", "42",
                                   "--output_dir", out_dir] + extra
            jobs.append((name, argv))
    if skipped:
        print(f"  ({skipped} cells already on disk, skipped)", flush=True)
    return _dispatch(jobs, dry)


def _best_lrs():
    """Read the lr stage's winners; fall back to 1e-3 for anything not swept."""
    import glob, json
    best = {}
    for path in glob.glob(os.path.join(OUT, "poisson", "lr", "*", "results", "*.json")):
        with open(path) as fh:
            rec = json.load(fh)
        lr = path.split(os.sep + "lr")[-1].split(os.sep)[0]
        arch = "deeponet" if rec.get("prefix", "").startswith("deeponet") else "fno"
        loss = rec.get("loss_type", "")
        key = (arch, loss)
        val = rec["stats"]["best_val_error"]
        if key not in best or val < best[key][1]:
            best[key] = (lr, val)
    return {k: v[0] for k, v in best.items()}


def stage_ablation(dry=False):
    """PINO ingredient 2x2: mollifier on/off x reduction rel/mse.

    The all-off corner should reproduce the bare `galerkin` arm (up to the measured
    A = (2/3)h^2 L scaling and the factor-2 conditioning) -- a cross-check on the whole port,
    in the same spirit as `blend t=0 == galerkin` for the Stokes blend family.
    """
    lr = _best_lrs().get(("fno", "pino"), "1e-3")
    jobs = []
    for mol in ("on", "off"):
        for red in ("rel", "mse"):
            if mol == "on" and red == "rel":
                continue                     # that cell is PINO as published, already in `poisson`
            name = f"abl_pino_mol-{mol}_red-{red}"
            argv = POISSON_BASE + ["--model", "fno", "--loss", "pino", "--epochs", "500",
                                   "--lr", lr, "--seed", "42",
                                   "--mollify", mol, "--pino_reduction", red,
                                   "--output_dir", os.path.join(OUT, "poisson", "ablation",
                                                                f"mol-{mol}_red-{red}")]
            jobs.append((name, argv))
    return _dispatch(jobs, dry)


def stage_poisson(dry=False):
    """The 7-arm head-to-head table, 3 seeds."""
    best = _best_lrs()
    jobs = []
    for seed in SEEDS:
        for model, loss, extra in POISSON_ARMS:
            lr = best.get((model, loss), "1e-3")
            name = f"poisson_s{seed}_{model}_{loss}"
            argv = POISSON_BASE + ["--model", model, "--loss", loss, "--epochs", "500",
                                   "--lr", lr, "--seed", str(seed),
                                   "--output_dir", os.path.join(OUT, "poisson", f"seed{seed}")] + extra
            jobs.append((name, argv))
    return _dispatch(jobs, dry)


def stage_href(dry=False):
    r"""Mesh refinement -- the experiment that separates the two explanations.

    The first Poisson results showed PINO reaching 0.45 % while our bare FEM residual sits at
    27.7 %, and the ingredient ablation pinned the entire difference on the **mollifier**
    (hard zero-Dirichlet ansatz), not on the reduction and not on conditioning: with it 0.46 %,
    without it ~50 %, while the reduction changes nothing.

    Two readings survive that, and they differ under refinement:

    * *representational* -- the FNO's FFT assumes periodicity, and a zero-Dirichlet solution
      extended periodically has a kink at the boundary; the mollifier factors that kink out.
      Then the mollifier keeps helping at every h.
    * *conditioning* -- the bare residual's Hessian grows as O(h^-4) regardless of any fixed
      reparametrization, so at fine enough h the mollifier must stop being enough.

    Three losses x three grids settles it, and the mollified-vs-bare pair at fixed loss is the
    single-variable comparison. Mollified and bare runs need separate output directories: both
    record ``loss_type='galerkin'`` and would otherwise share a run prefix and overwrite.
    """
    arms = [("pino", "pino", []),                       # mollifier on by default (--mollify auto)
            ("galerkin", "galerkin", []),               # bare FEM residual
            ("galerkin-moll", "galerkin", ["--mollify", "on"])]
    jobs, skipped = [], 0
    for tag, loss, extra in arms:
        lr = _best_lrs().get(("fno", loss), "1e-3")
        for gr in (33, 65, 129):
            out_dir = os.path.join(OUT, "href", tag, f"gr{gr}")
            # The queue's first pass wrote PINO into href/gr<N>; accept either layout.
            legacy = os.path.join(OUT, "href", f"gr{gr}")
            if _have_run(out_dir, "fno", loss) or (tag == "pino" and _have_run(legacy, "fno", loss)):
                skipped += 1
                continue
            name = f"href_{tag}_gr{gr}"
            argv = ["--pde", "poisson", "--model", "fno", "--loss", loss,
                    "--grid_resolution", str(gr),
                    "--n_train", "1024", "--n_val", "128", "--n_test", "256",
                    "-k", "4", "--batch_size", "32", "--epochs", "500",
                    "--lr", lr, "--lr_min", "1e-6", "--optimizer", "adam",
                    "--seed", "42", "--device", "cuda",
                    "--output_dir", out_dir] + extra
            jobs.append((name, argv))
    if skipped:
        print(f"  ({skipped} cells already on disk, skipped)", flush=True)
    return _dispatch(jobs, dry)


def stage_ac(dry=False):
    """The Allen-Cahn table. Dataset built ONCE per seed, all six arms run in-process."""
    def _todo(seed):
        """Arms for this seed that are not already on disk (the dataset build is expensive,
        so a seed with nothing left to do is skipped entirely rather than rebuilt)."""
        d = os.path.join(OUT, "allen_cahn", f"seed{seed}", "results")
        return [a for a in AC_ARMS if not glob.glob(os.path.join(d, a["glob"] + ".json"))]

    if dry:
        for seed in SEEDS:
            todo = _todo(seed)
            if not todo:
                print(f"  [ac] seed {seed}: complete, skipped")
                continue
            print(f"  [ac] seed {seed}: build dataset once, then "
                  + ", ".join(f"{a['model']}/{a['loss']}"
                              + ("+mg" if a["precond"] != "none" else "")
                              + ("+mm" if a["form"] != "galerkin" else "") for a in todo))
        return True

    import torch
    from tensorpils import cli as _cli
    from tensorpils.data import create_ac_datasets
    from tensorpils.trainer import ACTrainer
    from tensorpils.baselines import (PINOACTrainer, PIDeepONetACTrainer, DeepONetACTrainer)

    os.makedirs(LOGS, exist_ok=True)
    ok = True
    for seed in SEEDS:
        todo = _todo(seed)
        if not todo:
            print(f"  [{time.strftime('%H:%M:%S')}] ac seed {seed}: already complete, skipped",
                  flush=True)
            continue
        print(f"  [{time.strftime('%H:%M:%S')}] ac seed {seed}: building FEM Newton reference "
              f"(once for {len(todo)} arm(s)) ...", end="", flush=True)
        t0 = time.time()
        tr, va, te = create_ac_datasets(
            n_train=AC_BASE["n_train"], n_val=AC_BASE["n_val"], n_test=AC_BASE["n_test"],
            K=AC_BASE["K"], grid_resolution=AC_BASE["grid_resolution"], dt=AC_BASE["dt"],
            n_steps=AC_BASE["n_steps"], a=AC_BASE["a"], eps=AC_BASE["eps"],
            integrator="convex_concave", seed=seed)
        print(f" {(time.time()-t0)/60:.1f} min", flush=True)

        out_dir = os.path.join(OUT, "allen_cahn", f"seed{seed}")
        for arm in todo:
            model_name, loss = arm["model"], arm["loss"]
            precond, form = arm["precond"], arm["form"]
            tag = (f"ac_s{seed}_{model_name}_{loss}"
                   + ("_mg" if precond != "none" else "")
                   + ("_mm" if form != "galerkin" else ""))
            print(f"  [{time.strftime('%H:%M:%S')}] {tag} ...", end="", flush=True)
            t0 = time.time()
            try:
                # Reuse the CLI's own parser + model builder so the model is byte-identical to
                # what `--pde ac --model ... --loss ...` would have produced.
                args = _cli.build_parser().parse_args([
                    "--pde", "ac", "--model", model_name, "--loss", loss,
                    "--ac_precond", precond, "--grid_resolution",
                    str(AC_BASE["grid_resolution"]), "--seed", str(seed),
                    "--bptt_mode", AC_BASE["bptt_mode"],
                ])
                torch.manual_seed(seed)
                net = _cli._build_model(args, in_channels=1, train_ds=tr)
                lg = 0.0 if loss == "data" else 1.0
                ld = 1.0 if loss == "data" else 0.0
                common = dict(model=net, train_dataset=tr, val_dataset=va, test_dataset=te,
                              loss_type=loss, optimizer_name="adam", lr=1e-3, lr_min=1e-6,
                              batch_size=AC_BASE["batch_size"], epochs=AC_BASE["epochs"],
                              device="cuda", output_dir=out_dir,
                              lambda_galerkin=lg, lambda_data=ld,
                              rollout_steps=AC_BASE["rollout_steps"],
                              bptt_mode=AC_BASE["bptt_mode"],
                              ac_loss_form=form,
                              ac_precond=("" if precond == "none" else precond))
                cls = {"pino": PINOACTrainer, "pideeponet": PIDeepONetACTrainer}.get(
                    loss, DeepONetACTrainer if model_name == "deeponet" else ACTrainer)
                cls(**common).train()
                print(f" ok  ({(time.time()-t0)/60:.1f} min)", flush=True)
            except Exception as exc:                       # keep the queue going
                ok = False
                print(f" FAILED ({type(exc).__name__}: {exc})", flush=True)
                import traceback
                with open(os.path.join(LOGS, f"{tag}.log"), "w") as fh:
                    traceback.print_exc(file=fh)
    return ok


def _dispatch(jobs, dry):
    if dry:
        for name, argv in jobs:
            print(f"  {name}")
        return True
    ok = True
    for name, argv in jobs:
        ok &= _run_cli(name, argv)
    return ok


STAGES = {"lr": stage_lr, "ablation": stage_ablation, "poisson": stage_poisson,
          "href": stage_href, "ac": stage_ac}
ORDER = ["lr", "ablation", "poisson", "href", "ac"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", default=None,
                    help=f"subset of {ORDER} (default: all, in that order)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stages = args.stages or ORDER
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; expected any of {ORDER}")

    t0 = time.time()
    for s in stages:
        print(f"\n=== stage: {s} ===", flush=True)
        STAGES[s](dry=args.dry_run)
    print(f"\n=== queue finished in {(time.time()-t0)/3600:.2f} h ===", flush=True)


if __name__ == "__main__":
    main()
