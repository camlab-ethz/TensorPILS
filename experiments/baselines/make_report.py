"""Render the live baseline-queue report from whatever is on disk right now.

Reads ``logs/baselines/QUEUE.log`` for run status and ``output/baselines/**/results/*.json``
for numbers, and writes a self-contained HTML page. Safe to run at any point — runs that have
not finished simply show as pending, so the page is meaningful from the moment the queue starts.

    python experiments/baselines/make_report.py -o /path/to/report.html
"""

import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import run_all as RA                                                # single source of truth

OUT = os.path.join(REPO, "output", "baselines")
QUEUE_LOG = os.path.join(REPO, "logs", "baselines", "QUEUE.log")

# Arm identity: colour family + display name. Mirrors the matplotlib STYLE dicts so the page
# and the figures read as one system.
ARM = {
    ("fno", "data"):            ("ref",  "FNO · supervised"),
    ("fno", "galerkin"):        ("bare", "FNO · bare FEM LS"),
    ("fno", "pino"):            ("pino", "FNO · PINO"),
    ("fno", "pls"):             ("ours", "FNO · preconditioned LS"),
    ("fno", "min_movement"):    ("mm",   "FNO · minimizing movement"),
    ("deeponet", "data"):       ("ref",  "DeepONet · supervised"),
    ("deeponet", "pideeponet"): ("pid",  "DeepONet · PI-DeepONet"),
    ("deeponet", "pls"):        ("ours", "DeepONet · preconditioned LS"),
}
ORDER = [("fno", "data"), ("fno", "min_movement"), ("fno", "pls"), ("fno", "pino"),
         ("fno", "galerkin"), ("deeponet", "data"), ("deeponet", "pls"),
         ("deeponet", "pideeponet")]

# Measured before the queue ran (float64, interior blocks) — the prediction the runs test.
COND = [("17²", 51.7, 103.1, "2.7e3", "1.1e4"),
        ("33²", 207.3, 414.3, "4.3e4", "1.7e5"),
        ("65²", 829.9, 1659.0, "6.9e5", "2.8e6")]
DISCR = [("33²", "1.049e-2", "1.056e-2"), ("65²", "2.566e-3", "2.570e-3")]
TIMING = [("FNO · supervised", 0.50, 4.6), ("FNO · bare FEM LS", 0.53, 4.6),
          ("FNO · PINO", 0.53, 4.6), ("FNO · preconditioned", 0.89, 7.6),
          ("DeepONet · supervised", 0.07, 0.6), ("DeepONet · preconditioned", 0.56, 4.8),
          ("DeepONet · PI-DeepONet", 3.34, 27.9)]


# --------------------------------------------------------------------- state

def expected_runs():
    """Every run the queue will attempt, in order, as (stage, name, arm-key)."""
    runs = []
    for m, l, _ in RA.NEW_ARMS:
        for lr in RA.lr_grid(m):
            runs.append(("lr", f"lr_{m}_{l}_lr{lr}", (m, l)))
    for mol in ("on", "off"):
        for red in ("rel", "mse"):
            if mol == "on" and red == "rel":
                continue
            runs.append(("ablation", f"abl_pino_mol-{mol}_red-{red}", ("fno", "pino")))
    for seed in RA.SEEDS:
        for m, l, _ in RA.POISSON_ARMS:
            runs.append(("poisson", f"poisson_s{seed}_{m}_{l}", (m, l)))
    for tag, loss in (("pino", "pino"), ("galerkin", "galerkin"), ("galerkin-moll", "galerkin")):
        for gr in (33, 65, 129):
            runs.append(("href", f"href_{tag}_gr{gr}", ("fno", loss)))
    for seed in RA.SEEDS:
        for a in RA.AC_ARMS:
            m, l, p, f = a["model"], a["loss"], a["precond"], a["form"]
            runs.append(("ac",
                         f"ac_s{seed}_{m}_{l}" + ("_mg" if p != "none" else "")
                         + ("_mm" if f != "galerkin" else ""),
                         (m, "pls" if p != "none" else l)))
    return runs


def queue_state():
    """``name -> ('ok'|'failed', minutes)`` across every queue log, plus what is running now.

    Two wrinkles the naive parse gets wrong, both learned the hard way:

    * the study ran as several queues (``QUEUE.log``, ``QUEUE-run1.log``, …) as stages were
      re-scoped, so reading only the newest one reports a fraction of the real progress;
    * the shelled-out Poisson runs put their ``ok (N min)`` on the same line as the run name,
      but the in-process Allen-Cahn arms interleave training output in between, so their
      completion lands on a *later* physical line. Attributing a bare completion marker to the
      most recent un-finished name covers both.
    """
    done, running, stage = {}, None, None
    logs = sorted(glob.glob(os.path.join(os.path.dirname(QUEUE_LOG), "QUEUE*.log")),
                  key=lambda p: (os.path.basename(p) != "QUEUE.log", p))
    # oldest first: run1, run2, ..., then the current QUEUE.log
    logs = sorted(logs, key=lambda p: os.path.getmtime(p))
    pending = None
    for path in logs:
        for line in open(path, errors="replace"):
            line = line.rstrip("\n")
            m = re.match(r"=== stage: (\w+) ===", line.strip())
            if m:
                stage = m.group(1)
                continue
            m = re.search(r"\]\s+(\S+)\s+\.\.\.", line)
            if m:
                pending = m.group(1)
                running = pending
            ok = re.search(r"\bok\s+\(([\d.]+) min\)", line)
            if ok and pending:
                done[pending] = ("ok", float(ok.group(1)))
                pending, running = None, None
                continue
            if "FAILED" in line and pending:
                done[pending] = ("failed", None)
                pending, running = None, None
    # A queue that has printed its summary line is finished, whatever the last name was.
    if any("queue finished" in open(p, errors="replace").read() for p in logs[-1:]):
        running = None
    return done, running, stage


def load_results(subdir):
    recs = []
    for path in glob.glob(os.path.join(OUT, subdir, "**", "results", "*.json"), recursive=True):
        with open(path) as fh:
            r = json.load(fh)
        r["_path"] = path
        pre = r.get("prefix", "")
        r["_arch"] = "deeponet" if pre.startswith("deeponet") else "fno"
        lt = r.get("loss_type", "")
        # Several Allen-Cahn arms share loss_type='galerkin' and are told apart only by the
        # tags the trainer adds: '_precmg' (preconditioned LS), '_mm_' (minimizing movement).
        r["_loss"] = ("pls" if "_precmg" in pre else
                      "min_movement" if "_mm_" in pre else lt)
        m = re.search(r"seed(\d+)", path)
        r["_seed"] = int(m.group(1)) if m else None
        m = re.search(r"[/\\]lr([0-9.e+-]+)[/\\]", path)
        r["_lr"] = m.group(1) if m else None
        recs.append(r)
    return recs


def summarize(recs, key):
    """(arch, loss) -> (mean, half-spread, n) over seeds for metric `key`."""
    from collections import defaultdict
    by = defaultdict(list)
    for r in recs:
        if key in r:
            by[(r["_arch"], r["_loss"])].append(r[key])
    return {k: (sum(v) / len(v), (max(v) - min(v)) / 2, len(v)) for k, v in by.items()}


# --------------------------------------------------------------------- render

def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def chips(runs, done, running):
    out = []
    for _, name, key in runs:
        fam = ARM.get(key, ("ref", ""))[0]
        st = done.get(name)
        cls = "chip " + fam
        if st and st[0] == "ok":
            cls += " done"
            t = f"{name} — {st[1]:.1f} min" if st[1] else name
        elif st:
            cls += " failed"
            t = f"{name} — failed"
        elif name == running:
            cls += " running"
            t = f"{name} — running now"
        else:
            t = f"{name} — pending"
        out.append(f'<i class="{cls}" title="{esc(t)}"></i>')
    return "".join(out)


def result_table(summary, metric_label, empty_note):
    if not summary:
        return f'<p class="empty">{esc(empty_note)}</p>'
    order = ORDER
    meta = {"data": ("yes", "—"), "galerkin": ("no", "FEM"), "pls": ("no", "FEM + P"),
            "min_movement": ("no", "FEM energy"),
            "pino": ("no", "finite diff."), "pideeponet": ("no", "autodiff")}
    rows = []
    best = min((v[0] for v in summary.values()), default=None)
    for k in order:
        if k not in summary:
            continue
        mean, spread, n = summary[k]
        fam, name = ARM.get(k, ("ref", "/".join(k)))
        labels, deriv = meta.get(k[1], ("?", "?"))
        cell = f"{mean*100:.2f}" + (f" <span class='pm'>± {spread*100:.2f}</span>" if n > 1 else "")
        mark = " best" if mean == best else ""
        rows.append(
            f'<tr class="{fam}{mark}"><td><span class="dot"></span>{esc(name)}</td>'
            f'<td>{labels}</td><td class="mono">{deriv}</td>'
            f'<td class="num">{cell}</td><td class="num faint">{n}</td></tr>')
    return (f'<div class="scroll"><table><thead><tr><th>arm</th><th>labels</th>'
            f'<th>derivative</th><th class="num">{esc(metric_label)}</th>'
            f'<th class="num">seeds</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def lr_table(recs):
    from collections import defaultdict
    by = defaultdict(dict)
    for r in recs:
        if r["_lr"]:
            by[(r["_arch"], r["_loss"])][r["_lr"]] = r["stats"]["best_val_error"]
    if not by:
        return '<p class="empty">No learning-rate runs have finished yet.</p>'
    lrs = sorted({lr for row in by.values() for lr in row}, key=float)
    head = "".join(f'<th class="num mono">{esc(l)}</th>' for l in lrs)
    rows = []
    for k, row in by.items():
        fam, name = ARM.get(k, ("ref", "/".join(k)))
        best = min(row, key=row.get)
        cells = "".join(
            f'<td class="num{" win" if l == best else ""}">'
            + (f"{row[l]:.2e}" if l in row else "·") + "</td>" for l in lrs)
        rows.append(f'<tr class="{fam}"><td><span class="dot"></span>{esc(name)}</td>{cells}'
                    f'<td class="num mono win">{esc(best)}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>arm</th>' + head
            + '<th class="num">chosen</th></tr></thead><tbody>' + "".join(rows)
            + "</tbody></table></div>")


def _href_cells():
    """tag -> {grid: test rel-L2 %}. Accepts the queue's first-pass layout (href/gr<N>) too."""
    out = {}
    for tag in ("pino", "galerkin", "galerkin-moll"):
        cells = {}
        pats = [os.path.join(OUT, "href", tag, "**", "results", "*.json")]
        if tag == "pino":
            pats.append(os.path.join(OUT, "href", "gr*", "results", "*.json"))
        for pat in pats:
            for p in glob.glob(pat, recursive=True):
                gr = next((s for s in p.split(os.sep) if s.startswith("gr")), None)
                if gr:
                    cells[int(gr[2:])] = json.load(open(p))["test_rl2"] * 100
        if cells:
            out[tag] = cells
    return out


def href_table():
    cells = _href_cells()
    if not cells:
        return '<p class="empty">No mesh-refinement runs have finished yet.</p>'
    label = {"pino": ("pino", "PINO — finite diff. + mollifier"),
             "galerkin": ("bare", "bare FEM residual"),
             "galerkin-moll": ("ours", "FEM residual + mollifier")}
    rows = []
    for tag in ("pino", "galerkin-moll", "galerkin"):
        if tag not in cells:
            continue
        fam, name = label[tag]
        c = cells[tag]
        tds = "".join(f'<td class="num">{c[g]:.2f}</td>' if g in c else '<td class="num">·</td>'
                      for g in (33, 65, 129))
        growth = (f"{c[129]/c[33]:.1f}×" if 33 in c and 129 in c and c[33] else "·")
        rows.append(f'<tr class="{fam}"><td><span class="dot"></span>{esc(name)}</td>{tds}'
                    f'<td class="num mono">{growth}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>arm</th>'
            '<th class="num">33²</th><th class="num">65²</th><th class="num">129²</th>'
            '<th class="num">33²→129²</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def mollifier_table(poisson_summary):
    """The decisive single-variable comparison plus the PINO ingredient 2x2."""
    rows = []
    dec = glob.glob(os.path.join(OUT, "decisive", "galerkin_mollified", "results", "*.json"))
    bare = poisson_summary.get(("fno", "galerkin"))
    pls = poisson_summary.get(("fno", "pls"))
    sup = poisson_summary.get(("fno", "data"))
    def row(name, fam, val, note=""):
        if val is None:
            return
        rows.append(f'<tr class="{fam}"><td><span class="dot"></span>{esc(name)}</td>'
                    f'<td class="num">{val:.2f}</td><td class="faint">{esc(note)}</td></tr>')
    if bare:
        row("bare FEM residual", "bare", bare[0] * 100, "no hard BC — boundary only projected at eval")
    if dec:
        row("bare FEM residual + mollifier", "ours",
            json.load(open(dec[0]))["test_rl2"] * 100, "the only variable changed")
    if pls:
        row("preconditioned FEM residual", "ours", pls[0] * 100, "multigrid P, no mollifier")
    if sup:
        row("supervised", "ref", sup[0] * 100, "reference, uses labels")
    if not rows:
        return '<p class="empty">The decisive run has not finished yet.</p>'
    abl = []
    for p in sorted(glob.glob(os.path.join(OUT, "poisson", "ablation", "*", "results",
                                           "*.json"))):
        cell = p.split(os.sep + "ablation" + os.sep)[1].split(os.sep)[0]
        mol = "on" if "mol-on" in cell else "off"
        red = "relative Lp" if "red-rel" in cell else "mean square"
        abl.append((mol, red, json.load(open(p))["test_rl2"] * 100))
    pino = poisson_summary.get(("fno", "pino"))
    if pino:
        abl.append(("on", "relative Lp", pino[0] * 100))
    abl_rows = "".join(
        f'<tr class="pino"><td class="mono">{m}</td><td class="mono">{r}</td>'
        f'<td class="num">{v:.2f}</td></tr>'
        for m, r, v in sorted(abl, key=lambda t: (t[0] != "on", t[1])))
    return (f'<div class="scroll"><table><thead><tr><th>Poisson 64², one variable at a time</th>'
            f'<th class="num">rel. L² (%)</th><th></th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div>'
            + f'<p class="note" style="margin-top:6px">And inside PINO itself — the mollifier '
              f'is worth two orders of magnitude, the reduction nothing:</p>'
            + f'<div class="scroll"><table><thead><tr><th>mollifier</th><th>reduction</th>'
              f'<th class="num">rel. L² (%)</th></tr></thead><tbody>{abl_rows}'
              f'</tbody></table></div>')


def build(path):
    runs = expected_runs()
    done, running, stage = queue_state()
    n_ok = sum(1 for _, n, _ in runs if done.get(n, ("", ))[0] == "ok")
    n_fail = sum(1 for _, n, _ in runs if done.get(n, ("", ))[0] == "failed")
    pct = 100.0 * n_ok / len(runs)
    mins = sum(v[1] for v in done.values() if v[0] == "ok" and v[1])

    poisson = load_results("poisson")
    main_poisson = [r for r in poisson if r["_seed"] is not None]
    lr_runs = [r for r in poisson if r["_lr"]]
    ac = load_results("allen_cahn")

    stages = [
        ("lr", "Learning-rate sweep",
         "Each new arm tuned on its own. PINO's relative-Lp reduction and our sum-over-nodes "
         "losses differ in gradient scale by orders of magnitude, so a shared learning rate "
         "would quietly handicap one side — and an untuned baseline is the fastest way to lose "
         "a reviewer."),
        ("ablation", "PINO ingredient ablation",
         "Mollifier on/off × reduction relative/squared. If PINO beats the bare FEM arm, the "
         "only available explanations are these two — not conditioning. The all-off corner "
         "should reproduce the bare arm, which cross-checks the whole port."),
        ("poisson", "Poisson head-to-head",
         "The deliverable: seven arms, three seeds. Within each architecture block only the "
         "loss changes, so the loss effect is not confounded with the architecture."),
        ("href", "PINO under mesh refinement",
         "Distinguishes a better constant from better scaling. This is what ties the baseline "
         "to the paper's central conditioning claim rather than merely adding a table row."),
        ("ac", "Allen–Cahn head-to-head",
         "The nonlinear, time-dependent case — and the one where labels are genuinely "
         "expensive, so the label-free argument actually bites."),
    ]
    stage_html = []
    for sid, title, blurb in stages:
        sruns = [r for r in runs if r[0] == sid]
        sok = sum(1 for _, n, _ in sruns if done.get(n, ("",))[0] == "ok")
        live = " live" if sid == stage and running else ""
        stage_html.append(f"""
      <article class="stage{live}">
        <header>
          <h3>{esc(title)}</h3>
          <span class="count mono">{sok}/{len(sruns)}</span>
        </header>
        <p>{esc(blurb)}</p>
        <div class="chips">{chips(sruns, done, running)}</div>
      </article>""")

    cond_rows = "".join(
        f'<tr><td class="mono">{g}</td><td class="num">{a:.1f}</td><td class="num">{l:.1f}</td>'
        f'<td class="num mono">{a2}</td><td class="num mono">{l2}</td></tr>'
        for g, a, l, a2, l2 in COND)
    discr_rows = "".join(
        f'<tr><td class="mono">{g}</td><td class="num mono">{fd}</td>'
        f'<td class="num mono">{fe}</td></tr>' for g, fd, fe in DISCR)
    timing_rows = "".join(
        f'<tr><td>{esc(n)}</td><td class="num">{s:.2f}</td><td class="num">{m:.1f}</td></tr>'
        for n, s, m in TIMING)

    html = f"""<title>Does Preconditioning Beat PINO?</title>
<style>
:root {{
  --paper:#f6f7f9; --surface:#ffffff; --sunk:#eef1f5;
  --ink:#12151c; --ink-2:#454c5a; --ink-3:#79808f; --rule:#dee3ea;
  --ours:#0e7490; --pino:#ea580c; --pid:#a21caf; --bare:#b91c1c; --ref:#12151c;
  --ok:#15803d; --bad:#b91c1c;
  --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#0d1016; --surface:#151a22; --sunk:#10151c;
    --ink:#e6e9ef; --ink-2:#a3abb9; --ink-3:#6b7382; --rule:#262d38;
    --ours:#22a5c0; --pino:#fb923c; --pid:#d946ef; --bare:#ef4444; --ref:#e6e9ef;
    --ok:#4ade80; --bad:#ef4444;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#0d1016; --surface:#151a22; --sunk:#10151c;
  --ink:#e6e9ef; --ink-2:#a3abb9; --ink-3:#6b7382; --rule:#262d38;
  --ours:#22a5c0; --pino:#fb923c; --pid:#d946ef; --bare:#ef4444; --ref:#e6e9ef;
  --ok:#4ade80; --bad:#ef4444;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.65;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:980px; margin:0 auto; padding:48px 24px 96px;
        display:flex; flex-direction:column; gap:44px; }}
.eyebrow {{ font-family:var(--mono); font-size:11px; letter-spacing:.16em;
           text-transform:uppercase; color:var(--ink-3); margin:0 0 10px; }}
h1 {{ font-family:var(--mono); font-size:clamp(28px,4.4vw,42px); line-height:1.12;
     letter-spacing:-.02em; font-weight:600; margin:0; text-wrap:balance; }}
h2 {{ font-family:var(--mono); font-size:15px; letter-spacing:.06em; text-transform:uppercase;
     font-weight:600; margin:0 0 4px; color:var(--ink); }}
h3 {{ font-size:16px; font-weight:600; margin:0; letter-spacing:-.01em; }}
p {{ margin:0; color:var(--ink-2); max-width:66ch; }}
section {{ display:flex; flex-direction:column; gap:16px; }}
.lede {{ font-size:17px; color:var(--ink-2); max-width:64ch; }}

/* ---- status band ---- */
.band {{ background:var(--surface); border:1px solid var(--rule); border-radius:4px;
        padding:22px 24px; display:flex; flex-direction:column; gap:16px; }}
.bar {{ height:6px; background:var(--sunk); border-radius:3px; overflow:hidden; }}
.bar > i {{ display:block; height:100%; width:{pct:.2f}%; background:var(--ours);
           transition:width .4s ease; }}
.stats {{ display:flex; flex-wrap:wrap; gap:28px; }}
.stat b {{ display:block; font-family:var(--mono); font-size:26px; font-weight:600;
          letter-spacing:-.02em; font-variant-numeric:tabular-nums; }}
.stat span {{ font-family:var(--mono); font-size:11px; letter-spacing:.12em;
             text-transform:uppercase; color:var(--ink-3); }}

/* ---- stage cards ---- */
.stages {{ display:flex; flex-direction:column; gap:14px; }}
.stage {{ background:var(--surface); border:1px solid var(--rule); border-radius:4px;
         padding:18px 20px; display:flex; flex-direction:column; gap:12px; }}
.stage.live {{ border-color:var(--ours); box-shadow:0 0 0 1px var(--ours); }}
.stage > header {{ display:flex; align-items:baseline; justify-content:space-between; gap:16px; }}
.stage p {{ font-size:14.5px; }}
.count {{ font-size:13px; color:var(--ink-3); font-variant-numeric:tabular-nums; }}
.chips {{ display:flex; flex-wrap:wrap; gap:4px; }}
.chip {{ width:15px; height:15px; border-radius:2px; display:block;
        border:1.5px solid var(--ink-3); opacity:.4; }}
.chip.done {{ opacity:1; border-color:transparent; }}
.chip.ours.done {{ background:var(--ours); }}  .chip.pino.done {{ background:var(--pino); }}
.chip.pid.done  {{ background:var(--pid);  }}  .chip.bare.done {{ background:var(--bare); }}
.chip.ref.done  {{ background:var(--ref);  }}  .chip.mm.done   {{ background:#15803d; }}
.chip.failed {{ opacity:1; background:var(--bad); border-color:var(--bad); }}
.chip.running {{ opacity:1; border-color:var(--ours); border-width:2px;
                animation:pulse 1.4s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:.3; }} }}
@media (prefers-reduced-motion:reduce) {{ .chip.running {{ animation:none; }} }}

/* ---- tables ---- */
.scroll {{ overflow-x:auto; border:1px solid var(--rule); border-radius:4px;
          background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:14.5px; }}
th, td {{ padding:10px 14px; text-align:left; border-bottom:1px solid var(--rule);
         white-space:nowrap; }}
thead th {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.1em;
           text-transform:uppercase; color:var(--ink-3); font-weight:600;
           background:var(--sunk); }}
tbody tr:last-child td {{ border-bottom:0; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.mono {{ font-family:var(--mono); font-size:13px; }}
.faint {{ color:var(--ink-3); }}
.pm {{ color:var(--ink-3); font-size:12.5px; }}
.win {{ color:var(--ok); font-weight:600; }}
tr.best td {{ font-weight:600; }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%;
       margin-right:9px; vertical-align:baseline; background:var(--ink-3); }}
tr.ours .dot {{ background:var(--ours); }} tr.pino .dot {{ background:var(--pino); }}
tr.pid  .dot {{ background:var(--pid);  }} tr.bare .dot {{ background:var(--bare); }}
tr.ref  .dot {{ background:var(--ref);  }} tr.mm   .dot {{ background:#15803d; }}
.empty {{ font-size:14.5px; color:var(--ink-3); font-style:italic;
         padding:16px 18px; border:1px dashed var(--rule); border-radius:4px; }}
.note {{ font-size:14px; color:var(--ink-3); border-left:2px solid var(--rule);
        padding-left:14px; max-width:64ch; }}
footer {{ font-family:var(--mono); font-size:11.5px; color:var(--ink-3);
         border-top:1px solid var(--rule); padding-top:18px; }}
a {{ color:var(--ours); }}
:focus-visible {{ outline:2px solid var(--ours); outline-offset:2px; }}
</style>

<div class="wrap">

  <header>
    <p class="eyebrow">TensorPILS · ICLR 2027 · baseline study</p>
    <h1>Does preconditioning beat PINO?</h1>
    <p class="lede" style="margin-top:14px">The paper claims preconditioned physics-informed
    training outperforms previous approaches, but every label-free arm it compares against is
    our own negative control. This queue puts two published methods — <strong>PINO</strong> and
    <strong>PI-DeepONet</strong> — into the same table, scored by the same FEM relative-L²
    metric as every existing result.</p>
  </header>

  <div class="band">
    <div class="stats">
      <div class="stat"><b>{n_ok}<span style="color:var(--ink-3)">/{len(runs)}</span></b>
        <span>runs complete</span></div>
      <div class="stat"><b>{pct:.0f}%</b><span>of queue</span></div>
      <div class="stat"><b>{mins/60:.1f} h</b><span>compute so far</span></div>
      <div class="stat"><b style="color:{'var(--bad)' if n_fail else 'var(--ink)'}">{n_fail}</b>
        <span>failed</span></div>
    </div>
    <div class="bar"><i></i></div>
    <p class="mono" style="color:var(--ink-3);font-size:12px">
      {'running · ' + esc(running) if running else ('queue idle' if n_ok == len(runs) else 'queue not started')}
      · single RTX 4090, sequential
    </p>
  </div>

  <section>
    <h2>The prediction, made before the runs</h2>
    <p>PINO differentiates with finite differences; we use the FEM weak form. Both operators
    are conditioned as <span class="mono">O(h⁻²)</span> — measured rate −2.00 per halving — so
    both least-squares Hessians are <span class="mono">O(h⁻⁴)</span>. PINO's stencil is in fact
    a factor 2 <em>worse</em> conditioned, uniformly in h. <strong>So PINO is predicted to land
    near the bare FEM arm, not near the preconditioned one</strong> — and that is the outcome
    we want: it turns “our own control fails” into “the published method fails, for the reason
    we identify”.</p>
    <div class="scroll"><table>
      <thead><tr><th>grid</th><th class="num">κ(A) FEM</th><th class="num">κ(L) finite diff.</th>
        <th class="num">κ(A²) ours</th><th class="num">κ(L²) PINO</th></tr></thead>
      <tbody>{cond_rows}</tbody></table></div>
    <p>The comparison is only meaningful if the two residuals are the <em>same PDE at the same
    accuracy</em>. Measured against the analytical solution, they are — agreeing to 0.7 %, both
    converging at rate 2.03:</p>
    <div class="scroll"><table>
      <thead><tr><th>grid</th><th class="num">−Δ<sub>FD</sub>u* vs f</th>
        <th class="num">A u* vs M f</th></tr></thead>
      <tbody>{discr_rows}</tbody></table></div>
  </section>

  <section>
    <h2>What actually made physics-informed Poisson training work</h2>
    <p>The prediction above was wrong, and the way it was wrong is the result. PINO does not
    land near the bare FEM residual — it lands <em>ahead of supervised training</em>. The
    ingredient ablation puts the entire difference on one thing PINO's reference implementation
    happens to do and our bare arm does not: multiplying the output by
    <span class="mono">sin(πx)·sin(πy)</span>, imposing the zero Dirichlet condition by
    construction. Hand our own residual the same ansatz and it closes the gap completely.</p>
    {mollifier_table(summarize(main_poisson, "test_rl2"))}
    <p class="note">A plausible mechanism, not yet separately tested: the FNO's spectral layers
    assume periodicity, and a zero-Dirichlet solution extended periodically has a kink at the
    boundary. The mollifier factors that kink out. Supervised training never needed the help,
    which fits — only an ill-conditioned objective is fragile enough for the representation to
    become the binding constraint.</p>
  </section>

  <section>
    <h2>Does it survive refinement?</h2>
    <p>This is what separates a better constant from better scaling. Conditioning says the bare
    residual's Hessian grows as <span class="mono">O(h⁻⁴)</span> and that no fixed
    reparametrisation changes that, so the mollifier should eventually stop rescuing it.</p>
    {href_table()}
    <p class="note">The bare residual degrades exactly as predicted, and badly. The mollified
    ones hold. Caveat worth carrying into the paper: the two <span class="mono">galerkin</span>
    rows ran at lr 1e-3 while PINO ran at its tuned 3e-3, so part of the gap between the
    mollified rows at 129² is tuning rather than formulation.</p>
  </section>

  <section>
    <h2>Poisson head-to-head</h2>
    {result_table(summarize(main_poisson, "test_rl2"), "test rel. L² (%)",
                  "No Poisson table runs have finished yet — this stage starts after the "
                  "learning-rate sweep and the ablation.")}
    <p class="note">Lower is better. The supervised rows are the reference, not competitors:
    the question is which label-free arm reaches them. Inside the DeepONet block the loss makes
    no difference at all — that architecture is the binding constraint at ≈2.3 %, which is
    precisely what the supervised-DeepONet control exists to reveal.</p>
  </section>

  <section>
    <h2>Allen–Cahn head-to-head</h2>
    <p>The two problems tell opposite stories, and that is the paper's defence. Allen–Cahn at
    <span class="mono">ε=32</span> has sharp interfaces, so <span class="mono">sin·sin</span>
    factors out nothing — the mollifier stops helping, PINO drops to last place among the FNO
    arms, and the minimizing-movement objective essentially matches supervised training without
    ever seeing a label.</p>
    {result_table(summarize(ac, "test_st_rel_l2"), "space-time rel. L² (%)",
                  "Allen–Cahn runs last — it is the longest stage, so stopping early still "
                  "leaves the Poisson story complete.")}
    <p class="note">PI-DeepONet needed its collocation set subsampled to 1024 interior nodes:
    taking the Laplacian over all 4096 retains a double-backward graph per rollout step and
    exhausted the 24 GB card on all three seeds. It also ran at an untuned lr 1e-3, so read its
    number as an upper bound on the error rather than a converged result.</p>
  </section>

  <section>
    <h2>Learning rates chosen per arm</h2>
    {lr_table(lr_runs)}
    <p class="note">Best validation error at a fifth of the full budget. Existing arms keep
    their published 1e-3.</p>
  </section>

  <section>
    <h2>Queue</h2>
    <div class="stages">{"".join(stage_html)}</div>
  </section>

  <section>
    <h2>What one run costs</h2>
    <div class="scroll"><table>
      <thead><tr><th>arm</th><th class="num">s / epoch</th>
        <th class="num">min / 500-epoch run</th></tr></thead>
      <tbody>{timing_rows}</tbody></table></div>
    <p class="note">PI-DeepONet is the only expensive arm — three backward passes through the
    trunk at every collocation point. Allen–Cahn additionally pays a FEM Newton reference
    solve, which is why that stage builds its dataset once per seed rather than once per run.</p>
  </section>

  <footer>generated {time.strftime('%Y-%m-%d %H:%M')} · branch baselines-pino-deeponet</footer>
</div>
"""
    with open(path, "w") as fh:
        fh.write(html)
    return path, n_ok, len(runs), n_fail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=os.path.join(REPO, "output", "baselines",
                                                        "report.html"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    path, ok, tot, fail = build(a.out)
    print(f"{path}  ({ok}/{tot} runs, {fail} failed)")


if __name__ == "__main__":
    main()
