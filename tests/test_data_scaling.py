"""Tests for the data-scaling machinery: the streaming (infinite-data) Poisson dataset,
the fixed-eval splits, the ``steps_per_epoch`` budget, and patience-based early stopping.

No ``neuralop`` needed: the trainer tests use a tiny conv net in place of the FNO (the
trainer only requires a ``[B,1,H,W] -> [B,1,H,W]`` module).
"""

import glob
import json

import torch
import torch.nn as nn

from tensorpils.data import (PoissonDataset, StreamingPoissonDataset,
                             create_scaling_datasets)
from tensorpils.trainer import PoissonTrainer

G = 16  # small grid for speed


def _items_equal(a, b):
    (fg1, gs1, f1, u1), ug1 = a
    (fg2, gs2, f2, u2), ug2 = b
    return (gs1 == gs2 and torch.equal(fg1, fg2) and torch.equal(f1, f2)
            and torch.equal(u1, u2) and torch.equal(ug1, ug2))


# ----------------------------- streaming dataset -----------------------------

def test_stream_item_format_matches_finite():
    fin = PoissonDataset(num_samples=2, K=2, seed=0, grid_resolution=G)
    st = StreamingPoissonDataset(samples_per_epoch=4, K=2, seed=0, grid_resolution=G,
                                 mesh=fin.mesh, problem=fin.problem)
    (fg, gs, f, u), ug = next(iter(st))
    (fg0, gs0, f0, u0), ug0 = fin[0]
    assert fg.shape == fg0.shape and ug.shape == ug0.shape
    assert f.shape == f0.shape and u.shape == u0.shape
    assert gs == gs0
    assert fg.dtype == fg0.dtype and u.dtype == u0.dtype


def test_stream_deterministic_and_nonrepeating():
    kw = dict(samples_per_epoch=6, K=2, seed=7, grid_resolution=G, chunk_size=4)
    a, b = StreamingPoissonDataset(**kw), StreamingPoissonDataset(**kw)
    ep_a1, ep_b1 = list(iter(a)), list(iter(b))
    assert len(ep_a1) == 6
    # Same seed -> identical stream (reproducibility across processes).
    assert all(_items_equal(x, y) for x, y in zip(ep_a1, ep_b1))
    # Generator state persists across passes -> the second epoch continues the stream.
    ep_a2 = list(iter(a))
    assert len(ep_a2) == 6
    assert not any(_items_equal(x, y) for x, y in zip(ep_a1, ep_a2))


def test_stream_preview_getitem_is_stable():
    st = StreamingPoissonDataset(samples_per_epoch=4, K=2, seed=0, grid_resolution=G)
    before = st[0]
    list(iter(st))                       # consuming the stream must not move the preview
    assert _items_equal(before, st[0])


# ----------------------------- fixed-eval splits -----------------------------

def test_scaling_splits_share_eval_sets_across_n_train():
    tr4, val4, te4 = create_scaling_datasets(4, 3, 3, K=2, grid_resolution=G, seed=1)
    tr8, val8, te8 = create_scaling_datasets(8, 3, 3, K=2, grid_resolution=G, seed=1)
    assert len(tr4) == 4 and len(tr8) == 8
    for i in range(3):
        assert _items_equal(val4[i], val8[i])
        assert _items_equal(te4[i], te8[i])
    # Train sets are nested: the n=4 samples are the first 4 of the n=8 set.
    for i in range(4):
        assert _items_equal(tr4[i], tr8[i])


def test_scaling_stream_split():
    tr, val, te = create_scaling_datasets(0, 2, 2, K=2, grid_resolution=G, seed=1,
                                          stream_samples_per_epoch=4)
    assert isinstance(tr, StreamingPoissonDataset)
    assert tr.problem is val.problem is te.problem   # one shared PoissonProblem
    assert len(list(iter(tr))) == 4


# ----------------------------- trainer modality -----------------------------

class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


def _trainer(tmp_path, train_ds, val_ds, test_ds, **kw):
    return PoissonTrainer(model=_TinyNet(), train_dataset=train_ds, val_dataset=val_ds,
                          test_dataset=test_ds, loss_type="data", batch_size=2,
                          device="cpu", output_dir=str(tmp_path), **kw)


def test_steps_per_epoch_fixes_batch_count(tmp_path):
    tr, val, te = create_scaling_datasets(4, 2, 2, K=2, grid_resolution=G, seed=1)
    t = _trainer(tmp_path, tr, val, te, epochs=1, steps_per_epoch=5)
    assert len(list(t.train_loader)) == 5    # 5 batches although n_train is only 4


def test_patience_early_stop(tmp_path):
    tr, val, te = create_scaling_datasets(4, 2, 2, K=2, grid_resolution=G, seed=1)
    t = _trainer(tmp_path, tr, val, te, epochs=50, patience=2)
    vals = iter([1.0] + [2.0] * 49)          # improves once, then never again
    t.validate = lambda: (next(vals), 0.0, 0.0)
    t.train()
    assert t.stats.best_epoch == 0
    assert t.stats.stopped_epoch == 2        # first epoch with epoch - best_epoch >= 2
    assert len(t.stats.train_losses) == 3


def test_no_flags_runs_full_budget(tmp_path):
    tr, val, te = create_scaling_datasets(4, 2, 2, K=2, grid_resolution=G, seed=1)
    t = _trainer(tmp_path, tr, val, te, epochs=3)
    t.train()
    assert t.stats.stopped_epoch == -1
    assert len(t.stats.train_losses) == 3


def test_stream_trainer_smoke_and_results_json(tmp_path):
    tr, val, te = create_scaling_datasets(0, 2, 2, K=2, grid_resolution=G, seed=1,
                                          stream_samples_per_epoch=4)
    t = _trainer(tmp_path, tr, val, te, epochs=2, steps_per_epoch=2)
    t.train()
    paths = glob.glob(f"{tmp_path}/results/*.json")
    assert len(paths) == 1 and "samples-inf-" in paths[0]
    with open(paths[0]) as fh:
        rec = json.load(fh)
    assert rec["stream"] is True
    assert rec["n_train"] is None
    assert rec["steps_per_epoch"] == 2
    assert len(rec["stats"]["val_rel_l2_errors"]) == 2
