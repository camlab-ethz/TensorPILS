"""GAOT — Geometry-Aware Operator Transformer, vendored for TensorPILS.

The second architecture in this repo, and the one that does not need a grid. An FNO is tied to
the FFT and therefore to a structured mesh; GAOT encodes an arbitrary point cloud onto a fixed
structured *latent* token grid, runs a transformer there, and reads back out at arbitrary query
points. That is what makes the unstructured-mesh test cases reachable without touching
:mod:`tensorpils.losses` or :mod:`tensorpils.physics`, which were node-based all along.

Source: ``GAOT/src/model/layers/{magno,agno,attn,gemb,mlp}.py`` and ``src/model/gaot.py``,
transcribed rather than imported so this repo has one environment and one dependency set.
The deliberate departures from upstream, each commented where it happens:

* ``ops.py`` makes ``torch_scatter`` / ``torch_cluster`` optional fast paths with vectorised
  pure-torch fallbacks (upstream imports them unconditionally, and its own fallbacks are
  Python loops that also lack the segment ``max`` the cosine attention needs).
* ``omegaconf`` is gone; the configs are plain dataclasses built in :mod:`tensorpils.cli`.
* ``rotary-embedding-torch`` is imported lazily, only for ``positional_embedding='rope'``.
* The sequential ``autoregressive_predict`` is dropped -- TensorPILS rolls trajectories in
  :class:`~tensorpils.trainer.RolloutTrainer`, which owns the boundary projection.

Everything else -- MAGNO's multiscale attentional kernel integral, the geometric embedding, the
UViT processor with long-range skips -- is upstream's.

Entry point: :class:`~tensorpils.gaot.model.GAOTModel`, which wears the
:class:`~tensorpils.models.FNOModel` grid signature so the trainers need no branch for it.
"""

from .model import GAOT, GAOTArgs, GAOTConfig, GAOTModel
from .magno import MAGNOConfig, MAGNOEncoder, MAGNODecoder
from .attn import AttentionConfig, Transformer, TransformerConfig

__all__ = ["GAOTModel", "GAOT", "GAOTConfig", "GAOTArgs",
           "MAGNOConfig", "MAGNOEncoder", "MAGNODecoder",
           "TransformerConfig", "AttentionConfig", "Transformer"]
