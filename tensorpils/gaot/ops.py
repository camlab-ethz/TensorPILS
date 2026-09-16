"""Scatter reductions, radius search and edge sampling for the GAOT layers.

Upstream GAOT keeps these in ``src/model/layers/utils/{segment_csr,neighbor_search,edge_drop}.py``
and imports ``torch_scatter`` / ``torch_cluster`` unconditionally (``gemb.py`` does
``from torch_scatter import ...`` at module scope). Both are compiled extensions pinned to an
exact torch build, so a hard dependency would tie this repo's environment to one torch version.

Here they are **optional fast paths**. Every op has a vectorised pure-torch fallback that is
selected automatically, so ``tensorpils.gaot`` imports and runs in a plain-torch environment.
Two fixes to the upstream fallbacks were needed to make that true rather than nominal:

* ``segment_csr`` supports ``reduce="max"``. The upstream fallback raises on it, and the cosine
  attention in :mod:`~tensorpils.gaot.agno` needs a segment max for its softmax — so without
  torch_scatter the default model configuration simply did not run.
* The fallbacks are vectorised. The upstream ones loop in Python over query nodes
  (``for i in range(n_out)``, ``for q_idx, query in enumerate(queries)``), which is a non-starter
  at the ``64^2`` latent grid used here: 4096 Python iterations per call, per layer, per step.

The CSR convention is upstream's and is shared with ``open3d``/``neuraloperator``: a neighbour
list is ``{"neighbors_index": [E], "neighbors_row_splits": [m+1]}``, where the neighbours of
query ``j`` are ``neighbors_index[row_splits[j]:row_splits[j+1]]``.
"""

import warnings
from typing import Dict, Optional

import torch
from torch import nn

try:  # compiled fast path; see module docstring
    import torch_scatter as _torch_scatter
    HAS_TORCH_SCATTER = True
except ImportError:  # pragma: no cover - exercised only in a plain-torch env
    _torch_scatter = None
    HAS_TORCH_SCATTER = False

try:
    import torch_cluster as _torch_cluster
    HAS_TORCH_CLUSTER = True
except ImportError:  # pragma: no cover
    _torch_cluster = None
    HAS_TORCH_CLUSTER = False

__all__ = ["segment_csr", "scatter_sum", "scatter_mean", "scatter_max",
           "NeighborSearch", "apply_edge_drop_csr",
           "HAS_TORCH_SCATTER", "HAS_TORCH_CLUSTER"]


# ``torch_cluster.radius`` silently truncates at ``max_num_neighbors=32``. Upstream GAOT calls it
# without the argument, so at a resolution where a ball holds more than 32 points the graph is
# quietly thinned and the torch_cluster and native paths stop agreeing. We pass the cap
# explicitly, generously, and warn when it actually binds.
_MAX_NEIGHBORS_CAP = 256


def _shape_for(dim: int, ndim: int):
    """Broadcast shape that puts a per-segment vector on axis ``dim`` of an ``ndim`` tensor."""
    return [-1 if i == dim else 1 for i in range(ndim)]


def _segment_native(src: torch.Tensor, indptr: torch.Tensor, reduce: str, dim: int):
    """Vectorised ``segment_csr``: expand the row splits to a segment id, then scatter."""
    n_out = indptr.numel() - 1
    counts = indptr[1:] - indptr[:-1]
    index = torch.repeat_interleave(
        torch.arange(n_out, device=src.device), counts.to(src.device)
    )
    shape = list(src.shape)
    shape[dim] = n_out

    if reduce in ("sum", "mean"):
        out = src.new_zeros(shape).index_add_(dim, index, src)
        if reduce == "mean":
            denom = counts.clamp(min=1).to(src.dtype).reshape(_shape_for(dim, out.ndim))
            out = out / denom
        return out

    if reduce == "max":
        idx = index.reshape(_shape_for(dim, src.ndim)).expand_as(src)
        out = src.new_full(shape, float("-inf"))
        out = out.scatter_reduce(dim, idx, src, reduce="amax", include_self=True)
        # torch_scatter leaves empty segments at 0; match that rather than propagating -inf.
        return torch.where(torch.isneginf(out), torch.zeros_like(out), out)

    raise ValueError(f"reduce must be one of 'sum', 'mean', 'max'; got {reduce!r}")


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum",
                use_scatter: bool = True) -> torch.Tensor:
    """Reduce ``src`` over the segments a CSR ``indptr`` marks out.

    ``src`` is ``[E, ...]`` with a 1-D ``indptr`` of length ``m+1``, or ``[B, E, ...]`` with
    ``indptr`` repeated to ``[B, m+1]`` (the batched form the AGNO aggregation uses). The
    reduction axis is ``indptr.ndim - 1`` in both cases.
    """
    if use_scatter and HAS_TORCH_SCATTER:
        return _torch_scatter.segment_csr(src, indptr, reduce=reduce)
    dim = indptr.ndim - 1
    indptr_1d = indptr[0] if indptr.ndim == 2 else indptr
    return _segment_native(src, indptr_1d, reduce, dim)


def _scatter_native(src: torch.Tensor, index: torch.Tensor, dim_size: int, reduce: str):
    shape = list(src.shape)
    shape[0] = dim_size
    if reduce in ("sum", "mean"):
        out = src.new_zeros(shape).index_add_(0, index, src)
        if reduce == "mean":
            counts = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
            counts.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
            out = out / counts.clamp(min=1).reshape(_shape_for(0, out.ndim))
        return out
    idx = index.reshape(_shape_for(0, src.ndim)).expand_as(src)
    out = src.new_full(shape, float("-inf")).scatter_reduce(0, idx, src, reduce="amax",
                                                            include_self=True)
    return torch.where(torch.isneginf(out), torch.zeros_like(out), out)


def scatter_sum(src, index, dim: int = 0, dim_size: Optional[int] = None):
    """``torch_scatter.scatter_sum`` over ``dim=0`` (the only axis GAOT reduces along)."""
    if HAS_TORCH_SCATTER:
        return _torch_scatter.scatter_sum(src, index, dim=dim, dim_size=dim_size)
    assert dim == 0, "native fallback only implements dim=0"
    return _scatter_native(src, index, dim_size or int(index.max()) + 1, "sum")


def scatter_mean(src, index, dim: int = 0, dim_size: Optional[int] = None):
    if HAS_TORCH_SCATTER:
        return _torch_scatter.scatter_mean(src, index, dim=dim, dim_size=dim_size)
    assert dim == 0, "native fallback only implements dim=0"
    return _scatter_native(src, index, dim_size or int(index.max()) + 1, "mean")


def scatter_max(src, index, dim: int = 0, dim_size: Optional[int] = None):
    """Returns ``(values, argmax)`` like ``torch_scatter``; the fallback returns ``None`` for
    the argmax, which GAOT discards (``pooled_features, _ = scatter_max(...)``)."""
    if HAS_TORCH_SCATTER:
        return _torch_scatter.scatter_max(src, index, dim=dim, dim_size=dim_size)
    assert dim == 0, "native fallback only implements dim=0"
    return _scatter_native(src, index, dim_size or int(index.max()) + 1, "max"), None


class NeighborSearch(nn.Module):
    """Radius (or k-nearest) search between two point sets, returned in CSR form.

    For every point ``x`` in ``queries``, find all ``y`` in ``data`` with ``|x-y| <= radius``.
    ``method="auto"`` takes ``torch_cluster`` when it is importable and the chunked native
    search otherwise; the two agree up to a permutation of each neighbourhood, which every
    consumer here is invariant to (they all reduce over the neighbourhood).

    ``chunk_size`` bounds the native path's memory: it materialises a ``[chunk, n_data]``
    distance block at a time instead of the full ``[n_query, n_data]`` matrix.
    """

    def __init__(self, method: str = "auto", chunk_size: int = 4096,
                 max_neighbors_cap: int = _MAX_NEIGHBORS_CAP):
        super().__init__()
        self.chunk_size = chunk_size
        self.max_neighbors_cap = max_neighbors_cap

        if method == "auto":
            method = "torch_cluster" if HAS_TORCH_CLUSTER else "native"
        if method == "torch_cluster" and not HAS_TORCH_CLUSTER:
            warnings.warn("torch_cluster not installed; falling back to the native search.")
            method = "native"
        if method not in ("torch_cluster", "native"):
            raise ValueError(f"method must be 'auto', 'torch_cluster' or 'native'; got {method!r}")
        self.method = method

    def forward(self, data: torch.Tensor, queries: torch.Tensor, radius: float) -> Dict[str, torch.Tensor]:
        if self.method == "torch_cluster":
            return self._torch_cluster_search(data, queries, radius)
        return self._native_search(data, queries, radius)

    def _torch_cluster_search(self, data, queries, radius):
        row, col = _torch_cluster.radius(data, queries, r=float(radius),
                                         max_num_neighbors=self.max_neighbors_cap)
        n_queries = queries.shape[0]
        counts = torch.bincount(row, minlength=n_queries)
        if int(counts.max()) >= self.max_neighbors_cap:
            warnings.warn(
                f"radius search hit max_num_neighbors={self.max_neighbors_cap}; the graph is "
                f"truncated. Raise NeighborSearch(max_neighbors_cap=...) or lower the radius.")
        splits = torch.zeros(n_queries + 1, dtype=torch.long, device=data.device)
        torch.cumsum(counts, dim=0, out=splits[1:])
        return {"neighbors_index": col.long(), "neighbors_row_splits": splits}

    @torch.no_grad()
    def _native_search(self, data, queries, radius):
        n_queries = queries.shape[0]
        idx_chunks, counts = [], []
        for start in range(0, n_queries, self.chunk_size):
            q = queries[start:start + self.chunk_size]
            hit = torch.cdist(q, data) <= radius                      # [chunk, n_data]
            # nonzero() is row-major, so the indices already come out grouped by query.
            idx_chunks.append(hit.nonzero()[:, 1])
            counts.append(hit.sum(dim=1))
        splits = torch.zeros(n_queries + 1, dtype=torch.long, device=data.device)
        torch.cumsum(torch.cat(counts).long(), dim=0, out=splits[1:])
        return {"neighbors_index": torch.cat(idx_chunks).long(),
                "neighbors_row_splits": splits}


def apply_edge_drop_csr(
    neighbors: Dict[str, torch.Tensor],
    sampling_strategy: Optional[str],
    max_neighbors: Optional[int] = None,
    sample_ratio: Optional[float] = None,
    training: bool = True,
) -> Dict[str, torch.Tensor]:
    """Neighbour sampling as training-time regularisation (a no-op at eval).

    ``'ratio'`` keeps each edge independently with probability ``sample_ratio``;
    ``'max_neighbors'`` caps the degree of every query node. Verbatim from upstream GAOT
    except that ``'max_neighbors'`` is vectorised instead of looping over the over-full nodes.
    """
    if not training or sampling_strategy is None:
        return neighbors

    neighbors_index = neighbors["neighbors_index"]
    row_splits = neighbors["neighbors_row_splits"]
    device = neighbors_index.device
    if neighbors_index.numel() == 0:
        return neighbors

    n_nodes = row_splits.shape[0] - 1
    n_edges = neighbors_index.shape[0]
    counts = row_splits[1:] - row_splits[:-1]
    if (counts < 0).any() or int(counts.sum()) != n_edges:
        raise ValueError("invalid CSR structure passed to edge drop")

    owner = torch.arange(n_nodes, device=device).repeat_interleave(counts)

    if sampling_strategy == "ratio":
        if sample_ratio is None or sample_ratio >= 1.0:
            return neighbors
        keep = torch.rand(n_edges, device=device) < sample_ratio
        sel = keep.nonzero(as_tuple=True)[0]                  # already grouped by node
        new_counts = torch.bincount(owner[sel], minlength=n_nodes)

    elif sampling_strategy == "max_neighbors":
        if max_neighbors is None or not bool((counts > max_neighbors).any()):
            return neighbors
        # Shuffle globally, then re-group by node with a *stable* sort: each node's edges come
        # back in an independent random order, and keeping the first k of every group is a
        # uniform sample without replacement. The grouping is unchanged, so the kept edges are
        # still in CSR order and the new row splits are just the clamped counts.
        shuffled = torch.argsort(torch.rand(n_edges, device=device))
        regrouped = shuffled[torch.argsort(owner[shuffled], stable=True)]
        rank_in_node = (torch.arange(n_edges, device=device)
                        - row_splits[:-1].repeat_interleave(counts))
        sel = regrouped[rank_in_node < max_neighbors]
        new_counts = counts.clamp(max=max_neighbors)

    else:
        return neighbors

    new_splits = torch.zeros(n_nodes + 1, dtype=row_splits.dtype, device=device)
    torch.cumsum(new_counts, dim=0, out=new_splits[1:])
    return {"neighbors_index": neighbors_index[sel], "neighbors_row_splits": new_splits}
