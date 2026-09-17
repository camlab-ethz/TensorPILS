r"""GAOT — Geometry-Aware Operator Transformer — and its TensorPILS grid adapter.

Two classes live here, and the split is the point:

:class:`GAOT`
    The architecture, transcribed from ``GAOT/src/model/gaot.py``: a MAGNO encoder lifts values
    on an arbitrary point cloud onto a *structured* latent token grid, a vision transformer
    processes that grid, and a MAGNO decoder reads it back out at arbitrary query points. Its
    interface is the point-cloud one — ``(xcoord [N, 2], pndata [B, N, C]) -> [B, Q, C_out]`` —
    which is what makes it usable on an unstructured mesh, where an FNO is not.

:class:`GAOTModel`
    The TensorPILS wrapper. It presents the **same grid signature as**
    :class:`~tensorpils.models.FNOModel` — ``[B, C, H, W] -> [B, C_out, H, W]`` — so every FEM
    loss in :mod:`~tensorpils.losses` and the unmodified
    :class:`~tensorpils.trainer.PoissonTrainer` run on it with no changes at all. Inside, the
    grid is flattened to the repo's node ordering and the model never sees an image.

The node ordering is the whole bridge, and it is the repo invariant: node ``k = i*nx + j`` at
grid row ``i`` (y) and column ``j`` (x), matching :func:`~tensorpils.meshing.grid_to_node` and
the mesh :func:`~tensorpils.meshing.structured_quad_mesh` builds. ``GAOTModel`` builds its
coordinate buffer with exactly that formula, so

    ``grid_to_node(model(f_grid)[:, 0], nx, ny)``

is the same vector the network emitted at ``self.coords`` — which is what lets the Galerkin /
Deep Ritz / PLS losses consume it. :func:`tests.test_gaot_model` pins that down against the
mesh's own ``points``.

**Going unstructured** is then :meth:`GAOTModel.forward_nodes`: hand it ``pndata [B, N, C]``
and the coordinates, and it returns ``[B, N, C_out]`` on whatever point set you built the model
with. Nothing above it changes, because the FEM losses were always node-based.

Reference: Wen, Mishra et al., *Geometry Aware Operator Transformer as an Efficient and Accurate
Neural Surrogate for PDEs on Arbitrary Domains*, NeurIPS 2025.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .attn import Transformer, TransformerConfig
from .magno import MAGNOConfig, MAGNOEncoder, MAGNODecoder

__all__ = ["GAOT", "GAOTModel", "GAOTArgs", "GAOTConfig",
           "MAGNOConfig", "TransformerConfig"]


@dataclass
class GAOTArgs:
    """The two sub-module configs, under the attribute name :class:`GAOT` reads them from."""
    magno: MAGNOConfig = field(default_factory=MAGNOConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)


@dataclass
class GAOTConfig:
    """Top-level GAOT config.

    ``latent_tokens_size`` is the *structured* latent grid ``(H, W)`` the transformer sees; it
    is independent of the physical point set, which is the reason the same processor works on a
    mesh the FNO cannot consume. Upstream reads these fields off an OmegaConf object built from
    JSON; here they are plain dataclasses assembled by :class:`GAOTModel`.
    """
    latent_tokens_size: Tuple[int, int] = (64, 64)
    args: GAOTArgs = field(default_factory=GAOTArgs)


class GAOT(nn.Module):
    """MAGNO encoder → vision-transformer processor → MAGNO decoder.

    Transcribed from upstream ``GAOT.__init__/encode/process/decode/forward``. Only the 2-D
    branch is kept (TensorPILS is 2-D throughout) and ``autoregressive_predict`` is dropped:
    TensorPILS rolls trajectories forward in :class:`~tensorpils.trainer.RolloutTrainer`, which
    owns the boundary projection and the residual, so a second rollout implementation inside the
    model would be a silent fork of that logic.

    Parameters
    ----------
    input_size, output_size : int
        Channels in and out at each physical node.
    config : GAOTConfig
    """

    def __init__(self, input_size: int, output_size: int, config: GAOTConfig):
        super().__init__()
        coord_dim = config.args.magno.coord_dim
        if coord_dim != 2:
            raise ValueError(f"TensorPILS is 2-D; coord_dim must be 2, got {coord_dim}")

        self.input_size = input_size
        self.output_size = output_size
        self.coord_dim = coord_dim
        self.node_latent_size = config.args.magno.lifting_channels
        self.patch_size = config.args.transformer.patch_size

        latent_tokens_size = config.latent_tokens_size
        if len(latent_tokens_size) != 2:
            raise ValueError(f"latent_tokens_size must be (H, W), got {latent_tokens_size}")
        self.H, self.W = latent_tokens_size

        self.encoder = MAGNOEncoder(in_channels=input_size,
                                    out_channels=self.node_latent_size,
                                    config=config.args.magno)
        self.processor = self.init_processor(self.node_latent_size, config.args.transformer)
        self.decoder = MAGNODecoder(in_channels=self.node_latent_size,
                                    out_channels=output_size,
                                    config=config.args.magno)

    def init_processor(self, node_latent_size, config):
        patch_volume = self.patch_size * self.patch_size
        self.patch_linear = nn.Linear(patch_volume * node_latent_size,
                                      patch_volume * node_latent_size)
        self.positional_embedding_name = config.positional_embedding
        self.positions = self._get_patch_positions()
        return Transformer(input_size=node_latent_size * patch_volume,
                           output_size=node_latent_size * patch_volume,
                           config=config)

    def _get_patch_positions(self) -> torch.Tensor:
        """Integer ``(row, col)`` of every patch, in the order :meth:`process` unfolds them."""
        P = self.patch_size
        return torch.stack(torch.meshgrid(
            torch.arange(self.H // P, dtype=torch.float32),
            torch.arange(self.W // P, dtype=torch.float32),
            indexing="ij",
        ), dim=-1).reshape(-1, 2)

    @staticmethod
    def _compute_absolute_embeddings(positions: torch.Tensor, embed_dim: int) -> torch.Tensor:
        """Sinusoidal embedding of 2-D patch positions, split evenly over the two axes."""
        num_pos_dims = positions.size(1)
        dim_touse = embed_dim // (2 * num_pos_dims)
        freq_seq = torch.arange(dim_touse, dtype=torch.float32, device=positions.device)
        inv_freq = 1.0 / (10000 ** (freq_seq / dim_touse))
        sinusoid_inp = positions[:, :, None] * inv_freq[None, None, :]
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        return pos_emb.view(positions.size(0), -1)

    def encode(self, x_coord, pndata, latent_tokens_coord, encoder_nbrs=None):
        return self.encoder(x_coord=x_coord, pndata=pndata,
                            latent_tokens_coord=latent_tokens_coord,
                            encoder_nbrs=encoder_nbrs)

    def process(self, rndata: torch.Tensor, condition: Optional[float] = None) -> torch.Tensor:
        """``[B, H*W, C]`` latent tokens → patches → transformer → back to ``[B, H*W, C]``."""
        batch_size, n_regional_nodes, C = rndata.shape
        P, H, W = self.patch_size, self.H, self.W

        assert n_regional_nodes == H * W, \
            f"n_regional_nodes ({n_regional_nodes}) != H*W ({H}*{W})"
        assert H % P == 0 and W % P == 0, f"H({H}) and W({W}) must be divisible by P({P})"

        nph, npw = H // P, W // P
        rndata = rndata.view(batch_size, nph, P, npw, P, C)
        rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()
        rndata = rndata.view(batch_size, nph * npw, P * P * C)

        rndata = self.patch_linear(rndata)
        pos = self.positions.to(rndata.device)

        relative_positions = None
        if self.positional_embedding_name == "absolute":
            pos_emb = self._compute_absolute_embeddings(pos, P * P * self.node_latent_size)
            rndata = rndata + pos_emb
        elif self.positional_embedding_name == "rope":
            relative_positions = pos

        rndata = self.processor(rndata, condition=condition,
                                relative_positions=relative_positions)

        rndata = rndata.view(batch_size, nph, npw, P, P, C)
        rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()
        return rndata.view(batch_size, H * W, C)

    def decode(self, latent_tokens_coord, rndata, query_coord, decoder_nbrs=None):
        return self.decoder(latent_tokens_coord=latent_tokens_coord, rndata=rndata,
                            query_coord=query_coord, decoder_nbrs=decoder_nbrs)

    def forward(self, latent_tokens_coord: torch.Tensor, xcoord: torch.Tensor,
                pndata: torch.Tensor, query_coord: Optional[torch.Tensor] = None,
                encoder_nbrs=None, decoder_nbrs=None,
                condition: Optional[float] = None) -> torch.Tensor:
        """``pndata [B, N, C_in]`` at ``xcoord`` → ``[B, Q, C_out]`` at ``query_coord``.

        ``xcoord`` is ``[N, 2]`` when every sample shares the geometry (``fx`` mode) and
        ``[B, N, 2]`` when it varies per sample (``vx`` mode); MAGNO detects which from the rank.
        ``query_coord`` defaults to ``xcoord`` — output where the input lives.
        """
        rndata = self.encode(x_coord=xcoord, pndata=pndata,
                             latent_tokens_coord=latent_tokens_coord,
                             encoder_nbrs=encoder_nbrs)
        rndata = self.process(rndata=rndata, condition=condition)
        if query_coord is None:
            query_coord = xcoord
        return self.decode(latent_tokens_coord=latent_tokens_coord, rndata=rndata,
                           query_coord=query_coord, decoder_nbrs=decoder_nbrs)


def _uniform_grid_coords(nx: int, ny: int, xlims=(0.0, 1.0), ylims=(0.0, 1.0),
                         dtype=torch.float32) -> torch.Tensor:
    """``[ny*nx, 2]`` coordinates in the repo's node order: node ``k = i*nx + j`` at ``(x_j, y_i)``.

    Identical construction to :func:`tensorpils.meshing.structured_quad_mesh`, deliberately —
    the two must agree node-for-node or the FEM operators would be applied to a permuted field.
    """
    xs = torch.linspace(xlims[0], xlims[1], nx, dtype=dtype)
    ys = torch.linspace(ylims[0], ylims[1], ny, dtype=dtype)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")          # both [ny, nx]
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)


class GAOTModel(nn.Module):
    r"""GAOT behind the ``FNOModel`` grid signature — and the door to unstructured meshes.

    ``forward`` takes ``[B, C_in, H, W]`` and returns ``[B, C_out, H, W]``, so it substitutes
    for :class:`~tensorpils.models.FNOModel` everywhere: supervised ``data``, ``galerkin``,
    ``deepritz`` and ``pls`` all run unchanged, and the comparison against the FNO differs only
    in the architecture.

    Parameters
    ----------
    grid_size : (nx, ny)
        The physical grid. ``H = ny`` rows, ``W = nx`` columns, as everywhere in this repo.
    in_channels, out_channels : int
        Channels of the input field and of the emitted field.
    latent_grid : (H_l, W_l)
        Structured latent token grid. Must be divisible by ``patch_size``. This, not the
        physical grid, sets the transformer's cost — which is why the model is insensitive to
        how the physical points are arranged.
    patch_size : int
        Patch side on the latent grid. ``(H_l/P) * (W_l/P)`` transformer tokens, each of width
        ``P^2 * lifting_channels``.
    radius : float, optional
        Ball radius for both MAGNO searches, in domain units. ``None`` (default) derives it as
        ``radius_scale * max(h_phys, h_latent)``, so a resolution change keeps a comparable
        number of neighbours instead of silently emptying (too small) or exploding (too large)
        the graph. At the defaults (``64^2`` physical, ``64^2`` latent) that reproduces
        GAOT's published Poisson setting, ``0.033``.
    radius_scale : float
        The multiplier above. ``2.1`` gives ~13 neighbours per query on a uniform grid.
    scales : sequence of float
        Multiscale radii, as multiples of ``radius``; the encoder/decoder average over them.
    lifting_channels : int
        Latent width per token — the channel count the transformer sees.
    magno_hidden, mlp_layers : int
        Width and depth of the AGNO kernel MLPs.
    transformer_hidden, transformer_layers, num_heads : int
        The processor.
    use_geoembed : bool
        Geometric embedding of each neighbourhood (GAOT's "geometry aware" term). It is what
        carries point-density information, so it earns its keep on an irregular mesh and is
        close to inert on a uniform one.
    attention_type : {'cosine', 'dot_product'}
        Neighbour weighting inside AGNO.
    node_embedding : bool
        Sinusoidal encoding of coordinates before the kernel MLP.
    positional_embedding : {'absolute', 'rope'}
        Transformer positional embedding. ``rope`` needs ``rotary-embedding-torch``.
    coords : torch.Tensor, optional
        ``[N, 2]`` physical node coordinates. Defaults to the uniform ``grid_size`` grid in the
        repo's node order. **Pass a mesh's own ``points`` here to go unstructured** — then use
        :meth:`forward_nodes`, since there is no grid to reshape to.
    domain : ((x0, x1), (y0, y1)), optional
        Extent of the latent grid and of the default radius. ``None`` (default) takes the
        bounding box of ``coords``, or the unit square when the nodes are the uniform grid.
    """

    def __init__(self, grid_size: Tuple[int, int] = (64, 64),
                 in_channels: int = 1, out_channels: int = 1,
                 latent_grid: Tuple[int, int] = (64, 64), patch_size: int = 2,
                 radius: Optional[float] = None, radius_scale: float = 2.1,
                 scales: Sequence[float] = (1.0,),
                 lifting_channels: int = 64, magno_hidden: int = 64, mlp_layers: int = 3,
                 transformer_hidden: int = 256, transformer_layers: int = 3,
                 num_heads: int = 8, use_geoembed: bool = True,
                 attention_type: str = "cosine", node_embedding: bool = False,
                 positional_embedding: str = "absolute",
                 coords: Optional[torch.Tensor] = None,
                 domain: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None):
        super().__init__()
        nx, ny = grid_size
        if domain is None:
            # Derived from the points, so the latent grid covers the geometry it is meant to
            # encode. A hard-coded unit square would put most of the latent grid outside a
            # domain that does not happen to live there -- silently, since an empty
            # neighbourhood is a zero encoding, not an error.
            if coords is None:
                domain = ((0.0, 1.0), (0.0, 1.0))
            else:
                c = torch.as_tensor(coords, dtype=torch.float32)
                domain = ((float(c[:, 0].min()), float(c[:, 0].max())),
                          (float(c[:, 1].min()), float(c[:, 1].max())))
        (x0, x1), (y0, y1) = domain
        self.grid_size = (nx, ny)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.domain = domain

        uniform = _uniform_grid_coords(nx, ny, (x0, x1), (y0, y1))
        phys = uniform if coords is None else torch.as_tensor(coords, dtype=torch.float32)
        if phys.ndim != 2 or phys.shape[1] != 2:
            raise ValueError(f"coords must be [N, 2]; got {tuple(phys.shape)}")
        # The grid ``forward`` is only meaningful when the nodes *are* the grid, in the repo's
        # order. Checking the values (not just the count) means passing a structured mesh's own
        # ``points`` keeps the grid path, while any genuinely unstructured set loses it and has
        # to go through ``forward_nodes`` -- which is the honest failure, not a silent reshape.
        self.structured = (phys.shape == uniform.shape
                           and torch.allclose(phys, uniform, atol=1e-6))

        Hl, Wl = latent_grid
        latent = _uniform_grid_coords(Wl, Hl, (x0, x1), (y0, y1))   # token k = i*Wl + j

        # Buffers, not attributes: they must follow ``.to(device)`` and travel in the
        # checkpoint, so a reloaded model is scored on the geometry it was trained on.
        self.register_buffer("coords", phys)
        self.register_buffer("latent_tokens_coord", latent)

        if radius is None:
            area = (x1 - x0) * (y1 - y0)
            h_phys = math.sqrt(area / phys.shape[0])      # mean node spacing, mesh-agnostic
            h_latent = max((x1 - x0) / max(Wl - 1, 1), (y1 - y0) / max(Hl - 1, 1))
            radius = radius_scale * max(h_phys, h_latent)
        self.radius = float(radius)

        config = GAOTConfig(
            latent_tokens_size=(Hl, Wl),
            args=GAOTArgs(
                magno=MAGNOConfig(
                    coord_dim=2, radius=self.radius, hidden_size=magno_hidden,
                    mlp_layers=mlp_layers, lifting_channels=lifting_channels,
                    scales=list(scales), use_geoembed=use_geoembed,
                    attention_type=attention_type, node_embedding=node_embedding,
                ),
                transformer=TransformerConfig(
                    patch_size=patch_size, hidden_size=transformer_hidden,
                    num_layers=transformer_layers,
                    positional_embedding=positional_embedding,
                ),
            ),
        )
        config.args.transformer.attn_config.num_heads = num_heads
        config.args.transformer.attn_config.num_kv_heads = num_heads

        self.gaot = GAOT(input_size=in_channels, output_size=out_channels, config=config)
        self.config = config

    # -------------------- the two entry points --------------------

    def forward_nodes(self, pndata: torch.Tensor,
                      coords: Optional[torch.Tensor] = None,
                      query_coord: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``[B, N, C_in]`` node values → ``[B, Q, C_out]``. The unstructured-mesh entry point.

        ``coords`` defaults to the buffer set at construction. ``query_coord`` defaults to
        ``coords``, i.e. output at the input nodes — which is what a FEM residual needs, since
        it is assembled on exactly those nodes.
        """
        xcoord = self.coords if coords is None else coords
        return self.gaot(latent_tokens_coord=self.latent_tokens_coord,
                         xcoord=xcoord, pndata=pndata, query_coord=query_coord)

    def forward(self, f_grid: torch.Tensor) -> torch.Tensor:
        """``[B, C_in, H, W]`` → ``[B, C_out, H, W]`` — the ``FNOModel``-compatible signature.

        The flatten is the repo's node order (row-major over ``(i, j)``, node ``k = i*nx + j``),
        matching ``self.coords``; the un-flatten is its exact inverse. So a caller may equally
        well read the result through :func:`~tensorpils.meshing.grid_to_node` and hand it to a
        FEM loss.
        """
        if not self.structured:
            raise RuntimeError(
                "this GAOTModel was built on an explicit (unstructured) point set, which has no "
                "grid to reshape to; call forward_nodes() instead.")
        squeeze = (f_grid.dim() == 3)
        if squeeze:
            f_grid = f_grid.unsqueeze(0)
        B, C, H, W = f_grid.shape
        if (W, H) != self.grid_size:
            raise ValueError(f"expected grid (H, W) = ({self.grid_size[1]}, {self.grid_size[0]}), "
                             f"got ({H}, {W})")

        pndata = f_grid.reshape(B, C, H * W).transpose(1, 2)            # [B, N, C_in]
        u = self.forward_nodes(pndata)                                   # [B, N, C_out]
        u = u.transpose(1, 2).reshape(B, self.out_channels, H, W)
        return u.squeeze(0) if squeeze else u

    def clear_neighbor_cache(self):
        """Drop the cached radius-search results.

        MAGNO keys its cache on tensor *shapes*, not values, so swapping in a different point
        set of the same size would silently reuse the old graph. Fixed-geometry training never
        hits that; a variable-mesh dataset must call this (or set ``precompute_edges``).
        """
        self.gaot.encoder.neighbor_cache.clear()
        self.gaot.decoder.neighbor_cache.clear()

    def extra_repr(self) -> str:
        Hl, Wl = self.config.latent_tokens_size
        P = self.config.args.transformer.patch_size
        return (f"grid={self.grid_size}, nodes={self.coords.shape[0]}, latent={(Hl, Wl)}, "
                f"patch={P} -> {(Hl // P) * (Wl // P)} tokens, radius={self.radius:.4f}")
