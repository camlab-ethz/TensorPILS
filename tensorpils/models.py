"""Neural-operator models. Currently a thin wrapper around ``neuralop``'s FNO."""

import torch
import torch.nn as nn

try:
    from neuralop.models import FNO
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "neuralop is required for the FNO model. Install with `pip install neuraloperator`."
    ) from e

__all__ = ["FNOModel"]


class FNOModel(nn.Module):
    """Fourier Neural Operator mapping a source field to a solution field.

    Accepts input of shape ``[B, C, H, W]`` (or ``[C, H, W]`` for a single sample)
    and returns the same spatial shape with ``out_channels`` channels.
    """

    def __init__(self, n_modes=(16, 16), hidden_channels=64,
                 in_channels=1, out_channels=1, n_layers=4, **kwargs):
        super().__init__()
        self.fno = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            **kwargs,
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        squeeze = (f.dim() == 3)
        if squeeze:
            f = f.unsqueeze(0)
        u = self.fno(f)
        return u.squeeze(0) if squeeze else u
