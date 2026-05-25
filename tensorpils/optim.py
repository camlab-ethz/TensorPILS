"""Optimizer factory. Add a branch to register a custom optimizer (e.g. Shampoo)."""

import torch

__all__ = ["build_optimizer"]


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    **kwargs) -> torch.optim.Optimizer:
    """Centralized optimizer construction."""
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=kwargs.get("momentum", 0.9),
                               weight_decay=weight_decay)
    raise ValueError(
        f"Unknown optimizer {name!r}. Add a branch in build_optimizer() "
        "to register a custom optimizer (e.g. Shampoo)."
    )
