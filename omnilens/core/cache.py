from __future__ import annotations

import torch
from typing import Iterator


class ActivationCache:
    """A thin dict-like container for cached activations.

    Keys are standardized omnilens names (e.g. 'layers.0.attention.q.activations').
    Values are tensors, stored on whatever device they came from.
    """

    def __init__(self) -> None:
        self._cache: dict[str, torch.Tensor] = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._cache[key]

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self._cache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __iter__(self) -> Iterator[str]:
        return iter(self._cache)

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        keys = list(self._cache.keys())
        if len(keys) > 10:
            shown = keys[:5] + ["..."] + keys[-5:]
        else:
            shown = keys
        return f"ActivationCache({len(self._cache)} entries: {shown})"

    def keys(self):
        return self._cache.keys()

    def values(self):
        return self._cache.values()

    def items(self):
        return self._cache.items()

    def clear(self) -> None:
        self._cache.clear()

    def to(self, device: str | torch.device) -> ActivationCache:
        """Move all cached tensors to a device."""
        new_cache = ActivationCache()
        for key, tensor in self._cache.items():
            new_cache[key] = tensor.to(device)
        return new_cache

    def detach(self) -> ActivationCache:
        """Detach all cached tensors from the computation graph."""
        new_cache = ActivationCache()
        for key, tensor in self._cache.items():
            new_cache[key] = tensor.detach()
        return new_cache
