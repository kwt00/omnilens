"""Manages forward and backward hooks on all submodules of a model.

Hooks capture activations, gradients, and routing data, feeding them
into metric collectors via thread-safe queues.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from omnilens.viz.live.config import settings


@dataclass
class HookData:
    """Raw data captured by a single forward/backward hook invocation."""
    input_tensors: list[torch.Tensor] = field(default_factory=list)
    output_tensors: list[torch.Tensor] = field(default_factory=list)
    grad_input: list[torch.Tensor | None] = field(default_factory=list)
    grad_output: list[torch.Tensor | None] = field(default_factory=list)


def _to_tensor_list(x) -> list[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return [x.detach()]
    if isinstance(x, (tuple, list)):
        return [t.detach() for t in x if isinstance(t, torch.Tensor)]
    return []


class HookManager:
    def __init__(self, model: nn.Module):
        self._model = model
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._lock = threading.Lock()

        # Ring buffers: module_name -> deque of HookData
        self._forward_buffer: dict[str, deque[HookData]] = defaultdict(
            lambda: deque(maxlen=settings.metric_buffer_size)
        )
        self._backward_buffer: dict[str, deque[HookData]] = defaultdict(
            lambda: deque(maxlen=settings.metric_buffer_size)
        )

        # Execution order tracking for flow animation
        self._execution_order: list[tuple[str, int]] = []  # (name, timestamp_ns)
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    def install(self) -> None:
        """Register forward and backward hooks on every leaf module."""
        self.uninstall()

        for name, module in self._model.named_modules():
            if not name:
                continue

            # Forward hook: captures activations and registers tensor-level
            # grad hooks on the ORIGINAL (non-detached) output for backward
            # capture. This avoids the view+inplace conflict that
            # register_full_backward_hook causes with in-place ops like ReLU.
            def make_fwd_hook(mod_name: str):
                def hook(mod, inp, out):
                    import time
                    data = HookData(
                        input_tensors=_to_tensor_list(inp),
                        output_tensors=_to_tensor_list(out),
                    )
                    with self._lock:
                        self._forward_buffer[mod_name].append(data)
                        self._execution_order.append((mod_name, time.time_ns()))

                    # Register grad hook on the ORIGINAL output tensors
                    # (before detach) so they remain in the autograd graph.
                    raw_outs = [out] if isinstance(out, torch.Tensor) else []
                    if isinstance(out, (tuple, list)):
                        raw_outs = [t for t in out if isinstance(t, torch.Tensor)]
                    for t in raw_outs:
                        if t.requires_grad:
                            def make_grad_hook(name: str):
                                def grad_hook(grad):
                                    bwd_data = HookData(
                                        grad_output=[grad.detach()] if grad is not None else [],
                                    )
                                    with self._lock:
                                        self._backward_buffer[name].append(bwd_data)
                                return grad_hook
                            t.register_hook(make_grad_hook(mod_name))

                return hook

            h = module.register_forward_hook(make_fwd_hook(name))
            self._handles.append(h)

    def uninstall(self) -> None:
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def advance_step(self) -> None:
        """Called after each forward+backward pass to mark a step boundary."""
        with self._lock:
            self._step += 1

    def get_forward_data(self, module_name: str) -> list[HookData]:
        """Get buffered forward hook data for a module."""
        with self._lock:
            return list(self._forward_buffer.get(module_name, []))

    def get_backward_data(self, module_name: str) -> list[HookData]:
        """Get buffered backward hook data for a module."""
        with self._lock:
            return list(self._backward_buffer.get(module_name, []))

    def get_latest_forward(self, module_name: str) -> HookData | None:
        """Get the most recent forward hook data for a module."""
        with self._lock:
            buf = self._forward_buffer.get(module_name)
            return buf[-1] if buf else None

    def get_latest_backward(self, module_name: str) -> HookData | None:
        """Get the most recent backward hook data for a module."""
        with self._lock:
            buf = self._backward_buffer.get(module_name)
            return buf[-1] if buf else None

    def get_execution_order(self) -> list[tuple[str, int]]:
        """Get the execution order from the most recent forward pass."""
        with self._lock:
            return list(self._execution_order)

    def clear_execution_order(self) -> None:
        """Clear execution order tracking (call before each forward pass)."""
        with self._lock:
            self._execution_order.clear()

    def get_all_module_names(self) -> list[str]:
        """Get all module names that have been hooked."""
        with self._lock:
            return list(set(self._forward_buffer.keys()) | set(self._backward_buffer.keys()))
