"""Tracks tensor propagation through the model for flow animation.

Records per-layer tensor summaries with timestamps during forward passes.
"""

from __future__ import annotations

import torch

from omnilens.viz.live.engine.hook_manager import HookManager
from omnilens.viz.live.models.metric_schema import FlowFrame, FlowSequence


class TensorTracker:
    def __init__(self, hook_manager: HookManager):
        self._hooks = hook_manager

    def capture_flow(self) -> FlowSequence:
        """Build a FlowSequence from the most recent forward pass execution order."""
        execution_order = self._hooks.get_execution_order()
        frames: list[FlowFrame] = []

        for module_name, timestamp_ns in execution_order:
            fwd = self._hooks.get_latest_forward(module_name)
            if fwd is None or not fwd.output_tensors:
                continue

            t = fwd.output_tensors[0].float()
            flat = t.flatten()

            frames.append(FlowFrame(
                layer_id=module_name,
                timestamp_ns=timestamp_ns,
                shape=list(t.shape),
                mean=float(flat.mean()),
                std=float(flat.std()) if flat.numel() > 1 else 0.0,
                min_val=float(flat.min()),
                max_val=float(flat.max()),
            ))

        return FlowSequence(
            step=self._hooks.step,
            frames=frames,
        )
