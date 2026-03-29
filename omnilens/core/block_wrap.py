"""Provides residual stream hook points using PyTorch's native hooks.

Instead of replacing the block's forward method (fragile across architectures),
we use:
  - register_forward_pre_hook on the block: captures residual.input
  - register_forward_hook on the block: captures residual.block_out
  - register_forward_hook on the attention module: captures residual.attn_out
    by computing (block_input + attention_output)

This is architecture-agnostic — no forward method replacement needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from omnilens.core.attention_wrap import AttentionHookPoints


def register_residual_hooks(
    block_module: nn.Module,
    attn_module: nn.Module,
    layer_idx: int,
    hook_points: AttentionHookPoints,
    omnilens_prefix: str,
    handles: list,
) -> None:
    """Register hooks to capture residual stream values.

    Uses native PyTorch hooks — no forward method patching.

    Args:
        block_module: The transformer block module.
        attn_module: The attention submodule within the block.
        layer_idx: Layer index.
        hook_points: Shared hook points container for interventions.
        omnilens_prefix: e.g. 'layers.0'
        handles: List to append RemovableHook handles to.
    """
    input_name = f"{omnilens_prefix}.residual.input"
    attn_out_name = f"{omnilens_prefix}.residual.attn_out"
    block_out_name = f"{omnilens_prefix}.residual.block_out"

    # Storage for the block input so attn_out hook can compute residual + attn
    block_input_storage = {}

    def block_pre_hook(mod, args):
        hidden_states = args[0] if args else None
        if hidden_states is not None:
            result = hook_points.run(input_name, hidden_states)
            if result is not hidden_states:
                # Intervention happened — replace in args
                return (result,) + args[1:]
            block_input_storage["input"] = hidden_states

    def block_post_hook(mod, input, output):
        out_tensor = output[0] if isinstance(output, tuple) else output
        result = hook_points.run(block_out_name, out_tensor)
        if result is not out_tensor:
            if isinstance(output, tuple):
                return (result,) + output[1:]
            return result

    def attn_post_hook(mod, input, output):
        attn_output = output[0] if isinstance(output, tuple) else output
        block_input = block_input_storage.get("input")
        if block_input is not None:
            residual_after_attn = block_input + attn_output
            result = hook_points.run(attn_out_name, residual_after_attn)
            if result is not residual_after_attn:
                # Intervention: adjust attn output so residual + new_attn = desired
                new_attn = result - block_input
                if isinstance(output, tuple):
                    return (new_attn,) + output[1:]
                return new_attn

    h1 = block_module.register_forward_pre_hook(block_pre_hook)
    h2 = block_module.register_forward_hook(block_post_hook)
    h3 = attn_module.register_forward_hook(attn_post_hook)
    handles.extend([h1, h2, h3])
