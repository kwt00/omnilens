"""Wraps attention modules to expose intermediate computed values as hook points.

Instead of reimplementing the entire model (TransformerLens approach), we only
replace the attention forward method. The Q/K/V/output projections remain
native HuggingFace modules — we just insert hook points between them for:

  - qk_logits:       Q @ K^T / sqrt(d_head), before softmax
  - weights:         post-softmax attention weights
  - weighted_values: weights @ V, before output projection

The attention math (Q@K, softmax, @V) is identical across all transformer
architectures. The only thing that varies is the reshape from (batch, seq, d_model)
to (batch, heads, seq, head_dim), which we read from the model config.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


HookFn = Callable[[torch.Tensor, str], Optional[torch.Tensor]]


class AttentionHookPoints:
    """Stores hook functions for attention intermediate values."""

    def __init__(self) -> None:
        self.hooks: dict[str, list[HookFn]] = {}

    def add(self, name: str, fn: HookFn) -> None:
        self.hooks.setdefault(name, []).append(fn)

    def run(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        for fn in self.hooks.get(name, []):
            result = fn(tensor, name)
            if result is not None:
                tensor = result
        return tensor

    def clear(self) -> None:
        self.hooks.clear()

    def has_hooks(self) -> bool:
        return len(self.hooks) > 0


def wrap_attention_module(
    attn_module: nn.Module,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int | None,
    head_dim: int,
    hook_points: AttentionHookPoints,
    omnilens_prefix: str,
) -> None:
    """Replace an attention module's forward to expose intermediate values.

    This preserves all the module's parameters and submodules — we only
    replace the forward method to add hook points between the Q@K, softmax,
    and @V steps.

    Args:
        attn_module: The HuggingFace attention module to wrap.
        layer_idx: Layer index (for hook naming).
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads (for GQA). None means same as num_heads.
        head_dim: Dimension per head.
        hook_points: Shared hook points container.
        omnilens_prefix: Prefix for hook names (e.g. 'layers.0.attention').
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads

    num_kv_groups = num_heads // num_kv_heads
    original_forward = attn_module.forward

    # Detect which projection style this module uses
    proj_config = _detect_projections(attn_module)
    if proj_config is None:
        return  # Can't wrap — unknown projection layout

    qk_logits_name = f"{omnilens_prefix}.qk_logits"
    weights_name = f"{omnilens_prefix}.weights"
    weighted_values_name = f"{omnilens_prefix}.weighted_values"

    def wrapped_forward(*args, **kwargs):
        # hidden_states is always the first positional arg
        hidden_states = args[0] if args else kwargs.pop("hidden_states")
        # Extract attention mask if provided
        attention_mask = kwargs.get("attention_mask", None)
        position_embeddings = kwargs.get("position_embeddings", None)

        bsz, seq_len, _ = hidden_states.shape

        # Get Q, K, V through the original projections
        q, k, v = _get_qkv(attn_module, hidden_states, proj_config)

        # Reshape to (batch, heads, seq, head_dim)
        q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply rotary embeddings if available
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rotary(q, k, cos, sin)

        # Expand KV for grouped query attention
        if num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1)
            k = k.reshape(bsz, num_heads, seq_len, head_dim)
            v = v.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1)
            v = v.reshape(bsz, num_heads, seq_len, head_dim)

        # Q @ K^T / sqrt(d) — hookable!
        qk_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask
            if causal_mask.dim() == 2:
                causal_mask = causal_mask[:, None, None, :]
            qk_logits = qk_logits + causal_mask

        qk_logits = hook_points.run(qk_logits_name, qk_logits)

        # Softmax — hookable!
        attn_weights = F.softmax(qk_logits, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = hook_points.run(weights_name, attn_weights)

        # Weights @ V — hookable!
        attn_output = torch.matmul(attn_weights, v)
        attn_output = hook_points.run(weighted_values_name, attn_output)

        # Reshape back to (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len, num_heads * head_dim)

        # Output projection
        out_proj = _get_out_proj(attn_module, proj_config)
        attn_output = out_proj(attn_output)

        return attn_output, None  # (output, attn_weights) — match HF signature

    attn_module.forward = wrapped_forward
    attn_module._omnilens_original_forward = original_forward
    attn_module._omnilens_wrapped = True


def unwrap_attention_module(attn_module: nn.Module) -> None:
    """Restore the original forward method."""
    if hasattr(attn_module, "_omnilens_original_forward"):
        attn_module.forward = attn_module._omnilens_original_forward
        del attn_module._omnilens_original_forward
        del attn_module._omnilens_wrapped


def _detect_projections(attn_module: nn.Module) -> dict | None:
    """Detect which projection style this attention module uses."""
    children = {name for name, _ in attn_module.named_children()}

    # Separate Q/K/V projections (Llama, Mistral, Gemma, etc.)
    if all(p in children for p in ("q_proj", "k_proj", "v_proj")):
        return {
            "style": "separate",
            "q": "q_proj", "k": "k_proj", "v": "v_proj",
            "out": next(
                n for n in ("o_proj", "out_proj", "dense") if n in children
            ),
        }

    # Fused QKV (GPT-2)
    if "c_attn" in children:
        return {
            "style": "fused_qkv",
            "qkv": "c_attn",
            "out": "c_proj",
        }

    # BERT-style
    if all(p in children for p in ("query", "key", "value")):
        return {
            "style": "separate",
            "q": "query", "k": "key", "v": "value",
            "out": "dense" if "dense" in children else "out_proj",
        }

    return None


def _get_qkv(
    attn_module: nn.Module,
    hidden_states: torch.Tensor,
    proj_config: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get Q, K, V tensors using the appropriate projection style."""
    if proj_config["style"] == "separate":
        q = getattr(attn_module, proj_config["q"])(hidden_states)
        k = getattr(attn_module, proj_config["k"])(hidden_states)
        v = getattr(attn_module, proj_config["v"])(hidden_states)
        return q, k, v

    if proj_config["style"] == "fused_qkv":
        qkv = getattr(attn_module, proj_config["qkv"])(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v

    raise ValueError(f"Unknown projection style: {proj_config['style']}")


def _get_out_proj(attn_module: nn.Module, proj_config: dict) -> nn.Module:
    """Get the output projection module."""
    return getattr(attn_module, proj_config["out"])


def _apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K.

    Handles both full and partial rotary embeddings. When cos/sin have
    a smaller dimension than Q/K, only the first rotary_dim dimensions
    get rotated and the rest pass through unchanged.
    """
    rotary_dim = cos.shape[-1]

    if rotary_dim < q.shape[-1]:
        # Partial rotary (Phi-2, Qwen 3, etc.)
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

    return q, k


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
