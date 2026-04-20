"""Detect and analyze routing patterns: attention heads, MoE gating, skip connections.

Uses a registry of detector functions that inspect module types and attributes
to identify routing-relevant structures.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from omnilens.viz.live.engine.hook_manager import HookManager
from omnilens.viz.live.models.metric_schema import LayerMetrics


# ---------------------------------------------------------------------------
# Attention head analysis
# ---------------------------------------------------------------------------

def detect_attention_module(module: nn.Module) -> bool:
    """Check if a module is an attention layer."""
    if isinstance(module, nn.MultiheadAttention):
        return True
    # Heuristic: check for common attention-related attributes
    cls_name = type(module).__name__.lower()
    return any(kw in cls_name for kw in ("attention", "mha", "selfattn", "crossattn"))


def compute_attention_entropy(hook_data) -> float | None:
    """Compute mean attention entropy from attention weights.

    Higher entropy = more uniform attention = less focused.
    """
    if not hook_data or not hook_data.output_tensors:
        return None

    # nn.MultiheadAttention returns (output, attention_weights)
    # attention_weights shape: (batch, num_heads, seq_len, seq_len) or (batch, seq, seq)
    for t in hook_data.output_tensors:
        if t.dim() >= 3:
            # Likely attention weights
            attn = t.float().clamp(min=1e-8)
            entropy = -(attn * attn.log()).sum(dim=-1).mean()
            return float(entropy)

    return None


# ---------------------------------------------------------------------------
# MoE gating analysis
# ---------------------------------------------------------------------------

def detect_moe_module(module: nn.Module) -> bool:
    """Check if a module is a Mixture-of-Experts gating layer."""
    cls_name = type(module).__name__.lower()
    has_moe_name = any(kw in cls_name for kw in ("moe", "mixture", "expert", "gating", "router"))
    has_gate_attr = hasattr(module, "gate") or hasattr(module, "router") or hasattr(module, "top_k")
    return has_moe_name or has_gate_attr


def compute_expert_utilization(hook_data) -> list[float] | None:
    """Compute per-expert utilization from gating outputs.

    Returns a list of utilization fractions (one per expert).
    """
    if not hook_data or not hook_data.output_tensors:
        return None

    for t in hook_data.output_tensors:
        if t.dim() >= 2:
            # Assume gating output: (batch, num_experts) probabilities
            probs = t.float()
            if probs.min() >= 0 and probs.max() <= 1.1:
                utilization = probs.mean(dim=0)
                return utilization.tolist()

    return None


# ---------------------------------------------------------------------------
# Skip connection analysis
# ---------------------------------------------------------------------------

def compute_skip_ratio(
    module_name: str,
    hook_manager: HookManager,
    graph_edges: list | None = None,
) -> float | None:
    """Estimate skip connection ratio by comparing input and output magnitudes.

    For residual connections (y = x + F(x)), the skip ratio is |x| / |y|.
    A ratio near 1.0 means the skip path dominates; near 0.0 means the
    residual branch dominates.
    """
    fwd = hook_manager.get_latest_forward(module_name)
    if fwd is None or not fwd.input_tensors or not fwd.output_tensors:
        return None

    inp = fwd.input_tensors[0].float()
    out = fwd.output_tensors[0].float()

    if inp.shape != out.shape:
        return None

    inp_mag = inp.abs().mean().item()
    out_mag = out.abs().mean().item()
    residual_mag = (out - inp).abs().mean().item()

    if out_mag < 1e-8:
        return None

    # skip_ratio = how much of the output comes from the identity path
    skip_ratio = 1.0 - (residual_mag / (out_mag + 1e-8))
    return max(0.0, min(1.0, skip_ratio))


# ---------------------------------------------------------------------------
# Unified routing analysis
# ---------------------------------------------------------------------------

class RoutingAnalyzer:
    def __init__(self, model: nn.Module, hook_manager: HookManager):
        self._model = model
        self._hooks = hook_manager

        # Pre-scan model for routing-relevant modules
        self._attention_modules: set[str] = set()
        self._moe_modules: set[str] = set()
        self._potential_skip_modules: set[str] = set()

        for name, module in model.named_modules():
            if not name:
                continue
            if detect_attention_module(module):
                self._attention_modules.add(name)
            if detect_moe_module(module):
                self._moe_modules.add(name)
            # Heuristic: modules where input and output could have same shape
            # (residual blocks, layer norms after residual, etc.)
            cls_name = type(module).__name__.lower()
            if any(kw in cls_name for kw in ("residual", "block", "layer")):
                self._potential_skip_modules.add(name)

    def enrich_metrics(self, layer_id: str, metrics: LayerMetrics) -> LayerMetrics:
        """Add routing-specific metrics to a layer's metrics."""
        # Attention entropy
        if layer_id in self._attention_modules:
            fwd = self._hooks.get_latest_forward(layer_id)
            entropy = compute_attention_entropy(fwd)
            if entropy is not None:
                metrics.attention_entropy = entropy
                # Normalize entropy to routing importance (inverted: lower entropy = higher importance)
                metrics.routing_importance = max(0.0, 1.0 - entropy / 5.0)

        # MoE expert utilization
        if layer_id in self._moe_modules:
            fwd = self._hooks.get_latest_forward(layer_id)
            utilization = compute_expert_utilization(fwd)
            if utilization is not None:
                metrics.expert_utilization = utilization
                # Routing importance = 1 - balance (imbalance = more interesting routing)
                mean_util = sum(utilization) / len(utilization) if utilization else 0
                variance = sum((u - mean_util) ** 2 for u in utilization) / len(utilization) if utilization else 0
                metrics.routing_importance = min(1.0, variance * 10)

        # Skip connection ratio
        if layer_id in self._potential_skip_modules:
            ratio = compute_skip_ratio(layer_id, self._hooks)
            if ratio is not None:
                metrics.skip_ratio = ratio
                # Higher skip ratio = the residual path matters less
                metrics.routing_importance = max(
                    metrics.routing_importance,
                    abs(ratio - 0.5) * 2  # How far from balanced
                )

        return metrics
