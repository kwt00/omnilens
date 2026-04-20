"""Computes per-layer metrics from hook data.

Metrics: activation stats, KL divergence, loss attribution, routing importance.
Supports aggregation across compressed (repeated) blocks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from omnilens.viz.live.config import settings
from omnilens.viz.live.engine.hook_manager import HookManager, HookData
from omnilens.viz.live.models.metric_schema import ActivationStats, LayerMetrics, MetricSnapshot


def _compute_activation_stats(data: HookData) -> ActivationStats:
    if not data.output_tensors:
        return ActivationStats()

    t = data.output_tensors[0].float()
    flat = t.flatten()

    if flat.numel() == 0:
        return ActivationStats()

    hist_counts, hist_edges = torch.histogram(
        flat.cpu(), bins=settings.histogram_bins
    )

    return ActivationStats(
        mean=float(flat.mean()),
        std=float(flat.std()) if flat.numel() > 1 else 0.0,
        min_val=float(flat.min()),
        max_val=float(flat.max()),
        histogram_counts=hist_counts.int().tolist(),
        histogram_edges=hist_edges.tolist(),
    )


def _compute_kl_divergence(data: HookData, reference_std: float = 1.0) -> float:
    if not data.output_tensors:
        return 0.0

    t = data.output_tensors[0].float().flatten()
    if t.numel() < 10:
        return 0.0

    hist_counts, hist_edges = torch.histogram(t.cpu(), bins=settings.histogram_bins)
    p = hist_counts.float() + 1e-8
    p = p / p.sum()

    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    mean_val = t.mean().item()
    q = torch.exp(-0.5 * ((bin_centers - mean_val) / reference_std) ** 2)
    q = q + 1e-8
    q = q / q.sum()

    kl = float(F.kl_div(q.log(), p, reduction="sum"))
    return max(0.0, kl)


def _compute_loss_attribution(fwd: HookData, bwd: HookData) -> float:
    if not fwd.output_tensors or not bwd.grad_output:
        return 0.0

    act = fwd.output_tensors[0]
    grad = bwd.grad_output[0]

    if act is None or grad is None:
        return 0.0
    if not isinstance(grad, torch.Tensor):
        return 0.0

    try:
        attribution = (act.float().abs() * grad.float().abs()).mean()
        return float(attribution)
    except Exception:
        return 0.0


class MetricCollector:
    def __init__(self, hook_manager: HookManager):
        self._hooks = hook_manager
        # compression_map: representative_id -> list of all folded IDs
        self._compression_map: dict[str, list[str]] = {}

    def set_compression_map(self, compressed_nodes: list) -> None:
        """Set mapping from compressed node IDs to their constituent layers."""
        self._compression_map = {}
        for node in compressed_nodes:
            if node.repeat_count > 1 and node.compressed_ids:
                self._compression_map[node.id] = node.compressed_ids

    def compute_snapshot(self) -> MetricSnapshot:
        """Compute current metrics for all hooked layers.

        If compression is active, metrics for repeated blocks are averaged
        into their representative node.
        """
        layers: dict[str, LayerMetrics] = {}

        for name in self._hooks.get_all_module_names():
            fwd = self._hooks.get_latest_forward(name)
            bwd = self._hooks.get_latest_backward(name)

            if fwd is None:
                continue

            activation = _compute_activation_stats(fwd)
            kl_div = _compute_kl_divergence(fwd)
            loss_attr = _compute_loss_attribution(fwd, bwd) if bwd else 0.0

            layers[name] = LayerMetrics(
                layer_id=name,
                activation=activation,
                kl_divergence=kl_div,
                loss_attribution=loss_attr,
            )

        if self._compression_map:
            layers = self._aggregate_compressed(layers)

        return MetricSnapshot(
            step=self._hooks.step,
            layers=layers,
        )

    def _aggregate_compressed(
        self, layers: dict[str, LayerMetrics]
    ) -> dict[str, LayerMetrics]:
        """Average metrics across compressed (folded) layers."""
        folded_to_rep: dict[str, str] = {}
        for rep_id, member_ids in self._compression_map.items():
            for mid in member_ids:
                if mid != rep_id:
                    folded_to_rep[mid] = rep_id

        rep_metrics: dict[str, list[LayerMetrics]] = {}
        final: dict[str, LayerMetrics] = {}

        for layer_id, metrics in layers.items():
            rep = folded_to_rep.get(layer_id)
            if rep is not None:
                rep_metrics.setdefault(rep, []).append(metrics)
            else:
                final[layer_id] = metrics

        for rep_id, metric_list in rep_metrics.items():
            if rep_id in final:
                metric_list.append(final[rep_id])

            n = len(metric_list)
            avg = LayerMetrics(
                layer_id=rep_id,
                activation=ActivationStats(
                    mean=sum(m.activation.mean for m in metric_list) / n,
                    std=sum(m.activation.std for m in metric_list) / n,
                    min_val=min(m.activation.min_val for m in metric_list),
                    max_val=max(m.activation.max_val for m in metric_list),
                    histogram_counts=metric_list[0].activation.histogram_counts,
                    histogram_edges=metric_list[0].activation.histogram_edges,
                ),
                kl_divergence=sum(m.kl_divergence for m in metric_list) / n,
                loss_attribution=sum(m.loss_attribution for m in metric_list) / n,
                routing_importance=max(m.routing_importance for m in metric_list),
            )
            final[rep_id] = avg

        return final
