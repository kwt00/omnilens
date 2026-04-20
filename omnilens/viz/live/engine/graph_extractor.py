"""Extract computational graph from any PyTorch nn.Module.

Strategy A: torch.fx symbolic tracing (clean IR, preferred).
Strategy B: Runtime hook-based tracing (fallback for dynamic models).
"""

from __future__ import annotations

import time
from collections import OrderedDict

import torch
import torch.nn as nn

from omnilens.viz.live.models.graph_schema import GraphEdge, GraphNode, GraphResponse


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters(recurse=False))


def _total_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _shape_to_list(t: torch.Tensor | tuple | list | None) -> list[int] | None:
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return list(t.shape)
    if isinstance(t, (tuple, list)) and len(t) > 0 and isinstance(t[0], torch.Tensor):
        return list(t[0].shape)
    return None


def _is_container(module: nn.Module) -> bool:
    container_types = (nn.Sequential, nn.ModuleList, nn.ModuleDict)
    return isinstance(module, container_types)


def _get_parent_group(name: str) -> str | None:
    parts = name.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else None


# ---------------------------------------------------------------------------
# Strategy A: torch.fx symbolic tracing
# ---------------------------------------------------------------------------

def _extract_via_fx(
    model: nn.Module,
    sample_input: torch.Tensor,
) -> GraphResponse | None:
    try:
        import torch.fx
    except ImportError:
        return None

    try:
        traced = torch.fx.symbolic_trace(model)
    except Exception:
        return None

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    fx_node_to_id: dict[str, str] = {}

    # Add module nodes
    named_modules = dict(model.named_modules())

    for fx_node in traced.graph.nodes:
        node_id = fx_node.name

        if fx_node.op == "placeholder":
            nodes.append(GraphNode(
                id=node_id,
                label=fx_node.name,
                module_type="input",
            ))
            fx_node_to_id[fx_node.name] = node_id

        elif fx_node.op == "call_module":
            target_module = named_modules.get(fx_node.target)
            module_type = type(target_module).__name__ if target_module else fx_node.target
            parent = _get_parent_group(fx_node.target)

            nodes.append(GraphNode(
                id=node_id,
                label=fx_node.target,
                module_type=module_type,
                parent_group=parent,
                num_params=_count_params(target_module) if target_module else 0,
                is_container=_is_container(target_module) if target_module else False,
            ))
            fx_node_to_id[fx_node.name] = node_id

        elif fx_node.op == "call_function":
            func_name = getattr(fx_node.target, "__name__", str(fx_node.target))
            nodes.append(GraphNode(
                id=node_id,
                label=func_name,
                module_type=f"F.{func_name}",
            ))
            fx_node_to_id[fx_node.name] = node_id

        elif fx_node.op == "call_method":
            nodes.append(GraphNode(
                id=node_id,
                label=fx_node.target,
                module_type=f"method.{fx_node.target}",
            ))
            fx_node_to_id[fx_node.name] = node_id

        elif fx_node.op == "output":
            nodes.append(GraphNode(
                id=node_id,
                label="output",
                module_type="output",
            ))
            fx_node_to_id[fx_node.name] = node_id

        # Build edges from args
        for arg in fx_node.args:
            if isinstance(arg, torch.fx.Node) and arg.name in fx_node_to_id:
                edges.append(GraphEdge(
                    source=fx_node_to_id[arg.name],
                    target=node_id,
                ))
            elif isinstance(arg, (tuple, list)):
                for a in arg:
                    if isinstance(a, torch.fx.Node) and a.name in fx_node_to_id:
                        edges.append(GraphEdge(
                            source=fx_node_to_id[a.name],
                            target=node_id,
                        ))

    # Also add container group nodes
    for name, module in model.named_modules():
        if name and _is_container(module):
            parent = _get_parent_group(name)
            existing_ids = {n.id for n in nodes}
            if name not in existing_ids:
                nodes.append(GraphNode(
                    id=name,
                    label=name,
                    module_type=type(module).__name__,
                    parent_group=parent,
                    num_params=0,
                    is_container=True,
                ))

    model_name = type(model).__name__
    return GraphResponse(
        nodes=nodes,
        edges=edges,
        model_name=model_name,
        total_params=_total_params(model),
    )


# ---------------------------------------------------------------------------
# Strategy B: Hook-based runtime tracing (fallback)
# ---------------------------------------------------------------------------

def _extract_via_hooks(
    model: nn.Module,
    sample_input: torch.Tensor,
) -> GraphResponse:
    nodes: list[GraphNode] = []
    execution_order: list[str] = []
    tensor_producers: dict[int, str] = {}  # tensor id -> module name
    handles = []
    leaf_modules: set[str] = set()

    # Identify leaf modules (no children) — only these get hooks for edge tracing
    for name, module in model.named_modules():
        if not name:
            continue
        has_children = len(list(module.children())) > 0
        if not has_children:
            leaf_modules.add(name)

    # Add input node
    nodes.append(GraphNode(id="input", label="input", module_type="input"))

    for name, module in model.named_modules():
        if not name:
            continue

        has_children = len(list(module.children())) > 0
        parent = _get_parent_group(name)

        node = GraphNode(
            id=name,
            label=name,
            module_type=type(module).__name__,
            parent_group=parent,
            num_params=_count_params(module),
            is_container=has_children,
        )
        nodes.append(node)

        # Only hook leaf modules for clean edge tracing
        if name not in leaf_modules:
            continue

        def make_hook(module_name: str):
            def hook(mod, inp, out):
                execution_order.append(module_name)

                # Record outputs as produced by this module
                if isinstance(out, torch.Tensor):
                    tensor_producers[id(out)] = module_name
                    for n in nodes:
                        if n.id == module_name:
                            n.output_shape = list(out.shape)
                            break
                elif isinstance(out, (tuple, list)):
                    for t in out:
                        if isinstance(t, torch.Tensor):
                            tensor_producers[id(t)] = module_name

            return hook

        h = module.register_forward_hook(make_hook(name))
        handles.append(h)

    # Add output node
    nodes.append(GraphNode(id="output", label="output", module_type="output"))

    # Run forward pass
    with torch.no_grad():
        model.eval()
        _ = model(sample_input)

    # Clean up hooks
    for h in handles:
        h.remove()

    # Build edges from execution order — connect each leaf module
    # to the next one in execution sequence. This gives a clean chain.
    edges: list[GraphEdge] = []
    if execution_order:
        # Connect input to first executed module
        edges.append(GraphEdge(source="input", target=execution_order[0]))

        # Connect consecutive modules
        for i in range(len(execution_order) - 1):
            src = execution_order[i]
            tgt = execution_order[i + 1]
            if src != tgt:
                edges.append(GraphEdge(source=src, target=tgt))

        # Connect last to output
        edges.append(GraphEdge(source=execution_order[-1], target="output"))

    # Deduplicate edges
    seen = set()
    unique_edges = []
    for e in edges:
        key = (e.source, e.target)
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    model_name = type(model).__name__
    return GraphResponse(
        nodes=nodes,
        edges=unique_edges,
        model_name=model_name,
        total_params=_total_params(model),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_graph(
    model: nn.Module,
    sample_input: torch.Tensor,
    compress: bool = True,
) -> GraphResponse:
    """Extract the computational graph from a PyTorch model.

    Tries torch.fx symbolic tracing first, falls back to hook-based
    runtime tracing for models with dynamic control flow.
    If compress=True, collapses repeated blocks into single xN nodes.
    """
    from omnilens.viz.live.engine.graph_compressor import compress_graph

    # Count actual submodules to gauge expected detail
    num_modules = sum(1 for n, _ in model.named_modules() if n)

    # Try fx tracing first
    result = _extract_via_fx(model, sample_input)
    # Only use fx if it captures enough detail (at least 30% of modules)
    if result is not None and len(result.nodes) > max(2, num_modules * 0.3):
        if compress:
            result = compress_graph(result, model)
        return result

    # Fallback to hook-based tracing (captures every submodule)
    result = _extract_via_hooks(model, sample_input)
    if compress:
        result = compress_graph(result, model)
    return result
