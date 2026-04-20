"""Compress repeated blocks in a model graph.

Identifies structurally identical submodule sequences (e.g. 64 transformer
layers with the same architecture) and collapses them into a single visual
node with an "xN" badge. Also annotates edges with tensor volume and
split/merge flags for physical data flow visualization.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import torch.nn as nn

from omnilens.viz.live.models.graph_schema import GraphEdge, GraphNode, GraphResponse


def _fingerprint_module(module: nn.Module) -> str:
    """Create a structural fingerprint for a module subtree.

    Two modules with the same fingerprint are structurally identical:
    same layer types, same parameter shapes, same child structure.
    """
    parts = [type(module).__name__]

    # Parameter shapes (not values — just structure)
    for pname, param in module.named_parameters(recurse=False):
        parts.append(f"p:{pname}:{list(param.shape)}")

    # Buffer shapes
    for bname, buf in module.named_buffers(recurse=False):
        if buf is not None:
            parts.append(f"b:{bname}:{list(buf.shape)}")

    # Recurse into children (ordered)
    for cname, child in module.named_children():
        # Use relative name position, not actual name (so "layers.0" and "layers.1" match)
        child_fp = _fingerprint_module(child)
        parts.append(f"c:{child_fp}")

    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _find_repeated_sequences(model: nn.Module) -> list[dict]:
    """Find groups of consecutive sibling modules with identical structure.

    Returns a list of dicts:
      {
        "parent_name": "transformer",
        "children": ["transformer.layers.0", "transformer.layers.1", ...],
        "fingerprint": "abc123",
        "count": 64,
        "module_type": "TransformerEncoderLayer",
      }
    """
    groups = []

    for parent_name, parent_module in model.named_modules():
        # Only compress children of indexed containers (ModuleList, Sequential)
        # or parents whose children have numeric names (layers.0, layers.1, ...).
        # Do NOT compress named submodules within a single block
        # (e.g. norm1/norm2, dropout1/dropout2 are different roles).
        children = list(parent_module.named_children())
        if len(children) < 2:
            continue

        # Check if children have numeric names (indexed container)
        has_numeric = all(cname.isdigit() for cname, _ in children)
        is_indexed_container = isinstance(parent_module, (nn.ModuleList, nn.Sequential)) or has_numeric

        if not is_indexed_container:
            continue

        # Fingerprint each child
        child_fps = []
        for cname, cmod in children:
            full_name = f"{parent_name}.{cname}" if parent_name else cname
            fp = _fingerprint_module(cmod)
            child_fps.append((full_name, fp, cmod))

        # Find consecutive runs of the same fingerprint
        i = 0
        while i < len(child_fps):
            run_start = i
            fp = child_fps[i][1]
            while i < len(child_fps) and child_fps[i][1] == fp:
                i += 1
            run_len = i - run_start

            if run_len >= 2:
                groups.append({
                    "parent_name": parent_name,
                    "children": [child_fps[j][0] for j in range(run_start, run_start + run_len)],
                    "fingerprint": fp,
                    "count": run_len,
                    "module_type": type(child_fps[run_start][2]).__name__,
                })

    return groups


def _get_all_descendant_ids(module_name: str, model: nn.Module) -> set[str]:
    """Get all module IDs that are descendants of the given module."""
    ids = {module_name}
    prefix = module_name + "."
    for name, _ in model.named_modules():
        if name.startswith(prefix):
            ids.add(name)
    return ids


def compress_graph(
    graph: GraphResponse,
    model: nn.Module,
) -> GraphResponse:
    """Compress a graph by collapsing repeated blocks.

    Takes the full graph and model, identifies repeated structural patterns,
    and returns a compressed graph where repeated blocks become single nodes
    with repeat_count > 1.
    """
    repeated_groups = _find_repeated_sequences(model)

    if not repeated_groups:
        # Still annotate edges with volume/split/merge
        return _annotate_edges(graph)

    # Build a mapping: original_id -> compressed_id
    # For a group of N repeated children, keep the FIRST child's structure
    # and fold the rest into it.
    fold_map: dict[str, str] = {}  # original_id -> representative_id
    compressed_ids_map: dict[str, list[str]] = {}  # representative_id -> list of all folded IDs
    repeat_counts: dict[str, int] = {}  # representative_id -> count

    for group in repeated_groups:
        representative = group["children"][0]
        count = group["count"]
        folded = group["children"][1:]

        # Get all descendant IDs for the representative and folded blocks
        rep_descendants = _get_all_descendant_ids(representative, model)

        for folded_name in folded:
            folded_descendants = _get_all_descendant_ids(folded_name, model)
            for desc_id in folded_descendants:
                # Map folded descendant to the corresponding representative descendant
                suffix = desc_id[len(folded_name):]
                rep_id = representative + suffix
                fold_map[desc_id] = rep_id

        # Track metadata
        all_ids = []
        for child_name in group["children"]:
            all_ids.extend(_get_all_descendant_ids(child_name, model))
        compressed_ids_map[representative] = all_ids
        repeat_counts[representative] = count

    # Build new node list
    seen_nodes = set()
    new_nodes: list[GraphNode] = []
    for node in graph.nodes:
        if node.id in fold_map:
            # This node is folded — skip it
            continue
        if node.id in seen_nodes:
            continue
        seen_nodes.add(node.id)

        new_node = node.model_copy()
        if node.id in repeat_counts:
            new_node.repeat_count = repeat_counts[node.id]
            new_node.compressed_ids = compressed_ids_map.get(node.id, [])
            # Update label to show compression
            short = node.label.split(".")[-1] if "." in node.label else node.label
            new_node.label = f"{short} (x{repeat_counts[node.id]})"
        new_nodes.append(new_node)

    # Build new edge list — remap folded sources/targets to representative
    seen_edges = set()
    new_edges: list[GraphEdge] = []
    for edge in graph.edges:
        src = fold_map.get(edge.source, edge.source)
        tgt = fold_map.get(edge.target, edge.target)

        # Skip self-loops created by compression
        if src == tgt:
            continue

        # Skip if either endpoint was fully removed
        if src not in seen_nodes and src not in {n.id for n in new_nodes}:
            continue
        if tgt not in seen_nodes and tgt not in {n.id for n in new_nodes}:
            continue

        edge_key = (src, tgt)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        new_edges.append(GraphEdge(
            source=src,
            target=tgt,
            tensor_shape=edge.tensor_shape,
        ))

    compressed = GraphResponse(
        nodes=new_nodes,
        edges=new_edges,
        model_name=graph.model_name,
        total_params=graph.total_params,
        compressed=True,
    )

    return _annotate_edges(compressed)


def _annotate_edges(graph: GraphResponse) -> GraphResponse:
    """Annotate edges with tensor volume and split/merge flags."""
    # Count fan-out and fan-in for each node
    fan_out: dict[str, int] = {}
    fan_in: dict[str, int] = {}
    for edge in graph.edges:
        fan_out[edge.source] = fan_out.get(edge.source, 0) + 1
        fan_in[edge.target] = fan_in.get(edge.target, 0) + 1

    for edge in graph.edges:
        # Tensor volume
        if edge.tensor_shape:
            vol = 1
            for d in edge.tensor_shape:
                vol *= d
            edge.tensor_volume = vol

        # Split: source has fan-out > 1
        if fan_out.get(edge.source, 0) > 1:
            edge.is_split = True

        # Merge: target has fan-in > 1
        if fan_in.get(edge.target, 0) > 1:
            edge.is_merge = True

    return graph
