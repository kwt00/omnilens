from __future__ import annotations

from collections import Counter

import torch.nn as nn

from omnilens.registry.loader import Registry


# Common module name patterns for transformer components
ATTENTION_PATTERNS = {
    "q_proj", "k_proj", "v_proj", "o_proj",          # Llama, Mistral, Gemma
    "query", "key", "value", "dense",                  # BERT-style
    "c_attn", "c_proj",                                # GPT-2
    "q", "k", "v", "out",                              # generic
    "query_key_value",                                  # Falcon, Bloom (fused QKV)
    "W_Q", "W_K", "W_V", "W_O",                       # some research models
}

MLP_PATTERNS = {
    "up_proj", "down_proj", "gate_proj",               # Llama-style gated
    "c_fc", "c_proj",                                  # GPT-2
    "fc1", "fc2",                                      # OPT, generic
    "dense_h_to_4h", "dense_4h_to_h",                 # Bloom
    "w1", "w2", "w3",                                  # some research models
    "Wi_0", "Wi_1", "Wo",                              # T5-style
}

LAYERNORM_PATTERNS = {
    "input_layernorm", "post_attention_layernorm",     # Llama
    "ln_1", "ln_2",                                    # GPT-2
    "layer_norm", "layernorm",                         # generic
    "final_layer_norm",                                # OPT
    "norm", "norm1", "norm2",                          # various
}


def auto_detect_registry(model: nn.Module) -> Registry | None:
    """Attempt to infer a registry by inspecting the model's module tree.

    Walks the module tree looking for repeating block structures that
    contain attention and MLP components. Returns a Registry with '{i}'
    placeholders expanded for the detected number of layers, or None
    if detection fails.
    """
    blocks_path, n_layers = _find_repeating_blocks(model)
    if blocks_path is None:
        return None

    first_block = _get_module_by_path(model, f"{blocks_path}.0")
    if first_block is None:
        return None

    mapping = {}
    block_children = {name: mod for name, mod in first_block.named_modules() if name}

    attn_prefix = _find_submodule_group(block_children, ATTENTION_PATTERNS)
    mlp_prefix = _find_submodule_group(block_children, MLP_PATTERNS)

    if attn_prefix:
        attn_modules = {
            name: mod
            for name, mod in first_block.named_modules()
            if name.startswith(attn_prefix)
        }
        _map_attention(mapping, attn_modules, attn_prefix, blocks_path)

    if mlp_prefix:
        mlp_modules = {
            name: mod
            for name, mod in first_block.named_modules()
            if name.startswith(mlp_prefix)
        }
        _map_mlp(mapping, mlp_modules, mlp_prefix, blocks_path)

    _map_layernorms(mapping, block_children, blocks_path)
    _map_embeddings(mapping, model)
    _map_output(mapping, model)

    if not mapping:
        return None

    registry = Registry(mapping)
    return registry.expand_layers(n_layers)


def _find_repeating_blocks(model: nn.Module) -> tuple[str | None, int]:
    """Find the path to the repeating transformer blocks and count them."""
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 1:
            # Verify blocks have similar structure (attention + MLP)
            first_children = {n for n, _ in module[0].named_modules() if n}
            if len(first_children) > 3:  # non-trivial block
                return name, len(module)
    return None, 0


def _find_submodule_group(
    children: dict[str, nn.Module], patterns: set[str]
) -> str | None:
    """Find the common prefix for modules matching known patterns."""
    matches = []
    for name in children:
        leaf = name.split(".")[-1]
        if leaf in patterns:
            matches.append(name)

    if not matches:
        return None

    # Find the common prefix among matches
    if len(matches) == 1:
        parts = matches[0].split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else matches[0]

    prefixes = [".".join(m.split(".")[:-1]) for m in matches]
    # Return the most common prefix
    prefix_counts = Counter(prefixes)
    most_common = prefix_counts.most_common(1)[0][0]
    return most_common if most_common else None


def _map_attention(
    mapping: dict, modules: dict, attn_prefix: str, blocks_path: str
) -> None:
    """Map attention submodules to standardized names."""
    q_names = {"q_proj", "query", "q", "W_Q"}
    k_names = {"k_proj", "key", "k", "W_K"}
    v_names = {"v_proj", "value", "v", "W_V"}
    out_names = {"o_proj", "out", "dense", "c_proj", "W_O"}

    for name in modules:
        leaf = name.split(".")[-1]
        native = f"{blocks_path}.{{i}}.{name}"

        if leaf in q_names:
            mapping["layers.{i}.attention.q"] = native
        elif leaf in k_names:
            mapping["layers.{i}.attention.k"] = native
        elif leaf in v_names:
            mapping["layers.{i}.attention.v"] = native
        elif leaf in out_names and leaf != "c_proj":
            mapping["layers.{i}.attention.out_proj"] = native


def _map_mlp(
    mapping: dict, modules: dict, mlp_prefix: str, blocks_path: str
) -> None:
    """Map MLP submodules to standardized names."""
    up_names = {"up_proj", "c_fc", "fc1", "dense_h_to_4h", "w1", "Wi_0"}
    down_names = {"down_proj", "c_proj", "fc2", "dense_4h_to_h", "w2", "Wo"}
    gate_names = {"gate_proj", "w3", "Wi_1"}

    for name in modules:
        leaf = name.split(".")[-1]
        native = f"{blocks_path}.{{i}}.{name}"

        if leaf in up_names:
            mapping["layers.{i}.mlp.up_proj"] = native
        elif leaf in down_names:
            mapping["layers.{i}.mlp.down_proj"] = native
        elif leaf in gate_names:
            mapping["layers.{i}.mlp.gate_proj"] = native


def _map_layernorms(
    mapping: dict, children: dict, blocks_path: str
) -> None:
    """Map layer norm modules to standardized names."""
    attn_ln_names = {"input_layernorm", "ln_1", "norm1"}
    mlp_ln_names = {"post_attention_layernorm", "ln_2", "norm2"}

    for name in children:
        leaf = name.split(".")[-1]
        native = f"{blocks_path}.{{i}}.{name}"

        if leaf in attn_ln_names:
            mapping["layers.{i}.attention.layer_norm"] = native
        elif leaf in mlp_ln_names:
            mapping["layers.{i}.mlp.layer_norm"] = native


def _map_embeddings(mapping: dict, model: nn.Module) -> None:
    """Map embedding modules."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            if any(tok in name for tok in ("embed_tokens", "wte", "word_embeddings")):
                mapping["embed.tokens"] = name
            elif any(pos in name for pos in ("wpe", "position_embeddings")):
                mapping["embed.position"] = name


def _map_output(mapping: dict, model: nn.Module) -> None:
    """Map output modules (final layernorm, unembedding)."""
    for name, module in model.named_modules():
        if not name:
            continue
        leaf = name.split(".")[-1]

        if isinstance(module, (nn.LayerNorm,)) or leaf in ("norm", "ln_f", "final_layer_norm"):
            # Only map top-level norms, not those inside blocks
            depth = name.count(".")
            if depth <= 2:
                mapping["layer_norm_final"] = name

        if isinstance(module, nn.Linear) and leaf in ("lm_head",):
            mapping["unembed"] = name


def _get_module_by_path(model: nn.Module, path: str) -> nn.Module | None:
    """Get a module by dot-separated path, returning None on failure."""
    try:
        current = model
        for part in path.split("."):
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        return current
    except (AttributeError, IndexError, TypeError):
        return None
