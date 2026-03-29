# omnilens — Plan vs. Status

## Library Identity

| Item | Planned | Status |
|------|---------|--------|
| Name | `omnilens` | Done |
| PyPI available | Yes | Confirmed available |
| Core class | `TappedModel` | Done |
| Language | Pure Python | Done |
| Dependencies | `torch`, `transformers`, `pyyaml` | Done |
| Package structure | `omnilens/{core,registry,methods}` | Done |

---

## Architecture (Layer 1 — Primitives)

### TappedModel API

| Feature | Planned | Status |
|---------|---------|--------|
| `TappedModel.from_pretrained()` | Load any HF model, auto-resolve registry | Done |
| `run_with_cache(text=..., names=...)` | Forward pass, return (logits, cache) | Done |
| `run_with_cache(input_ids=...)` | Accept raw token tensors | Done |
| `run_with_hooks(hooks={name: fn})` | Intervene on activations mid-forward | Done |
| `module_names()` | List raw HF module paths | Done |
| `registry_names()` | List standardized omnilens names | Done |
| `print_module_tree()` | Pretty-print module hierarchy | Done |
| Raw module name fallback | Use HF names when no registry exists | Done |
| Hooks always cleaned up | Hooks removed even on error (try/finally) | Done |

### ActivationCache

| Feature | Planned | Status |
|---------|---------|--------|
| Dict-like access `cache["name"]` | Store tensors by hook name | Done |
| `.keys()`, `.values()`, `.items()` | Standard dict interface | Done |
| `.to(device)` | Move all tensors to a device | Done |
| `.detach()` | Detach from computation graph | Done |
| `.clear()` | Free memory | Done |
| Readable `__repr__` | Shows entry count and sample keys | Done |

### Attention Wrapping

| Feature | Planned | Status |
|---------|---------|--------|
| Wrap attention forward (not whole model) | Replace only attention math, keep native HF projections | Done |
| `attention.qk_logits` | Pre-softmax Q@K^T/sqrt(d) — cache and intervene | Done |
| `attention.weights` | Post-softmax attention weights — cache and intervene | Done |
| `attention.weighted_values` | weights @ V — cache and intervene | Done |
| Seamless API | User doesn't know derived vs. module hooks | Done |
| Separate Q/K/V projections | Llama, Mistral, Gemma style | Done |
| Fused QKV projections | GPT-2 style (c_attn split into Q/K/V) | Done |
| BERT-style projections | query/key/value naming | Done (detection code, untested) |
| GQA (grouped query attention) | Expand KV heads for multi-group | Done (tested on SmolLM2) |
| RoPE support | Apply rotary embeddings to Q/K | Done |
| Unwrap / restore original forward | `unwrap_attention_module()` | Done |

---

## Naming Scheme

### Planned Full Scheme

| Name | Description | Status |
|------|-------------|--------|
| **Embeddings** | | |
| `embed.tokens.activations` | Token embedding output | Partial — mapped as `embed.tokens` (module hook, no `.activations` suffix yet) |
| `embed.tokens.weight` | Token embedding weight matrix | Not yet — need `.weight` access pattern |
| `embed.position.activations` | Positional embedding output | Partial — mapped as `embed.position` |
| `embed.position.weight` | Positional embedding weight | Not yet |
| **Attention** | | |
| `layers.{i}.attention.layer_norm.activations` | LayerNorm output before attention | Partial — mapped as `layers.{i}.attention.layer_norm` |
| `layers.{i}.attention.layer_norm.weight` | LayerNorm weight | Not yet |
| `layers.{i}.attention.layer_norm.bias` | LayerNorm bias | Not yet |
| `layers.{i}.attention.q.activations` | Query projection output | Partial — mapped as `layers.{i}.attention.q` |
| `layers.{i}.attention.q.weight` | Query weight matrix | Not yet |
| `layers.{i}.attention.q.bias` | Query bias | Not yet |
| `layers.{i}.attention.k.activations` | Key projection output | Partial — same as above |
| `layers.{i}.attention.k.weight` | Key weight matrix | Not yet |
| `layers.{i}.attention.v.activations` | Value projection output | Partial — same as above |
| `layers.{i}.attention.v.weight` | Value weight matrix | Not yet |
| `layers.{i}.attention.qk_logits` | Pre-softmax QK^T/sqrt(d) | Done |
| `layers.{i}.attention.weights` | Post-softmax attention | Done |
| `layers.{i}.attention.weighted_values` | weights @ V | Done |
| `layers.{i}.attention.out_proj.activations` | Output projection activations | Partial — mapped as `layers.{i}.attention.out_proj` |
| `layers.{i}.attention.out_proj.weight` | Output projection weight | Not yet |
| `layers.{i}.attention.out_proj.bias` | Output projection bias | Not yet |
| **MLP** | | |
| `layers.{i}.mlp.layer_norm.activations` | LayerNorm output before MLP | Partial |
| `layers.{i}.mlp.layer_norm.weight` | LayerNorm weight | Not yet |
| `layers.{i}.mlp.layer_norm.bias` | LayerNorm bias | Not yet |
| `layers.{i}.mlp.up_proj.activations` | Up projection output | Partial |
| `layers.{i}.mlp.up_proj.weight` | Up projection weight | Not yet |
| `layers.{i}.mlp.gate_proj.activations` | Gate projection output (gated archs only) | Partial |
| `layers.{i}.mlp.gate_proj.weight` | Gate projection weight | Not yet |
| `layers.{i}.mlp.activation_fn.activations` | After activation function | Not yet — no hook point for this |
| `layers.{i}.mlp.down_proj.activations` | Down projection output | Partial |
| `layers.{i}.mlp.down_proj.weight` | Down projection weight | Not yet |
| `layers.{i}.mlp.down_proj.bias` | Down projection bias | Not yet |
| **Residual stream** | | |
| `layers.{i}.residual.input` | Residual stream entering block | Not yet |
| `layers.{i}.residual.attn_out` | After attention added back | Not yet |
| `layers.{i}.residual.block_out` | After MLP added back | Not yet |
| **Output** | | |
| `layer_norm_final.activations` | Final layer norm output | Partial — mapped as `layer_norm_final` |
| `layer_norm_final.weight` | Final layer norm weight | Not yet |
| `unembed.activations` | Unembedding output | Partial — mapped as `unembed` |
| `unembed.weight` | Unembedding weight matrix | Not yet |

### Naming Scheme Summary

The `.activations` / `.weight` / `.bias` suffix system is **designed but not yet implemented**. Currently, hook names map to modules directly (e.g. `layers.0.attention.q` hooks the `q_proj` module and captures its output). Adding the suffix system requires a resolver that distinguishes between "give me the output tensor" vs "give me the parameter tensor."

---

## Registry System

| Feature | Planned | Status |
|---------|---------|--------|
| **Priority: Built-in → Auto-detect → YAML → Dict** | | |
| Built-in YAML registries | Ship with package | Done (llama.yaml, gpt2.yaml) |
| Auto-detect from model structure | Infer registry by inspecting module tree | Done |
| Load from YAML file | `registry="./my_registry.yaml"` | Done |
| Inline dict | `registry={"name": "path"}` | Done |
| `{i}` placeholder expansion | Expand for N layers | Done |
| **Architecture coverage** | | |
| Llama / Llama 2 / Llama 3 | Built-in registry | Done, tested on SmolLM2-135M |
| GPT-2 | Built-in registry | Done, tested on tiny-gpt2 |
| Mistral | Built-in registry (same structure as Llama) | Not yet (llama.yaml likely works, untested) |
| Gemma | Built-in registry | Not yet |
| Qwen | Built-in registry | Not yet |
| **User-contributed registries** | | |
| Local YAML files | Load from path | Done |
| `auto_detect_registry()` standalone function | Propose a mapping for unknown models | Done |
| Save auto-detected registry to YAML | `save_registry(registry, path)` | Not yet |
| Interactive auto-detect (`Accept? [y/n]`) | Print proposed mapping, ask user | Not yet |

---

## Layer 2 — Built-in Methods

| Feature | Planned | Status |
|---------|---------|--------|
| Logit lens | Project residual stream through unembedding at each layer | Not yet |
| Activation patching | Swap activations between clean/corrupted runs | Not yet |
| Steering vectors | Add direction to residual stream | Not yet |
| Probing utilities | Train linear probes on cached activations | Not yet |
| SAE base classes | Encoder/decoder scaffold for training SAEs | Not yet |

**Design constraint**: All Layer 2 methods must be implementable using only Layer 1's public API.

---

## Infrastructure

| Feature | Planned | Status |
|---------|---------|--------|
| `pyproject.toml` | Package config | Done |
| Test suite | pytest | Done (19 tests, all passing) |
| Git repo | Initialized | Not yet |
| `.gitignore` | Standard Python | Not yet |
| CI / GitHub Actions | Auto-run tests | Not yet |
| PyPI publication | `pip install omnilens` | Not yet |
| README | Usage examples | Not yet |

---

## Remaining Design Decisions

1. **`.activations` / `.weight` / `.bias` suffix system** — How to implement the resolver that distinguishes between hooking a module's output vs accessing its parameters. Parameters don't need hooks (they're just `module.weight`), so this might just be a cache/accessor pattern rather than a hook.

2. **Residual stream hooks** — `residual.input`, `residual.attn_out`, `residual.block_out` require hooking at block boundaries. Need to identify which modules correspond to these points per architecture.

3. **`activation_fn.activations`** — The activation function (SiLU, GELU, etc.) is usually applied inline, not as a separate module. Same derived-value approach as attention, or skip it?

4. **MoE support** — Router logits, per-expert MLPs. Naming scheme: `layers.{i}.mlp.router.activations`, `layers.{i}.mlp.experts.{j}.up_proj.activations`?

5. **Fused QKV splitting in registries** — YAML format for specifying split instructions (GPT-2 `c_attn` → separate Q/K/V). Currently handled by the attention wrapper, but the registry doesn't reflect the split.

---

## Work Split (Kevin + Jack)

| Person | Scope |
|--------|-------|
| **Kevin** | Core infrastructure (TappedModel, hooks, attention wrap, registry system) |
| **Jack** | Architecture YAML files, Layer 2 methods, tests, docs |

---

## Test Results

```
19 passed — Python 3.14, pytest 9.0.2
Tested on: sshleifer/tiny-gpt2, HuggingFaceTB/SmolLM2-135M
```
