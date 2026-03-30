# omnilens Guide

omnilens is a mechanistic interpretability library for any PyTorch/HuggingFace model. It wraps native HF models with standardized hook access — no reimplementation, no weight copying. The same API works across Llama, GPT-2, Mistral, Gemma, Qwen2, Phi-3, and any architecture via auto-detection.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [TappedModel](#2-tappedmodel)
3. [Naming Scheme](#3-naming-scheme)
4. [ActivationCache](#4-activationcache)
5. [Registry System](#5-registry-system)
6. [Logit Lens](#6-logit-lens)
7. [Activation Patching](#7-activation-patching)
8. [SAE](#8-sae)
9. [Transcoder](#9-transcoder)
10. [Probe](#10-probe)
11. [SteeringVector](#11-steeringvector)
12. [Visualization](#12-visualization)
13. [Loading Pretrained SAEs](#13-loading-pretrained-saes)
14. [Supported Architectures](#14-supported-architectures)

---

## 1. Getting Started

### Install

```bash
pip install omnilens
```

### Load a model

```python
from omnilens import TappedModel

model = TappedModel.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype="auto")
```

### Run a forward pass and cache activations

```python
logits, cache = model.run_with_cache(text="The Eiffel Tower is in")

# Access by standardized name
resid = cache["layers.15.residual.block_out"]  # (1, seq, 4096)
q     = cache["layers.0.attention.q"]          # output of q_proj
```

### Run on GPU

```python
model = TappedModel.from_pretrained("meta-llama/Llama-3.1-8B", device_map="auto", torch_dtype="auto")
# or move after loading:
model.to("cuda")
```

---

## 2. TappedModel

`TappedModel` is the core class. It wraps any HuggingFace `AutoModelForCausalLM` and translates between standardized omnilens names and the model's native module paths.

### `TappedModel.from_pretrained`

```python
model = TappedModel.from_pretrained(
    model_name,               # str — HuggingFace model ID or local path
    registry=None,            # Registry | dict | str | Path | None
    tokenizer_name=None,      # str | None — defaults to model_name
    **model_kwargs,           # passed to AutoModelForCausalLM.from_pretrained
)
```

All `model_kwargs` go directly to `AutoModelForCausalLM.from_pretrained`. Common ones:

```python
model = TappedModel.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype="auto",       # or torch.float16, torch.bfloat16
    device_map="auto",        # multi-GPU / CPU offload
    load_in_8bit=True,        # bitsandbytes quantization
)
```

Registry resolution order when `registry=None`:

1. Built-in registry for this architecture (from `model.config.model_type`)
2. Auto-detect from model structure
3. Empty registry — raw module names still work

### `TappedModel.__init__`

For wrapping an already-loaded model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

raw_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = TappedModel(
    model=raw_model,
    tokenizer=tokenizer,
    registry=None,          # Registry | dict | str | Path | None
)
```

### `run_with_cache`

```python
logits, cache = model.run_with_cache(
    input_ids=None,         # torch.Tensor | None — (batch, seq)
    names=None,             # list[str] | None — names to cache; None = all registry names
    text=None,              # str | list[str] | None — tokenized automatically
    **model_kwargs,         # passed to model.forward (attention_mask, etc.)
)
# Returns: (logits: torch.Tensor, cache: ActivationCache)
```

Provide either `text` or `input_ids`, not both.

```python
# Cache specific names only (faster — skips everything else)
logits, cache = model.run_with_cache(
    text="Hello world",
    names=["layers.0.residual.block_out", "layers.15.residual.block_out"],
)

# Cache from token IDs
import torch
ids = torch.tensor([[1, 2, 3]])
logits, cache = model.run_with_cache(input_ids=ids)

# Batch input
logits, cache = model.run_with_cache(text=["sentence one", "sentence two"])
```

### `run_with_hooks`

Run a forward pass with intervention hooks. Each hook function receives `(activation, hook_name)` and can return a modified tensor or `None` (no modification).

```python
logits = model.run_with_hooks(
    input_ids=None,         # torch.Tensor | None
    hooks=None,             # dict[str, Callable] | None
    text=None,              # str | list[str] | None
    **model_kwargs,
)
# Returns: logits: torch.Tensor
```

```python
def zero_out_layer_10(activation, hook_name):
    return torch.zeros_like(activation)

logits = model.run_with_hooks(
    text="Hello world",
    hooks={"layers.10.residual.block_out": zero_out_layer_10},
)
```

Multiple hooks at once:

```python
logits = model.run_with_hooks(
    text="Hello world",
    hooks={
        "layers.5.attention.weights": lambda act, name: act.clamp(0, 1),
        "layers.10.residual.block_out": lambda act, name: act * 0.5,
    },
)
```

### `module_names`

```python
names = model.module_names()
# Returns: list[str] — all native module paths in the underlying model
# e.g. ["model.embed_tokens", "model.layers.0", "model.layers.0.self_attn", ...]
```

### `registry_names`

```python
names = model.registry_names()
# Returns: list[str] — all standardized names in the current registry
# e.g. ["embed.tokens", "layers.0.attention.q", "layers.0.attention.k", ...]
```

### `print_module_tree`

Prints the full module tree with types and native paths. Useful for understanding an unfamiliar model's structure.

```python
model.print_module_tree()
# embed_tokens (Embedding) -> model.embed_tokens
# layers (ModuleList) -> model.layers
#   0 (LlamaDecoderLayer) -> model.layers.0
#     self_attn (LlamaSdpaAttention) -> model.layers.0.self_attn
#       q_proj (Linear) -> model.layers.0.self_attn.q_proj
#       ...
```

### `.to(device)`

```python
model.to("cuda")
model.to("cpu")
model.to(torch.device("cuda:1"))
# Returns self for chaining
```

### `.generate`

Pass-through to the underlying model's `generate` method.

```python
output = model.generate(input_ids, max_new_tokens=50)
```

### `.config`

```python
cfg = model.config   # The HuggingFace ModelConfig
print(cfg.hidden_size, cfg.num_hidden_layers)
```

### `.device`

```python
print(model.device)  # torch.device('cuda:0')
```

### `.xray`

Lazy-loaded namespace for analysis methods.

```python
result = model.xray.logit_lens(text="Hello world")
result = model.xray.activation_patching(clean="...", corrupted="...", ...)
```

---

## 3. Naming Scheme

omnilens maps every common transformer component to a standardized dotted name. The same names work regardless of the underlying model architecture.

### Layer-indexed components

All per-layer names use `layers.{i}` where `{i}` is the zero-indexed layer number.

#### Attention components

| omnilens name | What it is |
|---|---|
| `layers.{i}.attention.layer_norm` | Pre-attention layer norm |
| `layers.{i}.attention.q` | Query projection output |
| `layers.{i}.attention.k` | Key projection output |
| `layers.{i}.attention.v` | Value projection output |
| `layers.{i}.attention.out_proj` | Output projection output |
| `layers.{i}.attention.qk_logits` | Q @ K^T / sqrt(d_head), pre-softmax |
| `layers.{i}.attention.weights` | Post-softmax attention weights |
| `layers.{i}.attention.weighted_values` | weights @ V, pre-output-projection |

#### MLP components

| omnilens name | What it is |
|---|---|
| `layers.{i}.mlp.layer_norm` | Pre-MLP layer norm |
| `layers.{i}.mlp.up_proj` | Up projection (or first linear) |
| `layers.{i}.mlp.gate_proj` | Gate projection (gated architectures) |
| `layers.{i}.mlp.down_proj` | Down projection (or second linear) |

#### Residual stream

| omnilens name | What it is |
|---|---|
| `layers.{i}.residual.input` | Residual stream entering layer i |
| `layers.{i}.residual.attn_out` | Residual stream after attention (input + attn output) |
| `layers.{i}.residual.block_out` | Residual stream after full block (input + attn + MLP) |

### Global components

| omnilens name | What it is |
|---|---|
| `embed.tokens` | Token embedding table |
| `embed.position` | Positional embedding table (GPT-2-style) |
| `layer_norm_final` | Final layer norm before unembedding |
| `unembed` | Unembedding / LM head |

### Suffixes

Append a suffix after any component name:

| Suffix | Meaning |
|---|---|
| `.activations` | Explicit marker for the module output (optional, stripped before lookup) |
| `.weight` | The module's weight parameter tensor (no hook, read directly) |
| `.bias` | The module's bias parameter tensor (no hook, read directly) |

```python
logits, cache = model.run_with_cache(
    text="hello",
    names=[
        "layers.0.attention.q",             # output of q_proj
        "layers.0.attention.q.activations", # same thing, explicit
        "layers.0.attention.q.weight",      # the q_proj weight matrix
        "layer_norm_final.weight",           # final norm scale
        "unembed.weight",                    # vocabulary embedding matrix
    ],
)
```

### Raw module name fallback

If a name is not in the registry but matches a valid native module path, it is used directly. This lets you access any module in the model even without a registry entry.

```python
# Use the native path directly when no registry mapping exists
logits, cache = model.run_with_cache(
    text="hello",
    names=["model.layers.0.self_attn.q_proj"],
)
```

### Examples

```python
# Layer 0 query output
cache["layers.0.attention.q"]           # shape: (batch, seq, d_model)

# Layer 16 residual stream after attention
cache["layers.16.residual.attn_out"]    # shape: (batch, seq, d_model)

# Layer 10 attention weights (all heads)
cache["layers.10.attention.weights"]    # shape: (batch, heads, seq, seq)

# Layer 10 pre-attention Q/K logits
cache["layers.10.attention.qk_logits"] # shape: (batch, heads, seq, seq)

# Q projection weight matrix
cache["layers.0.attention.q.weight"]   # shape: (d_model, d_model) or (d_head * n_heads, d_model)

# Final layer norm weight
cache["layer_norm_final.weight"]        # shape: (d_model,)
```

### Mamba (SSM)

Mamba layers use `layers.{i}.mixer.*` instead of `attention.*` / `mlp.*`:

```
layers.{i}.mixer.layer_norm
layers.{i}.mixer.in_proj
layers.{i}.mixer.conv1d
layers.{i}.mixer.x_proj
layers.{i}.mixer.dt_proj
layers.{i}.mixer.out_proj
```

### RWKV

RWKV layers use `layers.{i}.time_mix.*` and `layers.{i}.channel_mix.*` instead of `attention.*` / `mlp.*`:

```
layers.{i}.time_mix.layer_norm
layers.{i}.time_mix.key
layers.{i}.time_mix.value
layers.{i}.time_mix.receptance
layers.{i}.time_mix.output
layers.{i}.channel_mix.layer_norm
layers.{i}.channel_mix.key
layers.{i}.channel_mix.receptance
layers.{i}.channel_mix.value
```

---

## 4. ActivationCache

`ActivationCache` is a dict-like container returned by `run_with_cache`. Keys are omnilens names; values are detached tensors.

### Dict interface

```python
logits, cache = model.run_with_cache(text="hello")

# Access
resid = cache["layers.0.residual.block_out"]

# Check membership
"layers.0.residual.block_out" in cache  # True

# Iterate keys
for name in cache:
    print(name, cache[name].shape)

# .keys(), .values(), .items()
list(cache.keys())

# Length
len(cache)

# Repr (shows first/last 5 keys for large caches)
print(cache)
# ActivationCache(32 entries: ['embed.tokens', 'layers.0.attention.q', ...])
```

### `.to(device)`

Returns a new `ActivationCache` with all tensors moved to the target device. Does not modify in place.

```python
cache_cpu = cache.to("cpu")
cache_gpu = cache.to("cuda")
```

### `.detach()`

Returns a new `ActivationCache` with all tensors detached from the computation graph. Tensors returned by `run_with_cache` are already detached, so this is mainly useful when building a cache manually.

```python
cache2 = cache.detach()
```

### `.clear()`

Remove all entries from the cache.

```python
cache.clear()
len(cache)  # 0
```

---

## 5. Registry System

A `Registry` maps omnilens standardized names to native module paths in the underlying model. omnilens resolves the registry automatically, but you can override or extend it.

### How registry resolution works

When you create a `TappedModel`, omnilens tries registries in this order:

1. **Explicit argument** — pass `registry=...` to `from_pretrained` or `__init__`
2. **Built-in YAML** — looks up `model.config.model_type` against built-in architecture files
3. **Auto-detect** — inspects the module tree for known attention/MLP/layernorm patterns
4. **Empty registry** — raw module name fallback still works

### `Registry` class

```python
from omnilens import Registry

reg = Registry({
    "embed.tokens": "model.embed_tokens",
    "layers.{i}.attention.q": "model.layers.{i}.self_attn.q_proj",
    # ... more entries
})
```

`{i}` placeholders are expanded by `expand_layers(n_layers)`. Built-in registries contain `{i}` templates; when loaded for a specific model, they are expanded automatically.

```python
# Expand manually
expanded = reg.expand_layers(32)  # produces entries for layers 0–31
```

### Inline dict registry

Pass a plain dict and it will be wrapped in a `Registry`:

```python
model = TappedModel.from_pretrained(
    "my-custom-model",
    registry={
        "embed.tokens": "transformer.wte",
        "layers.{i}.attention.q": "transformer.h.{i}.attn.q",
        # ...
    },
)
```

### YAML file registry

```python
model = TappedModel.from_pretrained(
    "my-model",
    registry="/path/to/my_model.yaml",
)
```

YAML format:

```yaml
mapping:
  embed.tokens: model.embed_tokens
  layers.{i}.attention.q: model.layers.{i}.self_attn.q_proj
  layers.{i}.attention.k: model.layers.{i}.self_attn.k_proj
  layers.{i}.attention.v: model.layers.{i}.self_attn.v_proj
  layers.{i}.attention.out_proj: model.layers.{i}.self_attn.o_proj
  layers.{i}.mlp.layer_norm: model.layers.{i}.post_attention_layernorm
  layers.{i}.mlp.gate_proj: model.layers.{i}.mlp.gate_proj
  layers.{i}.mlp.up_proj: model.layers.{i}.mlp.up_proj
  layers.{i}.mlp.down_proj: model.layers.{i}.mlp.down_proj
  layer_norm_final: model.norm
  unembed: lm_head
```

Optionally include `n_layers: 32` in the YAML to expand `{i}` at load time.

### `load_registry`

```python
from omnilens.registry.loader import load_registry

# Load built-in by architecture name
reg = load_registry("llama")   # loads omnilens/registry/architectures/llama.yaml

# Load from file path
reg = load_registry("/path/to/my_model.yaml")
```

Returns `None` if the source is not found.

### `save_registry`

Save an auto-detected or manually built registry for reuse. Collapses numbered layer entries back to `{i}` templates.

```python
from omnilens import save_registry

save_registry(model._registry, "/path/to/my_model.yaml")
```

### `auto_detect_registry`

```python
from omnilens import auto_detect_registry

registry = auto_detect_registry(raw_hf_model)
```

Walks the module tree looking for repeating transformer blocks containing attention and MLP components. Matches against known naming patterns (q_proj, k_proj, v_proj, c_attn, up_proj, down_proj, etc.). Returns `None` if detection fails.

---

## 6. Logit Lens

The logit lens projects the residual stream at each layer through the final layer norm and unembedding matrix to see what the model "would predict" at that intermediate point.

### `model.xray.logit_lens`

```python
result = model.xray.logit_lens(
    text=None,          # str | list[str] | None
    input_ids=None,     # torch.Tensor | None
    layers=None,        # list[int] | None — None means all layers
)
# Returns: LogitLensResult
```

### `LogitLensResult` fields

| Field | Type | Description |
|---|---|---|
| `logits` | `torch.Tensor` | Raw logits at each layer. Shape: `(n_layers, batch, seq, vocab)` |
| `probabilities` | `torch.Tensor` | Softmax probabilities. Shape: `(n_layers, batch, seq, vocab)` |
| `predictions` | `torch.Tensor` | Argmax token IDs. Shape: `(n_layers, batch, seq)` |
| `tokens` | `list[list[list[str]]]` | Decoded top predictions. Indexed as `[layer][batch][position]` |
| `layer_indices` | `list[int]` | Which layers were analyzed |

### Example

```python
result = model.xray.logit_lens(text="The Eiffel Tower is in")

# What does the model predict at layer 15, position 5?
print(result.tokens[15][0][5])          # e.g. " Paris"

# Probability of the top prediction at each layer (last position)
top_probs = result.probabilities[:, 0, -1, :].max(dim=-1).values
print(top_probs)  # (n_layers,) tensor

# Analyze only specific layers
result = model.xray.logit_lens(text="hello", layers=[0, 8, 16, 24, 31])

# With token IDs
import torch
ids = torch.tensor([[1, 2, 3, 4, 5]])
result = model.xray.logit_lens(input_ids=ids)
```

---

## 7. Activation Patching

Activation patching identifies which components causally mediate a model's behavior. It caches activations from a source run and patches them into a target run one component at a time, measuring the effect on the output metric.

**Default mode** (`denoise=False`): run on the clean input, patch in activations from the corrupted run. Components that restore corrupted behavior have large positive effects.

**Denoise mode** (`denoise=True`): run on the corrupted input, patch in activations from the clean run. Components that recover clean behavior have large positive effects.

### `model.xray.activation_patching`

```python
result = model.xray.activation_patching(
    clean=None,             # str | list[str] | None
    corrupted=None,         # str | list[str] | None
    clean_ids=None,         # torch.Tensor | None
    corrupted_ids=None,     # torch.Tensor | None
    names=None,             # list[str] | None — components to patch
    answer_tokens=None,     # list[str] | None — [correct, incorrect] tokens
    metric=None,            # Callable[[torch.Tensor], float] | None
    positions=None,         # list[int] | None — token positions to patch; None = all
    denoise=False,          # bool
)
# Returns: PatchingResult
```

Either `text`/`corrupted` or `clean_ids`/`corrupted_ids` must be provided. Either `answer_tokens` or `metric` must be provided.

### `PatchingResult` fields

| Field | Type | Description |
|---|---|---|
| `effects` | `dict[str, float]` | Effect of patching each component (patched_metric - baseline_metric) |
| `baseline_metric` | `float` | Metric on the unpatched baseline run |
| `patched_metrics` | `dict[str, float]` | Metric value when each component is patched |
| `names` | `list[str]` | All components that were patched |

### `PatchingResult.top_effects(k=10)`

```python
top = result.top_effects(k=5)
# Returns: list[tuple[str, float]] — top-k by absolute effect size
for name, effect in top:
    print(f"{name}: {effect:.4f}")
```

### Layer sweep with `{i}`

```python
# Sweep all residual stream positions across every layer
result = model.xray.activation_patching(
    clean="The Colosseum is in Rome",
    corrupted="The Colosseum is in Paris",
    names=["layers.{i}.residual.block_out"],
    answer_tokens=[" Rome", " Paris"],
)

for name, effect in result.top_effects(k=5):
    print(f"{name}: {effect:.4f}")
```

### Specific components

```python
result = model.xray.activation_patching(
    clean="The Colosseum is in Rome",
    corrupted="The Colosseum is in Paris",
    names=[
        "layers.{i}.attention.out_proj",
        "layers.{i}.mlp.down_proj",
    ],
    answer_tokens=[" Rome", " Paris"],
)
```

### Custom metric

```python
def my_metric(logits):
    # logits: (batch, seq, vocab)
    return logits[0, -1, target_token_id].item()

result = model.xray.activation_patching(
    clean="...",
    corrupted="...",
    names=["layers.{i}.residual.block_out"],
    metric=my_metric,
)
```

### Patch specific token positions only

```python
result = model.xray.activation_patching(
    clean="The Eiffel Tower is in Paris",
    corrupted="The Colosseum is in Rome",
    names=["layers.{i}.residual.block_out"],
    answer_tokens=[" Paris", " Rome"],
    positions=[2, 3],    # patch only token positions 2 and 3
)
```

### Denoise mode

```python
result = model.xray.activation_patching(
    clean="...",
    corrupted="...",
    names=["layers.{i}.residual.block_out"],
    answer_tokens=[" correct", " wrong"],
    denoise=True,  # run corrupted, patch in clean — find what restores behavior
)
```

---

## 8. SAE

`SAE` (Sparse Autoencoder) encodes model activations into a high-dimensional sparse feature space and reconstructs them. It supports multiple activation functions, deep encoder/decoder architectures, tied weights, and custom encoders/decoders.

### Constructor

```python
from omnilens import SAE

sae = SAE(
    d_model,                  # int — input/output dimension
    n_features,               # int — number of sparse features (dictionary size)
    activation="relu",        # str | Callable — see below
    sparsity=None,            # Callable | None — custom sparsity loss; None = default for activation
    hidden_dims=None,         # list[int] | None — hidden layers for deep encoder/decoder
    tied_weights=False,       # bool — tie decoder weights to encoder (transposed)
    encoder=None,             # nn.Module | None — custom encoder; must pair with decoder
    decoder=None,             # nn.Module | None — custom decoder; must pair with encoder
    k=32,                     # int — number of active features for topk activation
    initial_threshold=0.001,  # float — initial threshold for jumprelu activation
)
```

### Built-in activations

| String | Behavior | Default sparsity |
|---|---|---|
| `"relu"` | ReLU on pre-activations | L1 (mean absolute value of features) |
| `"topk"` | Keep top-k features by magnitude, zero the rest | None (L0 is not differentiable) |
| `"jumprelu"` | Zero features below a learnable per-feature threshold | L0 |
| `"gated"` | Gated: sigmoid(gate_linear(x)) * relu(encoder(x)) | L1 |

### Common variants

```python
# Standard ReLU SAE
sae = SAE(d_model=4096, n_features=32768, activation="relu")

# TopK SAE — keeps exactly k features active
sae = SAE(d_model=4096, n_features=32768, activation="topk", k=64)

# JumpReLU — learnable threshold per feature
sae = SAE(d_model=4096, n_features=32768, activation="jumprelu", initial_threshold=0.01)

# Gated SAE
sae = SAE(d_model=4096, n_features=32768, activation="gated")

# Deep encoder/decoder with hidden layers
sae = SAE(d_model=4096, n_features=32768, activation="topk", k=64, hidden_dims=[8192])

# Tied weights — decoder is encoder.weight.T
sae = SAE(d_model=4096, n_features=32768, activation="relu", tied_weights=True)

# Custom activation function
def my_activation(pre_activations):
    return torch.relu(pre_activations - 0.1)  # shifted ReLU

sae = SAE(d_model=4096, n_features=32768, activation=my_activation, sparsity=my_sparsity_fn)

# Fully custom encoder/decoder
sae = SAE(
    d_model=4096,
    n_features=32768,
    encoder=my_encoder_module,
    decoder=my_decoder_module,
)
```

### Forward pass

```python
result = sae(activation_tensor)   # activation_tensor: (..., d_model)
# result.features             — sparse feature activations
# result.reconstruction       — reconstructed activation
# result.loss                 — total loss (recon + sparsity)
# result.reconstruction_loss  — MSE between reconstruction and input
# result.sparsity_loss        — sparsity penalty
```

### Encode / decode

```python
features = sae.encode(x)       # (..., n_features)
recon    = sae.decode(features) # (..., d_model)
```

### Training with `fit`

```python
logs = sae.fit(
    model,                  # TappedModel
    hook_point,             # str — which activation to train on
    dataset,                # torch.Tensor | list[str] | HF Dataset
    lr=3e-4,                # float
    sparsity_coeff=1.0,     # float — weight on sparsity loss
    batch_size=32,          # int
    n_steps=10000,          # int
    log_every=100,          # int — print frequency
    device=None,            # str | torch.device | None — defaults to model.device
)
# Returns: list[dict] with keys: step, loss, reconstruction_loss, sparsity_loss, n_active_features
```

```python
sae = SAE(d_model=4096, n_features=32768, activation="topk", k=64)
logs = sae.fit(
    model,
    hook_point="layers.16.residual.block_out",
    dataset=my_texts,   # list of strings
    n_steps=50000,
    sparsity_coeff=5.0,
)
```

### Training with `fit_on_activations`

When you already have cached activations (e.g. from a previous `run_with_cache` sweep):

```python
logs = sae.fit_on_activations(
    activations,            # torch.Tensor — (n_samples, d_model)
    lr=3e-4,
    sparsity_coeff=1.0,
    batch_size=32,
    n_steps=10000,
    log_every=100,
)
```

### Hook helpers

All hook helpers return a callable suitable for use in `run_with_hooks`.

#### `hook_ablate` — zero out specific features

```python
# Zero out features 42 and 99 in every forward pass
hook = sae.hook_ablate([42, 99])
logits = model.run_with_hooks(
    text="hello",
    hooks={"layers.16.residual.block_out": hook},
)
```

#### `hook_amplify` — scale a single feature

```python
# Amplify feature 42 by 3x
hook = sae.hook_amplify(feature=42, scale=3.0)
logits = model.run_with_hooks(
    text="hello",
    hooks={"layers.16.residual.block_out": hook},
)
```

#### `hook_clamp` — fix a feature to a constant value

```python
# Force feature 42 to always be 1.0
hook = sae.hook_clamp(feature=42, value=1.0)
logits = model.run_with_hooks(
    text="hello",
    hooks={"layers.16.residual.block_out": hook},
)
```

#### `hook_reconstruct` — replace activations with SAE reconstruction

```python
# Replace activations with SAE(encode then decode) — measures SAE reconstruction fidelity
hook = sae.hook_reconstruct()
logits = model.run_with_hooks(
    text="hello",
    hooks={"layers.16.residual.block_out": hook},
)
```

### Save and load

```python
# Save to directory (creates weights.pt and config.json)
sae.save("./my_sae")

# Load
sae = SAE.load("./my_sae")
```

### `TiedDecoder`

A decoder module that shares weights with the encoder (transposed). Exported from `omnilens` for use in custom decoder construction.

```python
from omnilens import TiedDecoder

encoder = nn.Linear(d_model, n_features, bias=True)
decoder = TiedDecoder(encoder)  # decoder(features) == F.linear(features, encoder.weight.T)
```

### Loading pretrained SAEs from SAELens

`SAE.from_saelens` loads weights from a SAELens-format directory or HuggingFace release. It maps SAELens weight keys (`W_enc`, `W_dec`, `b_enc`, `b_dec`) to omnilens format automatically and auto-detects the activation type (`standard` → `relu`, `topk`, `jumprelu`, `gated`).

```python
# Load from a local SAELens directory
sae = SAE.from_saelens("./path/to/sae_dir/")

# Load from HuggingFace (SAELens release)
sae = SAE.from_saelens("gemma-2b-res-jb", sae_id="blocks.0.hook_resid_post")

# Use it immediately
features = sae.encode(cache["layers.0.residual.block_out"])
```

---

## 9. Transcoder

`Transcoder` is a sparse MLP replacement. Unlike an SAE (which reconstructs its own input), a transcoder encodes MLP input and predicts MLP output through a sparse bottleneck. It can replace a single layer's MLP or predict across multiple output layers (cross-layer transcoder).

### Constructor

```python
from omnilens import Transcoder

tc = Transcoder(
    d_input,                  # int — MLP input dimension
    d_output,                 # int — MLP output dimension
    n_features,               # int — sparse bottleneck size
    activation="relu",        # str | Callable — same options as SAE
    sparsity=None,            # Callable | None
    hidden_dims=None,         # list[int] | None
    tied_weights=False,       # bool
    encoder=None,             # nn.Module | None — custom encoder
    decoder=None,             # nn.Module | None — custom decoder
    k=32,                     # int — for topk activation
    initial_threshold=0.001,  # float — for jumprelu activation
    skip=False,               # bool — add a linear skip connection from input to output
    output_layers=None,       # list[int] | None — for cross-layer transcoder
)
```

### Common variants

```python
# Standard single-layer transcoder
tc = Transcoder(d_input=4096, d_output=4096, n_features=32768, activation="topk", k=64)

# With skip connection (input is linearly projected and added to the transcoder output)
tc = Transcoder(d_input=4096, d_output=4096, n_features=32768, activation="relu", skip=True)

# Cross-layer transcoder — one decoder per output layer
tc = Transcoder(
    d_input=4096,
    d_output=4096,
    n_features=32768,
    activation="topk",
    k=64,
    output_layers=[16, 17, 18],  # contributes to layers 16, 17, and 18
)
```

### Forward pass

```python
result = tc(mlp_input, mlp_output)   # mlp_output=None during inference
# result.features         — sparse features
# result.prediction       — predicted MLP output
# result.loss             — prediction_loss + sparsity_loss
# result.prediction_loss  — MSE between prediction and mlp_output
# result.sparsity_loss    — sparsity penalty
```

### Encode / decode

```python
features   = tc.encode(mlp_input)             # (..., n_features)
prediction = tc.decode(features)              # (..., d_output)
prediction = tc.decode(features, layer=16)    # for cross-layer transcoder
```

### Attach and detach

`attach` replaces a layer's MLP forward method with the transcoder. `detach` restores the original.

```python
# Manual attach/detach
tc.attach(model, layer=16)
logits = model(input_ids)    # layer 16 MLP is now the transcoder
tc.detach(model, layer=16)

# Context manager (recommended)
with tc.attached(model, layer=16):
    logits = model(input_ids)   # transcoder active inside block
# MLP is restored here
```

### `attached` context manager

```python
with tc.attached(model, layer=16) as tc_ctx:
    # tc_ctx is the transcoder itself
    logits = model(input_ids)
    features = tc.last_features   # features from the last forward pass
```

### Feature ablation during inference

```python
tc.ablate_features([0, 1, 2])   # zero these features on every forward pass
with tc.attached(model, layer=16):
    logits = model(input_ids)

tc.restore_features()            # remove ablations
```

### Training with `fit`

```python
logs = tc.fit(
    model,                          # TappedModel
    input_point="layers.16.mlp.layer_norm",   # hook point for MLP input
    output_point="layers.16.mlp.down_proj",   # hook point for MLP output
    dataset=my_texts,
    lr=3e-4,
    sparsity_coeff=1.0,
    batch_size=32,
    n_steps=10000,
    log_every=100,
    device=None,
)
```

### Training with `fit_on_activations`

```python
logs = tc.fit_on_activations(
    inputs,         # torch.Tensor — (n_samples, d_input)
    outputs,        # torch.Tensor — (n_samples, d_output)
    lr=3e-4,
    sparsity_coeff=1.0,
    batch_size=32,
    n_steps=10000,
    log_every=100,
)
```

### Save and load

```python
tc.save("./my_transcoder")
tc = Transcoder.load("./my_transcoder")
```

---

## 10. Probe

`Probe` trains a small linear or MLP classifier/regressor on cached activations to measure what information is encoded at a given hook point. `Probe.sweep` trains a probe at every layer with a single call.

### Constructor

```python
from omnilens import Probe

probe = Probe(
    d_model,            # int — activation dimension
    n_classes=1,        # int — number of output classes; 1 implies regression
    task=None,          # str | None — "classification" or "regression"; inferred if None
    hidden_dims=None,   # list[int] | None — hidden layer dims; None = linear probe
    loss_fn=None,       # Callable | None — custom loss; None = CrossEntropy or MSE
)
```

Task inference: if `task=None`, uses `"regression"` when `n_classes <= 1`, otherwise `"classification"`.

```python
# Binary classification probe (linear)
probe = Probe(d_model=4096, n_classes=2)

# 5-class classification (linear)
probe = Probe(d_model=4096, n_classes=5, task="classification")

# Regression probe (linear)
probe = Probe(d_model=4096, task="regression")

# MLP probe
probe = Probe(d_model=4096, n_classes=2, hidden_dims=[256, 64])

# Custom loss
probe = Probe(d_model=4096, n_classes=2, loss_fn=nn.BCEWithLogitsLoss())
```

### `fit`

```python
result = probe.fit(
    model,              # TappedModel
    hook_point,         # str — which activation to probe
    texts,              # list[str]
    labels,             # list | torch.Tensor — int for classification, float for regression
    position=-1,        # int | str — token position; -1=last, int=specific, "all"=every token
    lr=1e-3,            # float
    n_epochs=50,        # int
    batch_size=32,      # int
    val_fraction=0.2,   # float — fraction held out for validation
    device=None,        # str | torch.device | None
)
# Returns: dict with "val_loss" and optionally "accuracy"
```

After training, `probe.accuracy` and `probe.train_loss` are set.

```python
probe = Probe(d_model=4096, n_classes=2)
result = probe.fit(
    model,
    hook_point="layers.20.residual.block_out",
    texts=["This is great", "This is terrible", ...],
    labels=[1, 0, ...],    # 1=positive, 0=negative
)
print(f"Validation accuracy: {probe.accuracy:.3f}")
```

#### Token-level probing (`position="all"`)

```python
# Probe every token position (e.g. for NER or POS tagging)
probe.fit(
    model,
    hook_point="layers.10.residual.block_out",
    texts=sentences,
    labels=per_token_labels,   # list of lists — one list of ints per sentence
    position="all",
)
```

### `fit_on_activations`

Train directly on pre-cached activations:

```python
_, cache = model.run_with_cache(text=texts, names=["layers.10.residual.block_out"])
acts = cache["layers.10.residual.block_out"][:, -1, :]   # last token

result = probe.fit_on_activations(
    activations=acts,
    labels=labels,
    lr=1e-3,
    n_epochs=50,
    batch_size=32,
    val_fraction=0.2,
)
```

### `Probe.sweep`

Train a probe at every layer (or a list of hook points) in one call.

```python
sweep_result = Probe.sweep(
    model,
    hook_points,            # str with {i} or list[str]
    texts,                  # list[str]
    labels,                 # list | torch.Tensor
    n_classes=2,            # int
    task=None,              # str | None
    hidden_dims=None,       # list[int] | None
    position=-1,            # int | str
    lr=1e-3,                # float
    n_epochs=50,            # int
    batch_size=32,          # int
    val_fraction=0.2,       # float
    device=None,            # str | torch.device | None
)
# Returns: ProbeResult
```

```python
# Find which layer best encodes sentiment
sweep_result = Probe.sweep(
    model,
    hook_points="layers.{i}.residual.block_out",   # expanded to all layers
    texts=sentiment_texts,
    labels=sentiment_labels,
    n_classes=2,
)

# Best layer by accuracy
best_layer = max(sweep_result.accuracies, key=sweep_result.accuracies.get)
print(f"Best layer: {best_layer} ({sweep_result.accuracies[best_layer]:.3f})")

# Retrieve the probe for a specific layer
probe = sweep_result.probes["layers.20.residual.block_out"]
```

### `ProbeResult` fields

| Field | Type | Description |
|---|---|---|
| `accuracies` | `dict[str, float]` | Validation accuracy per hook point (classification only) |
| `losses` | `dict[str, float]` | Validation loss per hook point |
| `probes` | `dict[str, Probe]` | Trained probe per hook point |

### Save and load

```python
probe.save("./my_probe")
probe = Probe.load("./my_probe")
# probe.accuracy and probe.train_loss are restored from config.json
```

---

## 11. SteeringVector

`SteeringVector` is a direction in activation space that can be added to (or subtracted from) the residual stream during inference to shift model behavior.

### Constructors

#### `from_contrastive`

Compute the mean activation difference between two sets of texts:

```python
from omnilens import SteeringVector

vec = SteeringVector.from_contrastive(
    model,
    hook_point="layers.15.residual.block_out",
    positive=["I love this", "This is wonderful", ...],
    negative=["I hate this", "This is terrible", ...],
    position=-1,    # int — which token position to use; -1 = last token
)
```

The direction is `mean(positive_activations) - mean(negative_activations)`.

#### `from_pair`

Single-pair shortcut:

```python
vec = SteeringVector.from_pair(
    model,
    hook_point="layers.15.residual.block_out",
    positive="I love this",
    negative="I hate this",
    position=-1,
)
```

#### `from_probe`

Extract the direction from a trained linear probe's weight row:

```python
probe = Probe(d_model=4096, n_classes=2)
probe.fit(model, hook_point="layers.15.residual.block_out", texts=texts, labels=labels)

vec = SteeringVector.from_probe(
    probe,
    hook_point="layers.15.residual.block_out",
    class_idx=1,    # int — which class direction to extract (default 1 = positive class)
)
```

#### `from_sae_feature`

Extract the direction from an SAE feature's decoder column:

```python
sae = SAE.load("./my_sae")

vec = SteeringVector.from_sae_feature(
    sae,
    feature=42,                              # int — feature index
    hook_point="layers.15.residual.block_out",
)
```

The decoder column for a feature IS the direction that feature represents in activation space.

#### Raw tensor

```python
vec = SteeringVector(
    direction=my_tensor,   # torch.Tensor — shape (d_model,)
    hook_point="layers.15.residual.block_out",
)
```

### Applying a steering vector

#### `hook(scale=1.0)` — single hook point

```python
hooks = vec.hook(scale=2.0)
# Returns: {"layers.15.residual.block_out": <hook_fn>}

logits = model.run_with_hooks(
    text="The movie was",
    hooks=vec.hook(scale=2.0),   # add 2x the direction
)

# Subtract the direction (steer in the opposite direction)
logits = model.run_with_hooks(
    text="The movie was",
    hooks=vec.hook(scale=-2.0),
)
```

#### `hooks(layers, scale=1.0, hook_template=None)` — multiple layers

```python
hooks = vec.hooks(
    layers=[10, 11, 12, 13, 14, 15],
    scale=1.5,
    hook_template=None,   # str | None — template with {i}; inferred from hook_point if None
)
# Returns: {"layers.10.residual.block_out": fn, "layers.11.residual.block_out": fn, ...}

logits = model.run_with_hooks(
    text="The movie was",
    hooks=vec.hooks(layers=list(range(10, 20)), scale=2.0),
)
```

With a custom template:

```python
hooks = vec.hooks(
    layers=[5, 10, 15],
    scale=1.0,
    hook_template="layers.{i}.attention.out_proj",
)
```

### Utilities

#### `normalize()`

Returns a unit-length copy of the vector:

```python
vec_unit = vec.normalize()
```

#### `cosine_similarity(other)`

```python
sim = vec1.cosine_similarity(vec2)   # float in [-1, 1]
```

### Save and load

```python
vec.save("./my_vector")
vec = SteeringVector.load("./my_vector")
```

### Full example

```python
# Build a "positive sentiment" steering vector
positive_texts = ["I love this movie", "Excellent, highly recommend", "Amazing experience"]
negative_texts = ["I hate this movie", "Terrible, avoid it", "Awful experience"]

vec = SteeringVector.from_contrastive(
    model,
    hook_point="layers.15.residual.block_out",
    positive=positive_texts,
    negative=negative_texts,
)

# Steer toward positive sentiment
logits = model.run_with_hooks(
    text="The food was",
    hooks=vec.hook(scale=3.0),
)

# Normalize and save
vec.normalize().save("./sentiment_vector")
```

---

## 12. Visualization

omnilens includes built-in plotting for attention patterns and result objects. Matplotlib is optional — install it with:

```bash
pip install omnilens[viz]
```

All plot functions return a `matplotlib.figure.Figure` for further customization or saving.

### Attention patterns

Plot attention weights for a single head or all heads in a layer:

```python
# All heads in layer 5
fig = model.xray.plot_attention(text="The cat sat on the mat", layer=5)

# Single head
fig = model.xray.plot_attention(text="The cat sat on the mat", layer=5, head=3)
```

### Logit lens

```python
results = model.xray.logit_lens(text="The capital of France is")
fig = results.plot()
```

### Activation patching

```python
results = model.xray.activation_patching(
    clean="The capital of France is",
    corrupted="The capital of Germany is",
    names=["layers.{i}.residual.block_out"],
    answer_tokens=[" Paris", " Berlin"],
)
fig = results.plot()
```

### Probe sweep

```python
results = Probe.sweep(
    model,
    hook_points="layers.{i}.residual.block_out",
    texts=texts,
    labels=labels,
    n_classes=2,
)
fig = results.plot()
```

### Saving figures

```python
# Save any figure returned by .plot() or plot_attention()
fig.savefig("my_plot.png", dpi=150)
```

---

## 13. Loading Pretrained SAEs

See [`SAE.from_saelens`](#loading-pretrained-saes-from-saelens) in the SAE section above for loading SAELens weights directly.

---

## 14. Supported Architectures

The following architectures have built-in YAML registries and are supported out of the box:

| Architecture | model_type | Models | Registry |
|---|---|---|---|
| GPT-2 | gpt2 | GPT-2 | gpt2.yaml |
| Llama | llama | Llama 2/3/3.1, DeepSeek, Yi, Code Llama | llama.yaml |
| Mistral | mistral | Mistral, Mixtral | mistral.yaml |
| Gemma | gemma | Gemma 1 | gemma.yaml |
| Gemma 2 | gemma2 | Gemma 2 | gemma2.yaml |
| Gemma 3 | gemma3 | Gemma 3 | gemma3.yaml |
| Qwen 2 | qwen2 | Qwen 2, Qwen 2.5 | qwen2.yaml |
| Qwen 3 | qwen3 | Qwen 3 | qwen3.yaml |
| Phi-2 | phi | Phi-2 | phi.yaml |
| Phi-3 | phi3 | Phi-3, Phi-3.5 | phi3.yaml |
| GPT-NeoX / Pythia | gpt_neox | Pythia, GPT-NeoX | gpt_neox.yaml |
| OPT | opt | OPT | opt.yaml |
| GPT-J | gptj | GPT-J | gptj.yaml |
| Falcon | falcon | Falcon | falcon.yaml |
| BLOOM | bloom | BLOOM, BLOOMZ | bloom.yaml |
| StableLM | stablelm | StableLM 2 | stablelm.yaml |
| Mamba | mamba | Mamba (SSM) | mamba.yaml |
| RWKV | rwkv | RWKV-4 | rwkv.yaml |

Note: Mamba uses `layers.{i}.mixer.*` naming and RWKV uses `layers.{i}.time_mix.*` and `layers.{i}.channel_mix.*` instead of `attention.*` / `mlp.*`. See the [Naming Scheme](#3-naming-scheme) section for details.

For any architecture not listed, omnilens falls back to auto-detection, which inspects the module tree for known attention/MLP patterns. Auto-detection covers many Llama-derived models (e.g. Vicuna, Alpaca, Mistral variants) even if not explicitly listed.

If auto-detection fails or produces an incorrect mapping, use an inline dict registry or write a YAML file:

```python
# Inline dict — fastest approach for a single experiment
model = TappedModel.from_pretrained(
    "my-custom-arch",
    registry={
        "embed.tokens": "model.embed_tokens",
        "layers.{i}.attention.layer_norm": "model.layers.{i}.input_layernorm",
        "layers.{i}.attention.q": "model.layers.{i}.self_attn.q_proj",
        "layers.{i}.attention.k": "model.layers.{i}.self_attn.k_proj",
        "layers.{i}.attention.v": "model.layers.{i}.self_attn.v_proj",
        "layers.{i}.attention.out_proj": "model.layers.{i}.self_attn.o_proj",
        "layers.{i}.mlp.layer_norm": "model.layers.{i}.post_attention_layernorm",
        "layers.{i}.mlp.gate_proj": "model.layers.{i}.mlp.gate_proj",
        "layers.{i}.mlp.up_proj": "model.layers.{i}.mlp.up_proj",
        "layers.{i}.mlp.down_proj": "model.layers.{i}.mlp.down_proj",
        "layer_norm_final": "model.norm",
        "unembed": "lm_head",
    },
)
```

To inspect what was detected for any model:

```python
print(model._registry)              # Registry(N entries)
print(model.registry_names()[:10])  # first 10 standardized names
model.print_module_tree()           # full native module tree
```
