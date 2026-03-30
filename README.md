# omnilens

Universal mechanistic interpretability for any PyTorch model.

omnilens wraps native HuggingFace models with zero-overhead PyTorch hooks — no reimplementation, no weight copying. One API works across transformers, SSMs (Mamba), linear attention (RWKV), and any architecture with auto-detection.

## Install

```bash
pip install omnilens
```

## Quick start

```python
from omnilens import TappedModel

model = TappedModel.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype="auto")
logits, cache = model.run_with_cache(text="The Eiffel Tower is in")
resid = cache["layers.15.residual.block_out"]   # (1, seq, 4096)
attn  = cache["layers.15.attention.weights"]    # (1, heads, seq, seq)
```

## Features

- **Zero reimplementation** — wraps the original HF model with native PyTorch hooks
- **Standardized naming** — `layers.{i}.attention.q`, `layers.{i}.residual.block_out`, etc. across all architectures
- **run_with_cache** — cache any combination of activations in one forward pass
- **run_with_hooks** — intervene on activations with arbitrary functions
- **Logit lens** — project intermediate residual states through the unembedding at every layer
- **Activation patching** — sweep components and measure causal effect on output
- **SAE** — composable sparse autoencoders (relu / topk / jumprelu / gated / custom), hook helpers, load pretrained from SAELens
- **Transcoder** — sparse MLP replacement with attach/detach context manager and cross-layer support
- **Probe** — linear and MLP probes with layer sweep via `{i}` expansion
- **SteeringVector** — construct from contrastive pairs, probes, or SAE features; apply with scale
- **Visualization** — `.plot()` on results, attention heatmaps (`pip install omnilens[viz]`)
- **Registry system** — built-in YAML registries for 18 architectures, auto-detect for unknown models, inline dict override

## Built-in architectures

**Transformers:** Llama 2/3/3.1, DeepSeek, Yi, Mistral, Gemma 1/2/3, Qwen 2/2.5/3, Phi-2, Phi-3, GPT-2, Pythia/GPT-NeoX, OPT, GPT-J, Falcon, BLOOM, StableLM

**Non-transformers:** Mamba (SSM), RWKV (linear attention)

Any model not listed above still works via raw module names or auto-detection.

## Documentation

See [docs/guide.md](docs/guide.md) for the full reference with examples for every feature.
