# omnilens

Universal mechanistic interpretability for any PyTorch/HuggingFace model.

omnilens wraps native HuggingFace models with zero-overhead PyTorch hooks — no reimplementation, no weight copying. One API works across Llama, GPT-2, Mistral, Gemma, Qwen2, Phi-3, and any architecture with auto-detection.

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
- **SAE** — sparse autoencoders with relu / topk / jumprelu / gated activations, full hook helpers
- **Transcoder** — sparse MLP replacement with attach/detach context manager and cross-layer support
- **Probe** — linear and MLP probes with layer sweep via `{i}` expansion
- **SteeringVector** — construct from contrastive pairs, probes, or SAE features; apply with scale
- **Registry system** — built-in YAML registries for 7 architectures, auto-detect for unknown models, inline dict override

## Built-in architectures

Llama 2/3/3.1, Code Llama, Mistral, Gemma, Gemma 2, Qwen2, Phi-3, GPT-2

## Documentation

See [docs/guide.md](docs/guide.md) for the full reference with examples for every feature.
