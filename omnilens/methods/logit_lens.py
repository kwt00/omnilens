from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from omnilens.core.tapped_model import TappedModel


@dataclass
class LogitLensResult:
    """Results from a logit lens analysis.

    Attributes:
        logits: Raw logits at each layer. Shape: (n_layers, batch, seq, vocab).
        probabilities: Softmax'd probabilities. Shape: (n_layers, batch, seq, vocab).
        predictions: Top predicted token IDs at each layer. Shape: (n_layers, batch, seq).
        tokens: Decoded top predictions. List of layers, each containing
            a list of positions, each a string token.
        layer_indices: Which layers were analyzed.
    """

    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    tokens: list[list[list[str]]]
    layer_indices: list[int]

    def __repr__(self) -> str:
        n_layers, batch, seq, vocab = self.logits.shape
        return (
            f"LogitLensResult(layers={n_layers}, batch={batch}, "
            f"seq={seq}, vocab={vocab})"
        )


def run_logit_lens(
    model: TappedModel,
    text: str | list[str] | None = None,
    input_ids: torch.Tensor | None = None,
    layers: list[int] | None = None,
) -> LogitLensResult:
    """Run logit lens analysis using only the Layer 1 public API.

    For each layer, takes the residual stream output, applies the final
    layer norm, and projects through the unembedding matrix to see what
    the model "would predict" at that intermediate point.

    Args:
        model: A TappedModel instance.
        text: Input text. Provide this or input_ids.
        input_ids: Token IDs. Provide this or text.
        layers: Which layers to analyze. None means all layers.
    """
    n_layers = model._detect_n_layers()
    if layers is None:
        layers = list(range(n_layers))

    # Cache residual stream at block output for each requested layer
    cache_names = [f"layers.{i}.residual.block_out" for i in layers]

    # Also need the final layer norm weight and unembedding weight
    cache_names.append("layer_norm_final.weight")
    cache_names.append("unembed.weight")

    # Try to include layer norm bias (not all architectures have it)
    try:
        resolved = model._resolve_name("layer_norm_final")
        ln_module = model._get_module(resolved)
        has_ln_bias = hasattr(ln_module, "bias") and ln_module.bias is not None
    except (KeyError, AttributeError):
        has_ln_bias = False

    if has_ln_bias:
        cache_names.append("layer_norm_final.bias")

    logits_out, cache = model.run_with_cache(
        text=text,
        input_ids=input_ids,
        names=cache_names,
    )

    # Get the final layer norm and unembedding
    ln_weight = cache["layer_norm_final.weight"]
    unembed_weight = cache["unembed.weight"]
    ln_bias = cache["layer_norm_final.bias"] if has_ln_bias else None

    # Get layer norm eps from config
    config = model.config
    ln_eps = getattr(config, "layer_norm_epsilon", None)
    if ln_eps is None:
        ln_eps = getattr(config, "layer_norm_eps", None)
    if ln_eps is None:
        ln_eps = getattr(config, "rms_norm_eps", 1e-6)

    # Check if this architecture uses RMSNorm (no bias, no mean subtraction)
    use_rms_norm = hasattr(config, "rms_norm_eps") or not hasattr(
        config, "layer_norm_epsilon"
    )

    # Get unembed bias if it exists
    unembed_bias = None
    try:
        resolved = model._resolve_name("unembed")
        unembed_module = model._get_module(resolved)
        if hasattr(unembed_module, "bias") and unembed_module.bias is not None:
            unembed_bias = unembed_module.bias.detach()
    except (KeyError, AttributeError):
        pass

    all_logits = []
    for i in layers:
        residual = cache[f"layers.{i}.residual.block_out"]

        # Apply final layer norm
        if use_rms_norm:
            normed = _rms_norm(residual, ln_weight, ln_eps)
        else:
            normed = F.layer_norm(
                residual,
                (residual.shape[-1],),
                weight=ln_weight,
                bias=ln_bias,
                eps=ln_eps,
            )

        # Project through unembedding
        layer_logits = normed @ unembed_weight.T
        if unembed_bias is not None:
            layer_logits = layer_logits + unembed_bias

        all_logits.append(layer_logits)

    logits_tensor = torch.stack(all_logits)  # (n_layers, batch, seq, vocab)
    probabilities = F.softmax(logits_tensor, dim=-1)
    predictions = logits_tensor.argmax(dim=-1)  # (n_layers, batch, seq)

    # Decode predictions to tokens
    tokens = []
    if model.tokenizer is not None:
        for layer_preds in predictions:
            layer_tokens = []
            for batch_preds in layer_preds:
                layer_tokens.append(
                    [model.tokenizer.decode(t.item()) for t in batch_preds]
                )
            tokens.append(layer_tokens)

    return LogitLensResult(
        logits=logits_tensor,
        probabilities=probabilities,
        predictions=predictions,
        tokens=tokens,
        layer_indices=layers,
    )


def _rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Apply RMS normalization (used by Llama, Mistral, etc.)."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x.float() * torch.rsqrt(variance + eps)
    return (x * weight.float()).to(x.dtype)
