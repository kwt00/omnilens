"""Visualization functions for omnilens results.

All functions use matplotlib and return the figure for further customization.
Called internally by .plot() methods on result objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from omnilens.core.tapped_model import TappedModel
    from omnilens.methods.logit_lens import LogitLensResult
    from omnilens.methods.activation_patching import PatchingResult
    from omnilens.methods.probe import ProbeResult


def plot_attention(
    model: TappedModel,
    text: str,
    layer: int,
    head: int | None = None,
    figsize: tuple[int, int] | None = None,
) -> object:
    """Plot attention patterns as heatmaps.

    Args:
        model: TappedModel instance.
        text: Input text.
        layer: Which layer to visualize.
        head: Specific head index, or None for all heads.
        figsize: Figure size override.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    # Get attention weights
    _, cache = model.run_with_cache(
        text=text, names=[f"layers.{layer}.attention.weights"]
    )
    weights = cache[f"layers.{layer}.attention.weights"]  # (batch, heads, seq, seq)
    weights = weights[0].detach().cpu().float()  # (heads, seq, seq)

    # Get token labels
    tokens = _get_token_labels(model, text)
    n_heads = weights.shape[0]

    if head is not None:
        # Single head
        if figsize is None:
            figsize = (6, 5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        _plot_single_attention(ax, weights[head], tokens, f"Layer {layer}, Head {head}")
        fig.tight_layout()
        return fig

    # All heads
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols
    if figsize is None:
        figsize = (4 * cols, 3.5 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_heads == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for h in range(n_heads):
        r, c = divmod(h, cols)
        _plot_single_attention(axes[r][c], weights[h], tokens, f"Head {h}")

    # Hide unused subplots
    for h in range(n_heads, rows * cols):
        r, c = divmod(h, cols)
        axes[r][c].axis("off")

    fig.suptitle(f"Layer {layer} Attention Patterns", fontsize=14)
    fig.tight_layout()
    return fig


def plot_logit_lens(
    result: LogitLensResult,
    position: int = -1,
    top_k: int = 1,
    figsize: tuple[int, int] | None = None,
) -> object:
    """Plot logit lens predictions across layers.

    Shows what the model would predict at each layer for a given position.

    Args:
        result: LogitLensResult from model.xray.logit_lens().
        position: Token position to visualize. -1 for last token.
        top_k: Show top-k predictions per layer.
        figsize: Figure size override.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    probs = result.probabilities[:, 0, position, :].detach().cpu().float()  # (n_layers, vocab)
    predictions = result.predictions[:, 0, position].detach().cpu()  # (n_layers,)
    n_layers = probs.shape[0]
    layers = result.layer_indices

    if figsize is None:
        figsize = (10, max(4, n_layers * 0.4))

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Get top-k token labels per layer
    labels = []
    top_probs = []
    for i in range(n_layers):
        topk_vals, topk_ids = probs[i].topk(top_k)
        layer_labels = []
        for val, tid in zip(topk_vals, topk_ids):
            if result.tokens and len(result.tokens) > i:
                token_str = result.tokens[i][0][position]
            else:
                token_str = str(tid.item())
            layer_labels.append(f"{token_str} ({val.item():.2f})")
        labels.append("\n".join(layer_labels))
        top_probs.append(topk_vals[0].item())

    y_pos = np.arange(n_layers)
    bars = ax.barh(y_pos, top_probs, color="steelblue", edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Layer {l}" for l in layers])
    ax.set_xlabel("Top Token Probability")
    ax.set_title("Logit Lens — Predictions Across Layers")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    # Add token labels on bars
    for i, (bar, label) in enumerate(zip(bars, labels)):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=9,
        )

    fig.tight_layout()
    return fig


def plot_patching(
    result: PatchingResult,
    figsize: tuple[int, int] | None = None,
) -> object:
    """Plot activation patching effects as a bar chart.

    Args:
        result: PatchingResult from model.xray.activation_patching().
        figsize: Figure size override.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(result.effects.keys())
    effects = [result.effects[n] for n in names]
    n = len(names)

    # Simplify labels: "layers.5.residual.block_out" -> "L5 residual"
    short_names = []
    for name in names:
        parts = name.split(".")
        if "layers" in parts:
            idx = parts.index("layers")
            layer_num = parts[idx + 1]
            rest = ".".join(parts[idx + 2:])
            short_names.append(f"L{layer_num} {rest}")
        else:
            short_names.append(name)

    if figsize is None:
        figsize = (8, max(4, n * 0.35))

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = ["#e74c3c" if e < 0 else "#2ecc71" for e in effects]
    y_pos = np.arange(n)
    ax.barh(y_pos, effects, color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names)
    ax.set_xlabel("Effect on Metric")
    ax.set_title("Activation Patching Effects")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.invert_yaxis()

    fig.tight_layout()
    return fig


def plot_probe_sweep(
    result: ProbeResult,
    figsize: tuple[int, int] | None = None,
) -> object:
    """Plot probe accuracy across hook points / layers.

    Args:
        result: ProbeResult from Probe.sweep().
        figsize: Figure size override.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if result.accuracies:
        data = result.accuracies
        ylabel = "Accuracy"
        title = "Probe Accuracy Across Layers"
    else:
        data = result.losses
        ylabel = "Loss"
        title = "Probe Loss Across Layers"

    names = list(data.keys())
    values = [data[n] for n in names]

    # Extract layer numbers for cleaner labels
    short_names = []
    for name in names:
        parts = name.split(".")
        if "layers" in parts:
            idx = parts.index("layers")
            short_names.append(f"Layer {parts[idx + 1]}")
        else:
            short_names.append(name)

    if figsize is None:
        figsize = (10, 5)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x_pos = np.arange(len(names))
    ax.plot(x_pos, values, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if result.accuracies:
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
        ax.legend()

    fig.tight_layout()
    return fig


def _plot_single_attention(ax, weights, tokens, title):
    """Plot a single attention head heatmap."""
    import numpy as np

    w = weights.detach().cpu().float().numpy()
    im = ax.imshow(w, cmap="Blues", vmin=0, vmax=w.max())
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")


def _get_token_labels(model, text):
    """Get token string labels for visualization."""
    if model.tokenizer is None:
        return [f"t{i}" for i in range(100)]
    input_ids = model.tokenizer.encode(text)
    return [model.tokenizer.decode(t) for t in input_ids]
