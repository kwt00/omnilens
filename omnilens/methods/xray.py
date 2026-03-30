from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from omnilens.methods.activation_patching import PatchingResult, run_activation_patching
from omnilens.methods.logit_lens import LogitLensResult, run_logit_lens

if TYPE_CHECKING:
    from omnilens.core.tapped_model import TappedModel


class XRay:
    """Analysis methods namespace for TappedModel.

    All methods here are built entirely on Layer 1's public API
    (run_with_cache, run_with_hooks, registry names).

    Access via model.xray:
        results = model.xray.logit_lens(text="Hello world")
    """

    def __init__(self, model: TappedModel) -> None:
        self._model = model

    def logit_lens(
        self,
        text: str | list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        layers: list[int] | None = None,
    ) -> LogitLensResult:
        """Run logit lens analysis.

        Projects the residual stream at each layer through the final
        layer norm and unembedding to see intermediate predictions.

        Args:
            text: Input text.
            input_ids: Token IDs.
            layers: Which layers to analyze. None means all.

        Returns:
            LogitLensResult with .logits, .probabilities, .predictions, .tokens
        """
        return run_logit_lens(
            self._model, text=text, input_ids=input_ids, layers=layers
        )

    def activation_patching(
        self,
        clean: str | list[str] | None = None,
        corrupted: str | list[str] | None = None,
        clean_ids: torch.Tensor | None = None,
        corrupted_ids: torch.Tensor | None = None,
        names: list[str] | None = None,
        answer_tokens: list[str] | None = None,
        metric: Callable[[torch.Tensor], float] | None = None,
        positions: list[int] | None = None,
        denoise: bool = False,
    ) -> PatchingResult:
        """Run activation patching analysis.

        Patches activations one component at a time and measures
        how the output changes.

        Args:
            clean: Clean input text.
            corrupted: Corrupted input text.
            names: Components to patch. Use '{i}' to sweep all layers.
            answer_tokens: [correct, incorrect] token strings for logit diff metric.
            metric: Custom metric function (overrides answer_tokens).
            positions: Token positions to patch. None means all.
            denoise: If False (default), run clean and patch in corrupted.
                If True, run corrupted and patch in clean.

        Returns:
            PatchingResult with .effects, .baseline_metric, .top_effects()
        """
        return run_activation_patching(
            self._model,
            clean=clean,
            corrupted=corrupted,
            clean_ids=clean_ids,
            corrupted_ids=corrupted_ids,
            names=names,
            answer_tokens=answer_tokens,
            metric=metric,
            positions=positions,
            denoise=denoise,
        )

    def plot_attention(
        self,
        text: str,
        layer: int,
        head: int | None = None,
        figsize: tuple[int, int] | None = None,
    ):
        """Plot attention patterns as heatmaps.

        Args:
            text: Input text.
            layer: Which layer to visualize.
            head: Specific head, or None for all heads.
            figsize: Figure size override.

        Returns:
            matplotlib Figure.
        """
        from omnilens.viz.plots import plot_attention
        return plot_attention(
            self._model, text=text, layer=layer, head=head, figsize=figsize
        )
