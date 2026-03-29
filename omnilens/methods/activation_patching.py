from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from omnilens.core.tapped_model import TappedModel


@dataclass
class PatchingResult:
    """Results from activation patching.

    Attributes:
        effects: Dict mapping each patched name to its effect (scalar).
            Effect = how much the metric changed when this component was patched.
        baseline_metric: The metric value with no patching.
        patched_metrics: Dict mapping each patched name to its metric value.
        names: List of names that were patched.
    """

    effects: dict[str, float]
    baseline_metric: float
    patched_metrics: dict[str, float]
    names: list[str]

    def __repr__(self) -> str:
        return f"PatchingResult({len(self.effects)} components patched)"

    def top_effects(self, k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k components by absolute effect size."""
        sorted_effects = sorted(
            self.effects.items(), key=lambda x: abs(x[1]), reverse=True
        )
        return sorted_effects[:k]


def run_activation_patching(
    model: TappedModel,
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

    Patches activations one component at a time and measures how the
    output changes. By default (denoise=False), runs the clean input
    and patches in corrupted activations to find what breaks.

    Args:
        model: A TappedModel instance.
        clean: Clean input text.
        corrupted: Corrupted input text.
        clean_ids: Clean input token IDs.
        corrupted_ids: Corrupted input token IDs.
        names: Which components to patch. Supports '{i}' to sweep all layers.
            e.g. ["layers.{i}.residual.block_out"] patches every layer.
        answer_tokens: Token strings to track logit difference.
            First token is the "correct" answer, second is the "incorrect" answer.
            Effect = change in (correct_logit - incorrect_logit).
        metric: Custom metric function. Takes logits tensor, returns a float.
            Overrides answer_tokens if both provided.
        positions: Token positions to patch. None means patch all positions.
        denoise: If False (default), run clean and patch in corrupted activations.
            If True, run corrupted and patch in clean activations.
    """
    n_layers = model._detect_n_layers()

    # Expand {i} in names
    expanded_names = []
    for name in (names or []):
        if "{i}" in name:
            for i in range(n_layers):
                expanded_names.append(name.replace("{i}", str(i)))
        else:
            expanded_names.append(name)

    if not expanded_names:
        raise ValueError("Must provide at least one name to patch.")

    # Build the metric function
    if metric is None:
        if answer_tokens is None or len(answer_tokens) < 2:
            raise ValueError(
                "Provide answer_tokens (at least 2) or a custom metric function."
            )
        metric = _make_logit_diff_metric(model, answer_tokens)

    # Tokenize both inputs together to ensure same length (padding)
    if clean is not None and corrupted is not None and model.tokenizer is not None:
        clean_list = [clean] if isinstance(clean, str) else clean
        corrupted_list = [corrupted] if isinstance(corrupted, str) else corrupted
        encoded = model.tokenizer(
            clean_list + corrupted_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        n_clean = len(clean_list)
        clean_ids = encoded["input_ids"][:n_clean].to(model.device)
        corrupted_ids = encoded["input_ids"][n_clean:].to(model.device)
        clean = None
        corrupted = None

    # Cache activations from the source (what we'll patch in)
    if denoise:
        source_logits, source_cache = model.run_with_cache(
            input_ids=clean_ids, names=expanded_names,
        )
        baseline_logits, _ = model.run_with_cache(
            input_ids=corrupted_ids, names=[],
        )
    else:
        source_logits, source_cache = model.run_with_cache(
            input_ids=corrupted_ids, names=expanded_names,
        )
        baseline_logits, _ = model.run_with_cache(
            input_ids=clean_ids, names=[],
        )

    baseline_metric = metric(baseline_logits)

    # Patch each component one at a time
    patched_metrics = {}
    effects = {}

    for name in expanded_names:
        source_activation = source_cache[name]

        def make_patch_fn(src_act: torch.Tensor, pos: list[int] | None):
            def patch_fn(activation, hook_name):
                if pos is not None:
                    patched = activation.clone()
                    patched[:, pos, ...] = src_act[:, pos, ...]
                    return patched
                return src_act

            return patch_fn

        if denoise:
            patched_logits = model.run_with_hooks(
                input_ids=corrupted_ids,
                hooks={name: make_patch_fn(source_activation, positions)},
            )
        else:
            patched_logits = model.run_with_hooks(
                input_ids=clean_ids,
                hooks={name: make_patch_fn(source_activation, positions)},
            )

        patched_metric = metric(patched_logits)
        patched_metrics[name] = patched_metric
        effects[name] = patched_metric - baseline_metric

    return PatchingResult(
        effects=effects,
        baseline_metric=baseline_metric,
        patched_metrics=patched_metrics,
        names=expanded_names,
    )


def _make_logit_diff_metric(
    model: TappedModel, answer_tokens: list[str]
) -> Callable[[torch.Tensor], float]:
    """Create a logit difference metric from answer token strings."""
    correct_token = answer_tokens[0]
    incorrect_token = answer_tokens[1]

    correct_id = model.tokenizer.encode(correct_token, add_special_tokens=False)
    incorrect_id = model.tokenizer.encode(incorrect_token, add_special_tokens=False)

    if not correct_id or not incorrect_id:
        raise ValueError(
            f"Could not tokenize answer tokens: {answer_tokens}"
        )

    # Use the first token if the answer tokenizes to multiple tokens
    correct_id = correct_id[0]
    incorrect_id = incorrect_id[0]

    def logit_diff(logits: torch.Tensor) -> float:
        # Take last sequence position
        last_logits = logits[0, -1]
        return (last_logits[correct_id] - last_logits[incorrect_id]).item()

    return logit_diff
