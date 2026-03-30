from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch

from omnilens.methods.sae import _collect_activations

if TYPE_CHECKING:
    from omnilens.core.tapped_model import TappedModel
    from omnilens.methods.probe import Probe
    from omnilens.methods.sae import SAE


class SteeringVector:
    """A direction in activation space for steering model behavior.

    A steering vector is a tensor that gets added to (or subtracted from)
    the residual stream during inference to shift model behavior.

    Can be created from:
      - Contrastive text pairs (mean diff)
      - A single positive/negative pair
      - A trained probe's weight vector
      - An SAE feature's decoder column
      - A raw tensor

    Usage:
        vec = SteeringVector.from_contrastive(model, hook_point, positive, negative)
        logits = model.run_with_hooks(text="...", hooks=vec.hook(scale=2.0))
    """

    def __init__(
        self,
        direction: torch.Tensor,
        hook_point: str,
    ) -> None:
        self.direction = direction
        self.hook_point = hook_point

    # --- Constructors ---

    @classmethod
    def from_contrastive(
        cls,
        model: TappedModel,
        hook_point: str,
        positive: list[str],
        negative: list[str],
        position: int = -1,
    ) -> SteeringVector:
        """Compute steering vector from contrastive text pairs.

        direction = mean(positive_activations) - mean(negative_activations)

        Args:
            model: TappedModel to extract activations from.
            hook_point: Which hook point to extract from.
            positive: Texts representing the positive direction.
            negative: Texts representing the negative direction.
            position: Token position to use. -1 for last token.
        """
        pos_acts = _collect_mean_activation(model, hook_point, positive, position)
        neg_acts = _collect_mean_activation(model, hook_point, negative, position)
        direction = pos_acts - neg_acts
        return cls(direction=direction, hook_point=hook_point)

    @classmethod
    def from_pair(
        cls,
        model: TappedModel,
        hook_point: str,
        positive: str,
        negative: str,
        position: int = -1,
    ) -> SteeringVector:
        """Compute steering vector from a single positive/negative pair."""
        return cls.from_contrastive(
            model, hook_point, [positive], [negative], position
        )

    @classmethod
    def from_probe(
        cls,
        probe: Probe,
        hook_point: str,
        class_idx: int = 1,
    ) -> SteeringVector:
        """Extract steering vector from a trained linear probe.

        The probe's weight row for the target class IS the direction
        in activation space that the probe associates with that class.

        Args:
            probe: A trained Probe instance.
            hook_point: Hook point the probe was trained on.
            class_idx: Which class direction to extract.
        """
        # Get the weight matrix from the probe
        if hasattr(probe.network, "weight"):
            # Linear probe
            weight = probe.network.weight.detach()
        else:
            # MLP probe — use the last linear layer
            for module in reversed(list(probe.network.modules())):
                if hasattr(module, "weight") and module.weight.ndim == 2:
                    weight = module.weight.detach()
                    break
            else:
                raise ValueError("Could not find weight matrix in probe network.")

        direction = weight[class_idx]
        return cls(direction=direction, hook_point=hook_point)

    @classmethod
    def from_sae_feature(
        cls,
        sae: SAE,
        feature: int,
        hook_point: str,
    ) -> SteeringVector:
        """Extract steering vector from an SAE feature's decoder column.

        The decoder column for a feature IS the direction that feature
        represents in activation space.

        Args:
            sae: A trained SAE instance.
            feature: Feature index to extract.
            hook_point: Hook point the SAE was trained on.
        """
        # Get decoder weight
        decoder = sae.decoder
        if hasattr(decoder, "weight"):
            # nn.Linear: weight is (d_output, n_features), we want column `feature`
            direction = decoder.weight[:, feature].detach()
        elif hasattr(decoder, "encoder"):
            # TiedDecoder: transpose of encoder weight
            direction = decoder.encoder.weight[feature].detach()
        else:
            # Sequential decoder — find last linear layer
            for module in reversed(list(decoder.modules())):
                if hasattr(module, "weight") and module.weight.ndim == 2:
                    direction = module.weight[:, feature].detach()
                    break
            else:
                raise ValueError("Could not find weight matrix in SAE decoder.")

        return cls(direction=direction, hook_point=hook_point)

    # --- Application ---

    def hook(self, scale: float = 1.0) -> dict[str, Callable]:
        """Return a hooks dict that adds this vector to the hook point.

        Args:
            scale: Multiplier for the steering vector. Positive adds,
                negative subtracts.

        Returns:
            Dict suitable for model.run_with_hooks(hooks=...).
        """
        direction = self.direction
        def hook_fn(activation: torch.Tensor, hook_name: str) -> torch.Tensor:
            return activation + scale * direction.to(activation.device)
        return {self.hook_point: hook_fn}

    def hooks(
        self,
        layers: list[int],
        scale: float = 1.0,
        hook_template: str | None = None,
    ) -> dict[str, Callable]:
        """Return hooks dict that applies this vector across multiple layers.

        Args:
            layers: Layer indices to apply the vector to.
            scale: Multiplier for the steering vector.
            hook_template: Template with {i} for layer index.
                Defaults to replacing the layer number in self.hook_point.

        Returns:
            Dict suitable for model.run_with_hooks(hooks=...).
        """
        if hook_template is None:
            # Infer template from hook_point by replacing the layer number
            import re
            hook_template = re.sub(
                r"layers\.(\d+)\.",
                r"layers.{i}.",
                self.hook_point,
            )

        direction = self.direction
        result = {}
        for layer in layers:
            point = hook_template.replace("{i}", str(layer))
            def make_hook(pt: str):
                def hook_fn(activation: torch.Tensor, hook_name: str) -> torch.Tensor:
                    return activation + scale * direction.to(activation.device)
                return hook_fn
            result[point] = make_hook(point)
        return result

    # --- Utilities ---

    def normalize(self) -> SteeringVector:
        """Return a unit-normalized copy of this vector."""
        return SteeringVector(
            direction=self.direction / (self.direction.norm() + 1e-8),
            hook_point=self.hook_point,
        )

    def cosine_similarity(self, other: SteeringVector) -> float:
        """Compute cosine similarity with another steering vector."""
        a = self.direction.float()
        b = other.direction.float()
        return (a @ b / (a.norm() * b.norm() + 1e-8)).item()

    # --- Save/Load ---

    def save(self, path: str | Path) -> None:
        """Save steering vector and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.direction, path / "direction.pt")
        config = {"hook_point": self.hook_point}
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> SteeringVector:
        """Load a steering vector from a directory."""
        path = Path(path)
        direction = torch.load(path / "direction.pt", weights_only=True)
        with open(path / "config.json") as f:
            config = json.load(f)
        return cls(direction=direction, hook_point=config["hook_point"])

    def __repr__(self) -> str:
        norm = self.direction.norm().item()
        dims = self.direction.shape[0]
        return f"SteeringVector(dims={dims}, norm={norm:.3f}, hook_point='{self.hook_point}')"


def _collect_mean_activation(
    model: TappedModel,
    hook_point: str,
    texts: list[str],
    position: int,
) -> torch.Tensor:
    """Collect activations and return the mean across all texts."""
    all_acts = []
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        with torch.no_grad():
            _, cache = model.run_with_cache(text=batch, names=[hook_point])
        acts = cache[hook_point]  # (batch, seq, d_model)
        all_acts.append(acts[:, position, :])  # (batch, d_model)
    return torch.cat(all_acts, dim=0).mean(dim=0)
