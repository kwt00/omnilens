from __future__ import annotations

import math
from abc import abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from omnilens.core.tapped_model import TappedModel


class SAEResult:
    """Output from an SAE forward pass."""

    def __init__(
        self,
        features: torch.Tensor,
        reconstruction: torch.Tensor,
        loss: torch.Tensor,
        reconstruction_loss: torch.Tensor,
        sparsity_loss: torch.Tensor,
    ) -> None:
        self.features = features
        self.reconstruction = reconstruction
        self.loss = loss
        self.reconstruction_loss = reconstruction_loss
        self.sparsity_loss = sparsity_loss


class BaseSAE(nn.Module):
    """Base class for all SAE variants.

    Subclasses must implement `_compute_features()` which defines
    the sparsity mechanism (ReLU, TopK, Gated, JumpReLU).

    The forward pass is always:
      1. Center input (subtract decoder bias)
      2. Encode → sparse features via _compute_features()
      3. Decode → reconstruction
      4. Compute loss

    Usage as a standalone nn.Module:
        features, reconstruction, loss = sae(activations)

    Usage with TappedModel:
        sae.fit(model, hook_point="layers.6.residual.block_out", dataset=data)
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        tied_weights: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.tied_weights = tied_weights

        self.encoder = nn.Linear(d_model, n_features, bias=True)
        if tied_weights:
            self.decoder = None
        else:
            self.decoder = nn.Linear(n_features, d_model, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(d_model))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming uniform."""
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        if self.decoder is not None:
            nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        # Normalize decoder columns to unit norm
        self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        """Normalize decoder weight columns to unit norm."""
        with torch.no_grad():
            w = self._decoder_weight()
            w.div_(w.norm(dim=0, keepdim=True) + 1e-8)

    def _decoder_weight(self) -> torch.Tensor:
        """Get the decoder weight matrix (handles tied weights)."""
        if self.tied_weights:
            return self.encoder.weight.T
        return self.decoder.weight

    @abstractmethod
    def _compute_features(self, pre_activations: torch.Tensor) -> torch.Tensor:
        """Apply the sparsity mechanism to encoder output.

        Args:
            pre_activations: Raw encoder output, shape (..., n_features).

        Returns:
            Sparse feature activations, same shape.
        """
        ...

    @abstractmethod
    def _sparsity_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Compute the sparsity penalty for this variant."""
        ...

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations into sparse features."""
        centered = x - self.decoder_bias
        pre_activations = self.encoder(centered)
        return self._compute_features(pre_activations)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space."""
        if self.tied_weights:
            return F.linear(features, self.encoder.weight.T) + self.decoder_bias
        return self.decoder(features) + self.decoder_bias

    def forward(
        self, x: torch.Tensor
    ) -> SAEResult:
        """Full forward pass: encode, decode, compute loss.

        Args:
            x: Input activations, shape (..., d_model).

        Returns:
            SAEResult with features, reconstruction, and losses.
        """
        features = self.encode(x)
        reconstruction = self.decode(features)

        reconstruction_loss = F.mse_loss(reconstruction, x)
        sparsity_loss = self._sparsity_loss(features)
        loss = reconstruction_loss + sparsity_loss

        return SAEResult(
            features=features,
            reconstruction=reconstruction,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
        )

    def fit(
        self,
        model: TappedModel,
        hook_point: str,
        dataset,
        lr: float = 3e-4,
        sparsity_coeff: float = 1.0,
        batch_size: int = 32,
        n_steps: int = 10000,
        log_every: int = 100,
        device: str | torch.device | None = None,
    ) -> list[dict]:
        """Train this SAE on activations from a TappedModel.

        Accepts three dataset formats:
          - torch.Tensor: pre-cached activations, shape (n_samples, d_model)
          - list[str]: raw text strings to run through the model
          - HuggingFace Dataset: dataset with a 'text' column

        Args:
            model: TappedModel to extract activations from.
            hook_point: Which hook point to train on.
            dataset: Training data (tensor, list of strings, or HF dataset).
            lr: Learning rate.
            sparsity_coeff: Weight for the sparsity loss term.
            batch_size: Batch size for training.
            n_steps: Number of training steps.
            log_every: Log metrics every N steps.
            device: Device to train on. Defaults to model's device.

        Returns:
            List of log dicts with training metrics.
        """
        if device is None:
            device = model.device

        self.to(device)
        activations = self._collect_activations(model, hook_point, dataset, device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        n_samples = activations.shape[0]
        logs = []

        for step in range(n_steps):
            # Sample a random batch
            indices = torch.randint(0, n_samples, (batch_size,))
            batch = activations[indices]

            result = self(batch)
            loss = result.reconstruction_loss + sparsity_coeff * result.sparsity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Keep decoder norms in check
            self._normalize_decoder()

            if step % log_every == 0:
                n_active = (result.features > 0).float().sum(dim=-1).mean().item()
                log = {
                    "step": step,
                    "loss": loss.item(),
                    "reconstruction_loss": result.reconstruction_loss.item(),
                    "sparsity_loss": result.sparsity_loss.item(),
                    "n_active_features": n_active,
                }
                logs.append(log)
                print(
                    f"Step {step:>5d} | loss: {loss.item():.4f} | "
                    f"recon: {result.reconstruction_loss.item():.4f} | "
                    f"sparsity: {result.sparsity_loss.item():.4f} | "
                    f"active: {n_active:.0f}"
                )

        return logs

    def _collect_activations(
        self,
        model: TappedModel,
        hook_point: str,
        dataset,
        device: torch.device,
    ) -> torch.Tensor:
        """Collect activations from the model for training.

        Handles three input formats:
          - torch.Tensor: already cached, just return
          - list[str]: run through model, cache activations
          - HF Dataset: extract text column, run through model
        """
        # Already a tensor
        if isinstance(dataset, torch.Tensor):
            return dataset.to(device)

        # HuggingFace dataset — extract text column
        if hasattr(dataset, "__getitem__") and hasattr(dataset, "column_names"):
            text_col = None
            for col in ("text", "content", "sentence"):
                if col in dataset.column_names:
                    text_col = col
                    break
            if text_col is None:
                raise ValueError(
                    f"Could not find text column in dataset. "
                    f"Available columns: {dataset.column_names}"
                )
            texts = dataset[text_col]
        elif isinstance(dataset, list):
            texts = dataset
        else:
            raise TypeError(
                f"dataset must be a tensor, list of strings, or HF Dataset. "
                f"Got {type(dataset)}"
            )

        # Run texts through the model and collect activations
        all_activations = []
        for i in range(0, len(texts), 32):
            batch_texts = texts[i : i + 32]
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    text=batch_texts, names=[hook_point]
                )
            acts = cache[hook_point]  # (batch, seq, d_model)
            # Flatten batch and seq dimensions
            all_activations.append(acts.reshape(-1, acts.shape[-1]).to(device))

        return torch.cat(all_activations, dim=0)

    def __repr__(self) -> str:
        variant = type(self).__name__
        tied = ", tied" if self.tied_weights else ""
        return f"{variant}(d_model={self.d_model}, n_features={self.n_features}{tied})"


class SAE(BaseSAE):
    """Vanilla SAE with ReLU activation and L1 sparsity penalty."""

    def __init__(
        self,
        d_model: int,
        n_features: int,
        tied_weights: bool = False,
    ) -> None:
        super().__init__(d_model, n_features, tied_weights)

    def _compute_features(self, pre_activations: torch.Tensor) -> torch.Tensor:
        return F.relu(pre_activations)

    def _sparsity_loss(self, features: torch.Tensor) -> torch.Tensor:
        return features.abs().mean()


class TopKSAE(BaseSAE):
    """TopK SAE — keeps only the top K features active per input.

    Sparsity is structural (enforced by the top-k operation), so no
    L1 penalty is needed. This is the approach used by Anthropic.
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        k: int = 32,
        tied_weights: bool = False,
    ) -> None:
        super().__init__(d_model, n_features, tied_weights)
        self.k = k

    def _compute_features(self, pre_activations: torch.Tensor) -> torch.Tensor:
        topk_values, topk_indices = pre_activations.topk(self.k, dim=-1)
        result = torch.zeros_like(pre_activations)
        result.scatter_(-1, topk_indices, F.relu(topk_values))
        return result

    def _sparsity_loss(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=features.device)

    def __repr__(self) -> str:
        tied = ", tied" if self.tied_weights else ""
        return (
            f"TopKSAE(d_model={self.d_model}, n_features={self.n_features}, "
            f"k={self.k}{tied})"
        )


class GatedSAE(BaseSAE):
    """Gated SAE — uses a learnable gate to decide which features fire.

    The gate is a separate linear layer with a sigmoid activation.
    Features = gate * encoder_output. This decouples the "should this
    feature fire?" decision from the "how strongly?" decision.
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        tied_weights: bool = False,
    ) -> None:
        super().__init__(d_model, n_features, tied_weights)
        self.gate = nn.Linear(d_model, n_features, bias=True)
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate.bias)

    def _compute_features(self, pre_activations: torch.Tensor) -> torch.Tensor:
        centered = pre_activations - self.encoder.bias  # undo bias for gate input
        # Gate input is the original centered activation, not the encoder output
        # We need to recompute from the raw input stored during encode
        gate_values = torch.sigmoid(self.gate(self._last_centered_input))
        return gate_values * F.relu(pre_activations)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.decoder_bias
        self._last_centered_input = centered
        pre_activations = self.encoder(centered)
        return self._compute_features(pre_activations)

    def _sparsity_loss(self, features: torch.Tensor) -> torch.Tensor:
        return features.abs().mean()


class JumpReLUSAE(BaseSAE):
    """JumpReLU SAE — features only fire above a learned threshold.

    Used by Google DeepMind in Gemma Scope. Each feature has its own
    threshold parameter that determines the minimum activation needed
    to fire.
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        initial_threshold: float = 0.001,
        tied_weights: bool = False,
    ) -> None:
        super().__init__(d_model, n_features, tied_weights)
        self.threshold = nn.Parameter(
            torch.full((n_features,), initial_threshold)
        )

    def _compute_features(self, pre_activations: torch.Tensor) -> torch.Tensor:
        # JumpReLU: output = x * (x > threshold), with straight-through gradient
        mask = (pre_activations > self.threshold).float()
        return pre_activations * mask

    def _sparsity_loss(self, features: torch.Tensor) -> torch.Tensor:
        # L0-style: penalize the number of active features
        active = (features > 0).float()
        return active.mean()

    def __repr__(self) -> str:
        tied = ", tied" if self.tied_weights else ""
        return (
            f"JumpReLUSAE(d_model={self.d_model}, n_features={self.n_features}{tied})"
        )
