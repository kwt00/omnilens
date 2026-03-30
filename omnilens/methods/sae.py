from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

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


# --- Built-in activation functions ---

def relu_activation(pre_activations: torch.Tensor) -> torch.Tensor:
    return F.relu(pre_activations)


def topk_activation(k: int) -> Callable[[torch.Tensor], torch.Tensor]:
    def activation(pre_activations: torch.Tensor) -> torch.Tensor:
        topk_values, topk_indices = pre_activations.topk(k, dim=-1)
        result = torch.zeros_like(pre_activations)
        result.scatter_(-1, topk_indices, F.relu(topk_values))
        return result
    return activation


def jumprelu_activation(threshold: nn.Parameter) -> Callable[[torch.Tensor], torch.Tensor]:
    def activation(pre_activations: torch.Tensor) -> torch.Tensor:
        mask = (pre_activations > threshold).float()
        return pre_activations * mask
    return activation


# --- Built-in sparsity losses ---

def l1_sparsity(features: torch.Tensor) -> torch.Tensor:
    return features.abs().mean()


def l0_sparsity(features: torch.Tensor) -> torch.Tensor:
    return (features > 0).float().mean()


def no_sparsity(features: torch.Tensor) -> torch.Tensor:
    return torch.tensor(0.0, device=features.device)


class SAE(nn.Module):
    """Sparse Autoencoder — composable from any encoder, decoder, activation, and sparsity.

    The SAE is defined by four components:
      - encoder: nn.Module mapping activations to pre-features
      - decoder: nn.Module mapping features back to activation space
      - activation_fn: callable applying sparsity to encoder output
      - sparsity_fn: callable computing the sparsity penalty

    Use directly with custom components, or use factory methods for common variants:
      SAE.relu(d_model, n_features)
      SAE.topk(d_model, n_features, k=32)
      SAE.jumprelu(d_model, n_features)
      SAE.gated(d_model, n_features)

    Usage:
        sae = SAE.topk(d_model=4096, n_features=32768, k=64)
        sae.fit(model, hook_point="layers.16.residual.block_out", dataset=data)
        features = sae.encode(activations)
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = relu_activation,
        sparsity_fn: Callable[[torch.Tensor], torch.Tensor] = l1_sparsity,
        pre_bias: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.activation_fn = activation_fn
        self.sparsity_fn = sparsity_fn
        self.pre_bias = pre_bias if pre_bias is not None else nn.Parameter(
            torch.zeros(self._detect_input_dim())
        )

    def _detect_input_dim(self) -> int:
        """Infer input dimension from the decoder's output."""
        for module in reversed(list(self.decoder.modules())):
            if hasattr(module, "out_features"):
                return module.out_features
            if hasattr(module, "weight") and module.weight.ndim == 2:
                return module.weight.shape[0]
        # Fallback: try encoder input
        for module in self.encoder.modules():
            if hasattr(module, "in_features"):
                return module.in_features
            if hasattr(module, "weight") and module.weight.ndim == 2:
                return module.weight.shape[1]
        return 0

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space."""
        return self.decoder(features) + self.pre_bias

    def forward(self, x: torch.Tensor) -> SAEResult:
        """Full forward pass: encode, decode, compute loss."""
        features = self.encode(x)
        reconstruction = self.decode(features)

        reconstruction_loss = F.mse_loss(reconstruction, x)
        sparsity_loss = self.sparsity_fn(features)
        loss = reconstruction_loss + sparsity_loss

        return SAEResult(
            features=features,
            reconstruction=reconstruction,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
        )

    # --- Factory methods for common variants ---

    @staticmethod
    def _build_encoder_decoder(
        d_model: int,
        n_features: int,
        hidden_dims: list[int] | None = None,
        tied_weights: bool = False,
    ) -> tuple[nn.Module, nn.Module]:
        """Build encoder and decoder, optionally with hidden layers."""
        if hidden_dims is None:
            encoder = nn.Linear(d_model, n_features, bias=True)
            if tied_weights:
                decoder = TiedDecoder(encoder)
            else:
                decoder = nn.Linear(n_features, d_model, bias=False)
            return encoder, decoder

        # Deep encoder: d_model → h1 → h2 → ... → n_features
        encoder_layers = []
        dims = [d_model] + hidden_dims + [n_features]
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            if i < len(dims) - 2:  # ReLU between hidden layers, not after last
                encoder_layers.append(nn.ReLU())
        encoder = nn.Sequential(*encoder_layers)

        # Deep decoder: n_features → ... → h1 → d_model (reversed)
        decoder_layers = []
        rev_dims = list(reversed(dims))
        for i in range(len(rev_dims) - 1):
            bias = i < len(rev_dims) - 2  # no bias on final layer
            decoder_layers.append(nn.Linear(rev_dims[i], rev_dims[i + 1], bias=bias))
            if i < len(rev_dims) - 2:
                decoder_layers.append(nn.ReLU())
        decoder = nn.Sequential(*decoder_layers)

        return encoder, decoder

    @classmethod
    def relu(
        cls,
        d_model: int,
        n_features: int,
        hidden_dims: list[int] | None = None,
        tied_weights: bool = False,
    ) -> SAE:
        """Vanilla SAE with ReLU activation and L1 sparsity."""
        encoder, decoder = cls._build_encoder_decoder(
            d_model, n_features, hidden_dims, tied_weights
        )
        return cls(
            encoder=encoder,
            decoder=decoder,
            activation_fn=relu_activation,
            sparsity_fn=l1_sparsity,
            pre_bias=nn.Parameter(torch.zeros(d_model)),
        )

    @classmethod
    def topk(
        cls,
        d_model: int,
        n_features: int,
        k: int = 32,
        hidden_dims: list[int] | None = None,
        tied_weights: bool = False,
    ) -> SAE:
        """TopK SAE — structural sparsity, no L1 penalty."""
        encoder, decoder = cls._build_encoder_decoder(
            d_model, n_features, hidden_dims, tied_weights
        )
        return cls(
            encoder=encoder,
            decoder=decoder,
            activation_fn=topk_activation(k),
            sparsity_fn=no_sparsity,
            pre_bias=nn.Parameter(torch.zeros(d_model)),
        )

    @classmethod
    def jumprelu(
        cls,
        d_model: int,
        n_features: int,
        initial_threshold: float = 0.001,
        hidden_dims: list[int] | None = None,
        tied_weights: bool = False,
    ) -> SAE:
        """JumpReLU SAE — learnable per-feature threshold."""
        encoder, decoder = cls._build_encoder_decoder(
            d_model, n_features, hidden_dims, tied_weights
        )
        threshold = nn.Parameter(torch.full((n_features,), initial_threshold))
        sae = cls(
            encoder=encoder,
            decoder=decoder,
            activation_fn=jumprelu_activation(threshold),
            sparsity_fn=l0_sparsity,
            pre_bias=nn.Parameter(torch.zeros(d_model)),
        )
        sae.threshold = threshold
        return sae

    @classmethod
    def gated(
        cls,
        d_model: int,
        n_features: int,
        hidden_dims: list[int] | None = None,
        tied_weights: bool = False,
    ) -> SAE:
        """Gated SAE — learnable gate decides which features fire."""
        encoder, decoder = cls._build_encoder_decoder(
            d_model, n_features, hidden_dims, tied_weights
        )
        gate = nn.Linear(d_model, n_features, bias=True)
        nn.init.kaiming_uniform_(gate.weight, a=math.sqrt(5))
        nn.init.zeros_(gate.bias)

        sae = cls(
            encoder=encoder,
            decoder=decoder,
            activation_fn=relu_activation,  # placeholder, overridden below
            sparsity_fn=l1_sparsity,
            pre_bias=nn.Parameter(torch.zeros(d_model)),
        )
        sae.gate = gate

        def gated_activation(pre_activations: torch.Tensor) -> torch.Tensor:
            centered = sae._last_centered
            gate_values = torch.sigmoid(gate(centered))
            return gate_values * F.relu(pre_activations)

        sae.activation_fn = gated_activation
        return sae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations into sparse features."""
        centered = x - self.pre_bias
        self._last_centered = centered  # stored for gated variant
        pre_activations = self.encoder(centered)
        return self.activation_fn(pre_activations)

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

        Returns:
            List of log dicts with training metrics.
        """
        if device is None:
            device = model.device

        self.to(device)
        activations = _collect_activations(model, hook_point, dataset, device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        n_samples = activations.shape[0]
        logs = []

        for step in range(n_steps):
            indices = torch.randint(0, n_samples, (batch_size,))
            batch = activations[indices]

            result = self(batch)
            loss = result.reconstruction_loss + sparsity_coeff * result.sparsity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

    def __repr__(self) -> str:
        return (
            f"SAE(\n"
            f"  encoder={self.encoder},\n"
            f"  decoder={self.decoder},\n"
            f"  activation_fn={self.activation_fn},\n"
            f"  sparsity_fn={self.sparsity_fn}\n"
            f")"
        )


class TiedDecoder(nn.Module):
    """Decoder that shares weights with the encoder (transposed)."""

    def __init__(self, encoder: nn.Linear) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.encoder.weight.T)

    @property
    def out_features(self) -> int:
        return self.encoder.in_features

    def __repr__(self) -> str:
        return f"TiedDecoder(features={self.encoder.out_features} -> {self.encoder.in_features})"


def _collect_activations(
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
    if isinstance(dataset, torch.Tensor):
        return dataset.to(device)

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

    all_activations = []
    for i in range(0, len(texts), 32):
        batch_texts = texts[i : i + 32]
        with torch.no_grad():
            _, cache = model.run_with_cache(
                text=batch_texts, names=[hook_point]
            )
        acts = cache[hook_point]
        all_activations.append(acts.reshape(-1, acts.shape[-1]).to(device))

    return torch.cat(all_activations, dim=0)
