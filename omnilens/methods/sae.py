from __future__ import annotations

import json
import math
from pathlib import Path
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


# --- Activation/sparsity resolution ---

ACTIVATION_REGISTRY: dict[str, dict] = {
    "relu": {"activation_fn": relu_activation, "sparsity_fn": l1_sparsity},
    "topk": {"activation_fn": None, "sparsity_fn": no_sparsity},  # needs k
    "jumprelu": {"activation_fn": None, "sparsity_fn": l0_sparsity},  # needs threshold
    "gated": {"activation_fn": None, "sparsity_fn": l1_sparsity},  # needs gate setup
}


class SAE(nn.Module):
    """Sparse Autoencoder — composable, flexible, works with any TappedModel hook point.

    Usage:
        # Common variants via string shortcuts
        sae = SAE(d_model=4096, n_features=32768, activation="relu")
        sae = SAE(d_model=4096, n_features=32768, activation="topk", k=64)
        sae = SAE(d_model=4096, n_features=32768, activation="jumprelu")
        sae = SAE(d_model=4096, n_features=32768, activation="gated")

        # Deep encoder/decoder
        sae = SAE(d_model=4096, n_features=32768, activation="topk", k=64, hidden_dims=[8192])

        # Fully custom
        sae = SAE(d_model=4096, n_features=32768, activation=my_fn, sparsity=my_loss)

        # Custom encoder/decoder modules
        sae = SAE(d_model=4096, n_features=32768, encoder=my_encoder, decoder=my_decoder)

        # Train
        sae.fit(model, hook_point="layers.16.residual.block_out", dataset=data)
        sae.fit_on_activations(activations, n_steps=10000)

        # Intervene
        model.run_with_hooks(text="...", hooks={"layers.16.residual.block_out": sae.hook_ablate([42])})
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        activation: str | Callable = "relu",
        sparsity: Callable | None = None,
        hidden_dims: list[int] | None = None,
        tied_weights: bool = False,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        k: int = 32,
        initial_threshold: float = 0.001,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self._activation_name = activation if isinstance(activation, str) else "custom"
        self._k = k
        self._initial_threshold = initial_threshold
        self._hidden_dims = hidden_dims
        self._tied_weights = tied_weights

        # Build or use provided encoder/decoder
        if encoder is not None and decoder is not None:
            self.encoder = encoder
            self.decoder = decoder
        else:
            self.encoder, self.decoder = _build_encoder_decoder(
                d_model, n_features, hidden_dims, tied_weights
            )

        self.pre_bias = nn.Parameter(torch.zeros(d_model))

        # Resolve activation function
        if callable(activation) and not isinstance(activation, str):
            self.activation_fn = activation
            self.sparsity_fn = sparsity if sparsity is not None else l1_sparsity
        elif isinstance(activation, str):
            self._setup_activation(activation, sparsity, n_features, d_model, k, initial_threshold)
        else:
            raise ValueError(f"activation must be a string or callable, got {type(activation)}")

    def _setup_activation(
        self,
        name: str,
        sparsity_override: Callable | None,
        n_features: int,
        d_model: int,
        k: int,
        initial_threshold: float,
    ) -> None:
        """Configure activation and sparsity from a string name."""
        if name not in ACTIVATION_REGISTRY:
            raise ValueError(
                f"Unknown activation '{name}'. "
                f"Options: {list(ACTIVATION_REGISTRY.keys())} or pass a callable."
            )

        defaults = ACTIVATION_REGISTRY[name]
        self.sparsity_fn = sparsity_override if sparsity_override is not None else defaults["sparsity_fn"]

        if name == "relu":
            self.activation_fn = relu_activation
        elif name == "topk":
            self.activation_fn = topk_activation(k)
        elif name == "jumprelu":
            self.threshold = nn.Parameter(torch.full((n_features,), initial_threshold))
            self.activation_fn = jumprelu_activation(self.threshold)
        elif name == "gated":
            self.gate = nn.Linear(d_model, n_features, bias=True)
            nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
            nn.init.zeros_(self.gate.bias)

            def gated_activation(pre_activations: torch.Tensor) -> torch.Tensor:
                gate_values = torch.sigmoid(self.gate(self._last_centered))
                return gate_values * F.relu(pre_activations)

            self.activation_fn = gated_activation

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations into sparse features."""
        centered = x - self.pre_bias
        self._last_centered = centered
        pre_activations = self.encoder(centered)
        return self.activation_fn(pre_activations)

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

    # --- Hook helpers ---

    def hook_ablate(self, features: list[int]) -> Callable:
        """Return a hook function that zeros out specific features."""
        def hook_fn(activation: torch.Tensor, hook_name: str) -> torch.Tensor:
            f = self.encode(activation)
            f[..., features] = 0
            return self.decode(f)
        return hook_fn

    def hook_amplify(self, feature: int, scale: float = 2.0) -> Callable:
        """Return a hook function that amplifies a specific feature."""
        def hook_fn(activation: torch.Tensor, hook_name: str) -> torch.Tensor:
            f = self.encode(activation)
            f[..., feature] *= scale
            return self.decode(f)
        return hook_fn

    def hook_clamp(self, feature: int, value: float) -> Callable:
        """Return a hook function that clamps a feature to a fixed value."""
        def hook_fn(activation: torch.Tensor, hook_name: str) -> torch.Tensor:
            f = self.encode(activation)
            f[..., feature] = value
            return self.decode(f)
        return hook_fn

    def hook_reconstruct(self) -> Callable:
        """Return a hook function that replaces activations with SAE reconstruction."""
        def hook_fn(activation: torch.Tensor, hook_name: str) -> torch.Tensor:
            return self.decode(self.encode(activation))
        return hook_fn

    # --- Training ---

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
        """Train on activations extracted from a TappedModel.

        Dataset formats: torch.Tensor, list[str], or HuggingFace Dataset.
        """
        if device is None:
            device = model.device
        self.to(device)
        activations = _collect_activations(model, hook_point, dataset, device)
        return self._train_loop(activations, lr, sparsity_coeff, batch_size, n_steps, log_every)

    def fit_on_activations(
        self,
        activations: torch.Tensor,
        lr: float = 3e-4,
        sparsity_coeff: float = 1.0,
        batch_size: int = 32,
        n_steps: int = 10000,
        log_every: int = 100,
    ) -> list[dict]:
        """Train directly on pre-cached activation tensors. No model needed."""
        device = activations.device
        self.to(device)
        return self._train_loop(activations, lr, sparsity_coeff, batch_size, n_steps, log_every)

    def _train_loop(
        self,
        activations: torch.Tensor,
        lr: float,
        sparsity_coeff: float,
        batch_size: int,
        n_steps: int,
        log_every: int,
    ) -> list[dict]:
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

    # --- Save/Load with config ---

    def save(self, path: str | Path) -> None:
        """Save SAE weights and config to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), path / "weights.pt")

        config = {
            "d_model": self.d_model,
            "n_features": self.n_features,
            "activation": self._activation_name,
            "k": self._k,
            "initial_threshold": self._initial_threshold,
            "hidden_dims": self._hidden_dims,
            "tied_weights": self._tied_weights,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> SAE:
        """Load an SAE from a directory saved with .save()."""
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)

        sae = cls(
            d_model=config["d_model"],
            n_features=config["n_features"],
            activation=config["activation"],
            k=config.get("k", 32),
            initial_threshold=config.get("initial_threshold", 0.001),
            hidden_dims=config.get("hidden_dims"),
            tied_weights=config.get("tied_weights", False),
        )
        sae.load_state_dict(torch.load(path / "weights.pt", weights_only=True))
        return sae

    def __repr__(self) -> str:
        parts = [f"d_model={self.d_model}", f"n_features={self.n_features}"]
        parts.append(f"activation='{self._activation_name}'")
        if self._activation_name == "topk":
            parts.append(f"k={self._k}")
        if self._hidden_dims:
            parts.append(f"hidden_dims={self._hidden_dims}")
        if self._tied_weights:
            parts.append("tied_weights=True")
        return f"SAE({', '.join(parts)})"


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

    encoder_layers = []
    dims = [d_model] + hidden_dims + [n_features]
    for i in range(len(dims) - 1):
        encoder_layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
        if i < len(dims) - 2:
            encoder_layers.append(nn.ReLU())
    encoder = nn.Sequential(*encoder_layers)

    decoder_layers = []
    rev_dims = list(reversed(dims))
    for i in range(len(rev_dims) - 1):
        bias = i < len(rev_dims) - 2
        decoder_layers.append(nn.Linear(rev_dims[i], rev_dims[i + 1], bias=bias))
        if i < len(rev_dims) - 2:
            decoder_layers.append(nn.ReLU())
    decoder = nn.Sequential(*decoder_layers)

    return encoder, decoder


def _collect_activations(
    model: TappedModel,
    hook_point: str,
    dataset,
    device: torch.device,
) -> torch.Tensor:
    """Collect activations from the model for training."""
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
