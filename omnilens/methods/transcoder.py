from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnilens.methods.sae import (
    ACTIVATION_REGISTRY,
    TiedDecoder,
    _build_encoder_decoder,
    _collect_activations,
    jumprelu_activation,
    l0_sparsity,
    l1_sparsity,
    no_sparsity,
    relu_activation,
    topk_activation,
)

if TYPE_CHECKING:
    from omnilens.core.tapped_model import TappedModel


class TranscoderResult:
    """Output from a transcoder forward pass."""

    def __init__(
        self,
        features: torch.Tensor,
        prediction: torch.Tensor,
        loss: torch.Tensor,
        prediction_loss: torch.Tensor,
        sparsity_loss: torch.Tensor,
    ) -> None:
        self.features = features
        self.prediction = prediction
        self.loss = loss
        self.prediction_loss = prediction_loss
        self.sparsity_loss = sparsity_loss


class Transcoder(nn.Module):
    """Transcoder — sparse replacement for MLP layers.

    Unlike an SAE (which reconstructs its own input), a transcoder
    encodes MLP input and predicts MLP output through a sparse bottleneck.

    Supports single-layer, cross-layer (CLT), and skip connection variants.

    Usage:
        tc = Transcoder(d_input=4096, d_output=4096, n_features=32768, activation="topk", k=64)
        tc.fit(model, input_point="layers.16.mlp.layer_norm", output_point="layers.16.mlp.down_proj", dataset=data)

        with tc.attached(model, layer=16):
            logits = model(tokens)  # layer 16 MLP is replaced by transcoder
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        n_features: int,
        activation: str | Callable = "relu",
        sparsity: Callable | None = None,
        hidden_dims: list[int] | None = None,
        tied_weights: bool = False,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        k: int = 32,
        initial_threshold: float = 0.001,
        skip: bool = False,
        output_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.n_features = n_features
        self._activation_name = activation if isinstance(activation, str) else "custom"
        self._k = k
        self._initial_threshold = initial_threshold
        self._hidden_dims = hidden_dims
        self._tied_weights = tied_weights
        self._skip = skip
        self._output_layers = output_layers

        # Build or use provided encoder
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder, _ = _build_encoder_decoder(
                d_input, n_features, hidden_dims, tied_weights=False
            )

        # Build decoder(s)
        if decoder is not None:
            self.decoder = decoder
            self.multi_decoder = None
        elif output_layers is not None and len(output_layers) > 1:
            # Cross-layer: one decoder per output layer
            self.decoder = None
            self.multi_decoder = nn.ModuleDict({
                str(layer): nn.Linear(n_features, d_output, bias=False)
                for layer in output_layers
            })
        else:
            _, self.decoder = _build_encoder_decoder(
                d_output, n_features, hidden_dims=None, tied_weights=False
            )
            # Swap dims: decoder goes n_features -> d_output
            self.decoder = nn.Linear(n_features, d_output, bias=False)
            self.multi_decoder = None

        self.encoder_bias = nn.Parameter(torch.zeros(d_input))

        # Skip connection
        if skip:
            self.skip_linear = nn.Linear(d_input, d_output, bias=True)
        else:
            self.skip_linear = None

        # Resolve activation
        if callable(activation) and not isinstance(activation, str):
            self.activation_fn = activation
            self.sparsity_fn = sparsity if sparsity is not None else l1_sparsity
        elif isinstance(activation, str):
            self._setup_activation(activation, sparsity, n_features, d_input, k, initial_threshold)
        else:
            raise ValueError(f"activation must be a string or callable, got {type(activation)}")

        # State
        self.last_features: torch.Tensor | None = None
        self._ablated_features: list[int] = []
        self._original_forwards: dict[int, Callable] = {}
        self._hook_handles: list = []

    def _setup_activation(
        self,
        name: str,
        sparsity_override: Callable | None,
        n_features: int,
        d_input: int,
        k: int,
        initial_threshold: float,
    ) -> None:
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
            self.gate = nn.Linear(d_input, n_features, bias=True)
            nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
            nn.init.zeros_(self.gate.bias)

            def gated_activation(pre_activations: torch.Tensor) -> torch.Tensor:
                gate_values = torch.sigmoid(self.gate(self._last_centered))
                return gate_values * F.relu(pre_activations)

            self.activation_fn = gated_activation

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode MLP input into sparse features."""
        centered = x - self.encoder_bias
        self._last_centered = centered
        pre_activations = self.encoder(centered)
        features = self.activation_fn(pre_activations)

        # Apply ablations if any
        if self._ablated_features:
            features[..., self._ablated_features] = 0

        self.last_features = features.detach()
        return features

    def decode(self, features: torch.Tensor, layer: int | None = None) -> torch.Tensor:
        """Decode features into predicted MLP output."""
        if self.multi_decoder is not None:
            if layer is None:
                raise ValueError("Must specify layer for cross-layer transcoder.")
            return self.multi_decoder[str(layer)](features)

        return self.decoder(features)

    def forward(
        self, mlp_input: torch.Tensor, mlp_output: torch.Tensor | None = None
    ) -> TranscoderResult:
        """Forward pass: encode input, predict output, compute loss.

        Args:
            mlp_input: Input activations (pre-MLP).
            mlp_output: Target MLP output (for training). If None, loss is zero.
        """
        features = self.encode(mlp_input)
        prediction = self.decode(features)

        if self.skip_linear is not None:
            prediction = prediction + self.skip_linear(mlp_input)

        if mlp_output is not None:
            prediction_loss = F.mse_loss(prediction, mlp_output)
        else:
            prediction_loss = torch.tensor(0.0, device=mlp_input.device)

        sparsity_loss = self.sparsity_fn(features)
        loss = prediction_loss + sparsity_loss

        return TranscoderResult(
            features=features,
            prediction=prediction,
            loss=loss,
            prediction_loss=prediction_loss,
            sparsity_loss=sparsity_loss,
        )

    # --- Attach/Detach ---

    def attach(self, model: TappedModel, layer: int) -> None:
        """Replace a layer's MLP with this transcoder.

        The original MLP forward is saved and restored on detach().
        """
        registry = model._registry
        ln_key = f"layers.{layer}.mlp.layer_norm"
        if ln_key not in registry:
            raise KeyError(f"Cannot find MLP layer norm for layer {layer} in registry.")

        ln_path = registry[ln_key]
        block_path = ".".join(ln_path.split(".")[:-1])
        ln_attr = ln_path.split(".")[-1]

        # Find the MLP module name from registry
        for suffix in ("up_proj", "gate_proj", "down_proj"):
            key = f"layers.{layer}.mlp.{suffix}"
            if key in registry:
                mlp_path = ".".join(registry[key].split(".")[:-1])
                mlp_attr = mlp_path.split(".")[-1]
                break
        else:
            raise KeyError(f"Cannot find MLP module for layer {layer} in registry.")

        block = model._get_module(block_path)
        mlp_module = getattr(block, mlp_attr)

        # Save original forward
        self._original_forwards[layer] = mlp_module.forward

        # Replace MLP forward
        transcoder = self

        def transcoder_forward(hidden_states, *args, **kwargs):
            features = transcoder.encode(hidden_states)
            output = transcoder.decode(features)
            if transcoder.skip_linear is not None:
                output = output + transcoder.skip_linear(hidden_states)
            return output

        mlp_module.forward = transcoder_forward

    def detach(self, model: TappedModel, layer: int) -> None:
        """Restore the original MLP forward method."""
        if layer not in self._original_forwards:
            return

        registry = model._registry
        for suffix in ("up_proj", "gate_proj", "down_proj"):
            key = f"layers.{layer}.mlp.{suffix}"
            if key in registry:
                ln_key = f"layers.{layer}.mlp.layer_norm"
                ln_path = registry[ln_key]
                block_path = ".".join(ln_path.split(".")[:-1])
                mlp_path = ".".join(registry[key].split(".")[:-1])
                mlp_attr = mlp_path.split(".")[-1]
                break

        block = model._get_module(block_path)
        mlp_module = getattr(block, mlp_attr)
        mlp_module.forward = self._original_forwards.pop(layer)

    @contextmanager
    def attached(self, model: TappedModel, layer: int):
        """Context manager: attach on enter, detach on exit."""
        self.attach(model, layer)
        try:
            yield self
        finally:
            self.detach(model, layer)

    # --- Feature control ---

    def ablate_features(self, features: list[int]) -> None:
        """Set features to be zeroed during encoding."""
        self._ablated_features = features

    def restore_features(self) -> None:
        """Remove all feature ablations."""
        self._ablated_features = []

    # --- Training ---

    def fit(
        self,
        model: TappedModel,
        input_point: str,
        output_point: str,
        dataset,
        lr: float = 3e-4,
        sparsity_coeff: float = 1.0,
        batch_size: int = 32,
        n_steps: int = 10000,
        log_every: int = 100,
        device: str | torch.device | None = None,
    ) -> list[dict]:
        """Train on (MLP input, MLP output) pairs from a TappedModel.

        Args:
            model: TappedModel to extract activations from.
            input_point: Hook point for MLP input (e.g. "layers.16.mlp.layer_norm").
            output_point: Hook point for MLP output (e.g. "layers.16.mlp.down_proj").
            dataset: Training data (tensor, list of strings, or HF dataset).
        """
        if device is None:
            device = model.device
        self.to(device)

        inputs = _collect_activations(model, input_point, dataset, device)
        outputs = _collect_activations(model, output_point, dataset, device)

        return self._train_loop(inputs, outputs, lr, sparsity_coeff, batch_size, n_steps, log_every)

    def fit_on_activations(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        lr: float = 3e-4,
        sparsity_coeff: float = 1.0,
        batch_size: int = 32,
        n_steps: int = 10000,
        log_every: int = 100,
    ) -> list[dict]:
        """Train directly on pre-cached (input, output) activation pairs."""
        device = inputs.device
        self.to(device)
        return self._train_loop(inputs, outputs, lr, sparsity_coeff, batch_size, n_steps, log_every)

    def _train_loop(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        lr: float,
        sparsity_coeff: float,
        batch_size: int,
        n_steps: int,
        log_every: int,
    ) -> list[dict]:
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        n_samples = inputs.shape[0]
        logs = []

        for step in range(n_steps):
            indices = torch.randint(0, n_samples, (batch_size,))
            batch_in = inputs[indices]
            batch_out = outputs[indices]

            result = self(batch_in, batch_out)
            loss = result.prediction_loss + sparsity_coeff * result.sparsity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                n_active = (result.features > 0).float().sum(dim=-1).mean().item()
                log = {
                    "step": step,
                    "loss": loss.item(),
                    "prediction_loss": result.prediction_loss.item(),
                    "sparsity_loss": result.sparsity_loss.item(),
                    "n_active_features": n_active,
                }
                logs.append(log)
                print(
                    f"Step {step:>5d} | loss: {loss.item():.4f} | "
                    f"pred: {result.prediction_loss.item():.4f} | "
                    f"sparsity: {result.sparsity_loss.item():.4f} | "
                    f"active: {n_active:.0f}"
                )

        return logs

    # --- Save/Load ---

    def save(self, path: str | Path) -> None:
        """Save transcoder weights and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), path / "weights.pt")

        config = {
            "d_input": self.d_input,
            "d_output": self.d_output,
            "n_features": self.n_features,
            "activation": self._activation_name,
            "k": self._k,
            "initial_threshold": self._initial_threshold,
            "hidden_dims": self._hidden_dims,
            "tied_weights": self._tied_weights,
            "skip": self._skip,
            "output_layers": self._output_layers,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Transcoder:
        """Load a transcoder from a directory saved with .save()."""
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)

        tc = cls(
            d_input=config["d_input"],
            d_output=config["d_output"],
            n_features=config["n_features"],
            activation=config["activation"],
            k=config.get("k", 32),
            initial_threshold=config.get("initial_threshold", 0.001),
            hidden_dims=config.get("hidden_dims"),
            tied_weights=config.get("tied_weights", False),
            skip=config.get("skip", False),
            output_layers=config.get("output_layers"),
        )
        tc.load_state_dict(torch.load(path / "weights.pt", weights_only=True))
        return tc

    def __repr__(self) -> str:
        parts = [f"d_input={self.d_input}", f"d_output={self.d_output}", f"n_features={self.n_features}"]
        parts.append(f"activation='{self._activation_name}'")
        if self._activation_name == "topk":
            parts.append(f"k={self._k}")
        if self._hidden_dims:
            parts.append(f"hidden_dims={self._hidden_dims}")
        if self._skip:
            parts.append("skip=True")
        if self._output_layers:
            parts.append(f"output_layers={self._output_layers}")
        return f"Transcoder({', '.join(parts)})"
