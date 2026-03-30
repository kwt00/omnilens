from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnilens.methods.sae import _collect_activations

if TYPE_CHECKING:
    from omnilens.core.tapped_model import TappedModel


class ProbeResult:
    """Results from a probe sweep across layers."""

    def __init__(
        self,
        accuracies: dict[str, float] | None = None,
        losses: dict[str, float] | None = None,
        probes: dict[str, Probe] | None = None,
    ) -> None:
        self.accuracies = accuracies or {}
        self.losses = losses or {}
        self.probes = probes or {}

    def __repr__(self) -> str:
        n = len(self.accuracies or self.losses)
        return f"ProbeResult({n} hook points)"


class Probe(nn.Module):
    """Linear or MLP probe for reading information from model activations.

    Trains a small classifier (or regressor) on cached activations to
    determine what information is encoded at specific hook points.

    Usage:
        # Classification: does layer 10 encode sentiment?
        probe = Probe(d_model=4096, n_classes=2)
        probe.fit(model, hook_point="layers.10.residual.block_out", texts=texts, labels=labels)
        print(probe.accuracy)

        # Regression
        probe = Probe(d_model=4096, task="regression")
        probe.fit(model, hook_point="layers.10.residual.block_out", texts=texts, labels=scores)

        # Layer sweep: where does sentiment emerge?
        results = Probe.sweep(model, hook_points="layers.{i}.residual.block_out",
                              texts=texts, labels=labels)
    """

    def __init__(
        self,
        d_model: int,
        n_classes: int = 1,
        task: str | None = None,
        hidden_dims: list[int] | None = None,
        loss_fn: Callable | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self._hidden_dims = hidden_dims

        # Infer task
        if task is not None:
            self.task = task
        elif n_classes <= 1:
            self.task = "regression"
        else:
            self.task = "classification"

        # Build network
        if hidden_dims is None:
            self.network = nn.Linear(d_model, max(n_classes, 1))
        else:
            layers = []
            dims = [d_model] + hidden_dims + [max(n_classes, 1)]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            self.network = nn.Sequential(*layers)

        # Loss function
        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif self.task == "regression":
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # Metrics populated after training
        self.accuracy: float | None = None
        self.train_loss: float | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def fit(
        self,
        model: TappedModel,
        hook_point: str,
        texts: list[str],
        labels: list | torch.Tensor,
        position: int | str = -1,
        lr: float = 1e-3,
        n_epochs: int = 50,
        batch_size: int = 32,
        val_fraction: float = 0.2,
        device: str | torch.device | None = None,
    ) -> dict:
        """Train the probe on activations from a TappedModel.

        Args:
            model: TappedModel to extract activations from.
            hook_point: Which hook point to probe.
            texts: Input texts.
            labels: Target labels. Ints for classification, floats for regression.
                For position="all", provide list of lists (per-token labels).
            position: Which token position to probe.
                -1 (default): last token.
                int: specific position index.
                "all": all positions (token-level task).
            lr: Learning rate.
            n_epochs: Number of training epochs.
            batch_size: Batch size.
            val_fraction: Fraction of data for validation.
            device: Device to train on.

        Returns:
            Dict with training metrics.
        """
        if device is None:
            device = model.device
        self.to(device)

        # Collect activations
        activations = _collect_probe_activations(
            model, hook_point, texts, position, device
        )

        # Prepare labels
        if isinstance(labels, list):
            if position == "all":
                labels = torch.tensor([l for seq in labels for l in seq], device=device)
            else:
                labels = torch.tensor(labels, device=device)
        else:
            labels = labels.to(device)

        if self.task == "regression":
            labels = labels.float()
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
        else:
            labels = labels.long()

        return self._train(activations, labels, lr, n_epochs, batch_size, val_fraction)

    def fit_on_activations(
        self,
        activations: torch.Tensor,
        labels: list | torch.Tensor,
        lr: float = 1e-3,
        n_epochs: int = 50,
        batch_size: int = 32,
        val_fraction: float = 0.2,
    ) -> dict:
        """Train directly on pre-cached activations. No model needed."""
        device = activations.device
        self.to(device)

        if isinstance(labels, list):
            labels = torch.tensor(labels, device=device)
        else:
            labels = labels.to(device)

        if self.task == "regression":
            labels = labels.float()
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
        else:
            labels = labels.long()

        return self._train(activations, labels, lr, n_epochs, batch_size, val_fraction)

    def _train(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        lr: float,
        n_epochs: int,
        batch_size: int,
        val_fraction: float,
    ) -> dict:
        n_samples = activations.shape[0]
        n_val = max(1, int(n_samples * val_fraction))
        n_train = n_samples - n_val

        # Shuffle and split
        perm = torch.randperm(n_samples)
        train_acts = activations[perm[:n_train]]
        train_labels = labels[perm[:n_train]]
        val_acts = activations[perm[n_train:]]
        val_labels = labels[perm[n_train:]]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(n_epochs):
            self.train()
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_train, batch_size):
                batch_acts = train_acts[i : i + batch_size]
                batch_labels = train_labels[i : i + batch_size]

                logits = self(batch_acts)
                loss = self.loss_fn(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss /= max(n_batches, 1)

        # Evaluate on validation set
        self.eval()
        with torch.no_grad():
            val_logits = self(val_acts)
            val_loss = self.loss_fn(val_logits, val_labels).item()

            if self.task == "classification":
                preds = val_logits.argmax(dim=-1)
                self.accuracy = (preds == val_labels).float().mean().item()
            else:
                self.accuracy = None

        self.train_loss = val_loss

        result = {"val_loss": val_loss}
        if self.accuracy is not None:
            result["accuracy"] = self.accuracy

        return result

    @classmethod
    def sweep(
        cls,
        model: TappedModel,
        hook_points: str | list[str],
        texts: list[str],
        labels: list | torch.Tensor,
        n_classes: int = 2,
        task: str | None = None,
        hidden_dims: list[int] | None = None,
        position: int | str = -1,
        lr: float = 1e-3,
        n_epochs: int = 50,
        batch_size: int = 32,
        val_fraction: float = 0.2,
        device: str | torch.device | None = None,
    ) -> ProbeResult:
        """Train a probe at each hook point and compare.

        Args:
            hook_points: Either a single string with '{i}' (expands to all layers)
                or a list of specific hook point names.

        Returns:
            ProbeResult with accuracies and losses per hook point.
        """
        n_layers = model._detect_n_layers()

        if isinstance(hook_points, str):
            if "{i}" in hook_points:
                expanded = [hook_points.replace("{i}", str(i)) for i in range(n_layers)]
            else:
                expanded = [hook_points]
        else:
            expanded = hook_points

        if device is None:
            device = model.device

        # Detect d_model from first hook point
        _, cache = model.run_with_cache(text=texts[0], names=[expanded[0]])
        act = cache[expanded[0]]
        if position == "all":
            d_model = act.shape[-1]
        else:
            d_model = act.shape[-1]

        accuracies = {}
        losses = {}
        probes = {}

        for hook_point in expanded:
            probe = cls(
                d_model=d_model,
                n_classes=n_classes,
                task=task,
                hidden_dims=hidden_dims,
            )
            result = probe.fit(
                model,
                hook_point=hook_point,
                texts=texts,
                labels=labels,
                position=position,
                lr=lr,
                n_epochs=n_epochs,
                batch_size=batch_size,
                val_fraction=val_fraction,
                device=device,
            )

            if probe.accuracy is not None:
                accuracies[hook_point] = probe.accuracy
            losses[hook_point] = result["val_loss"]
            probes[hook_point] = probe

            acc_str = f" | acc: {probe.accuracy:.3f}" if probe.accuracy is not None else ""
            print(f"{hook_point}: loss={result['val_loss']:.4f}{acc_str}")

        return ProbeResult(
            accuracies=accuracies,
            losses=losses,
            probes=probes,
        )

    # --- Save/Load ---

    def save(self, path: str | Path) -> None:
        """Save probe weights and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), path / "weights.pt")

        config = {
            "d_model": self.d_model,
            "n_classes": self.n_classes,
            "task": self.task,
            "hidden_dims": self._hidden_dims,
            "accuracy": self.accuracy,
            "train_loss": self.train_loss,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Probe:
        """Load a probe from a directory saved with .save()."""
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)

        probe = cls(
            d_model=config["d_model"],
            n_classes=config["n_classes"],
            task=config["task"],
            hidden_dims=config.get("hidden_dims"),
        )
        probe.load_state_dict(torch.load(path / "weights.pt", weights_only=True))
        probe.accuracy = config.get("accuracy")
        probe.train_loss = config.get("train_loss")
        return probe

    def __repr__(self) -> str:
        parts = [f"d_model={self.d_model}"]
        if self.task == "classification":
            parts.append(f"n_classes={self.n_classes}")
        parts.append(f"task='{self.task}'")
        if self._hidden_dims:
            parts.append(f"hidden_dims={self._hidden_dims}")
        if self.accuracy is not None:
            parts.append(f"accuracy={self.accuracy:.3f}")
        return f"Probe({', '.join(parts)})"


def _collect_probe_activations(
    model: TappedModel,
    hook_point: str,
    texts: list[str],
    position: int | str,
    device: torch.device,
) -> torch.Tensor:
    """Collect activations for probing, handling position selection."""
    all_activations = []

    for i in range(0, len(texts), 32):
        batch_texts = texts[i : i + 32]
        with torch.no_grad():
            _, cache = model.run_with_cache(text=batch_texts, names=[hook_point])

        acts = cache[hook_point]  # (batch, seq, d_model)

        if position == "all":
            all_activations.append(acts.reshape(-1, acts.shape[-1]).to(device))
        elif isinstance(position, int):
            all_activations.append(acts[:, position, :].to(device))
        else:
            raise ValueError(f"position must be int or 'all', got {position}")

    return torch.cat(all_activations, dim=0)
