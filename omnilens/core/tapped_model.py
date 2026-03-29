from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable, Optional
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from omnilens.core.cache import ActivationCache
from omnilens.registry.loader import load_registry, Registry


HookFn = Callable[[torch.Tensor, str], Optional[torch.Tensor]]


class TappedModel:
    """Wraps any HuggingFace model with standardized hook access.

    The underlying model is untouched — no reimplementation, no weight
    copying. TappedModel registers PyTorch hooks on the original modules
    and translates between omnilens standardized names and the model's
    native module names.

    Usage:
        model = TappedModel.from_pretrained("meta-llama/Llama-3.1-8B")
        logits, cache = model.run_with_cache(tokens)
        resid = cache["layers.0.residual.attn_out"]
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        registry: Registry | dict | str | Path | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._registry = self._resolve_registry(registry)
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        registry: Registry | dict | str | Path | None = None,
        tokenizer_name: str | None = None,
        **model_kwargs,
    ) -> TappedModel:
        """Load a HuggingFace model and wrap it.

        Registry resolution order:
          1. Explicit registry arg (dict, YAML path, or Registry object)
          2. Built-in registry for this architecture
          3. Auto-detect from model structure
        """
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(model=model, tokenizer=tokenizer, registry=registry)

    def run_with_cache(
        self,
        input_ids: torch.Tensor | None = None,
        names: list[str] | None = None,
        text: str | list[str] | None = None,
        **model_kwargs,
    ) -> tuple[torch.Tensor, ActivationCache]:
        """Run a forward pass and cache activations.

        Args:
            input_ids: Token IDs. Provide this or text, not both.
            names: Which hook points to cache. None means cache everything
                in the registry.
            text: Raw text to tokenize. Provide this or input_ids, not both.
            **model_kwargs: Passed to the underlying model's forward method.

        Returns:
            (logits, cache) tuple.
        """
        if text is not None and input_ids is not None:
            raise ValueError("Provide either text or input_ids, not both.")

        if text is not None:
            if self.tokenizer is None:
                raise ValueError("No tokenizer available. Pass input_ids directly.")
            encoded = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = encoded["input_ids"].to(self._device)
            if "attention_mask" not in model_kwargs:
                model_kwargs["attention_mask"] = encoded["attention_mask"].to(
                    self._device
                )

        cache = ActivationCache()
        targets = names if names is not None else list(self._registry.keys())

        self._register_cache_hooks(cache, targets)
        try:
            with torch.no_grad():
                output = self.model(input_ids=input_ids, **model_kwargs)
        finally:
            self._remove_hooks()

        logits = output.logits if hasattr(output, "logits") else output[0]
        return logits, cache

    def run_with_hooks(
        self,
        input_ids: torch.Tensor | None = None,
        hooks: dict[str, HookFn] | None = None,
        text: str | list[str] | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Run a forward pass with intervention hooks.

        Args:
            input_ids: Token IDs. Provide this or text, not both.
            hooks: Dict mapping omnilens names to hook functions.
                Each hook receives (activation, hook_name) and optionally
                returns a modified activation. If None is returned, the
                original activation is used unchanged.
            text: Raw text to tokenize.
            **model_kwargs: Passed to the underlying model's forward method.

        Returns:
            Model output logits.
        """
        if text is not None and input_ids is not None:
            raise ValueError("Provide either text or input_ids, not both.")

        if text is not None:
            if self.tokenizer is None:
                raise ValueError("No tokenizer available. Pass input_ids directly.")
            encoded = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = encoded["input_ids"].to(self._device)
            if "attention_mask" not in model_kwargs:
                model_kwargs["attention_mask"] = encoded["attention_mask"].to(
                    self._device
                )

        hooks = hooks or {}
        self._register_intervention_hooks(hooks)
        try:
            output = self.model(input_ids=input_ids, **model_kwargs)
        finally:
            self._remove_hooks()

        return output.logits if hasattr(output, "logits") else output[0]

    def module_names(self) -> list[str]:
        """List all named modules in the underlying model."""
        return [name for name, _ in self.model.named_modules() if name]

    def print_module_tree(self) -> None:
        """Print the model's module tree for manual inspection."""
        for name, module in self.model.named_modules():
            if not name:
                continue
            depth = name.count(".")
            short_name = name.split(".")[-1]
            module_type = type(module).__name__
            indent = "  " * depth
            print(f"{indent}{short_name} ({module_type}) -> {name}")

    def registry_names(self) -> list[str]:
        """List all standardized names available in the current registry."""
        return list(self._registry.keys())

    # -- Internal methods --

    def _resolve_registry(
        self, registry: Registry | dict | str | Path | None
    ) -> Registry:
        """Resolve registry from explicit arg, built-in, or auto-detect."""
        if isinstance(registry, dict):
            return Registry(registry)
        if isinstance(registry, (str, Path)):
            return load_registry(registry)
        if isinstance(registry, Registry):
            return registry

        # Try built-in registry for this model's architecture
        arch_name = self._detect_architecture()
        builtin = load_registry(arch_name)
        if builtin is not None:
            n_layers = self._detect_n_layers()
            if n_layers is not None:
                return builtin.expand_layers(n_layers)
            return builtin

        # Fallback: auto-detect
        from omnilens.registry.auto_detect import auto_detect_registry

        detected = auto_detect_registry(self.model)
        if detected is not None:
            return detected

        # Empty registry — user can still use raw module names
        return Registry({})

    def _detect_architecture(self) -> str | None:
        """Get the architecture name from the model config."""
        if hasattr(self.model, "config") and hasattr(
            self.model.config, "model_type"
        ):
            return self.model.config.model_type
        return None

    def _detect_n_layers(self) -> int | None:
        """Get the number of layers from model config or structure."""
        config = getattr(self.model, "config", None)
        if config is not None:
            for attr in ("num_hidden_layers", "n_layer", "num_layers"):
                if hasattr(config, attr):
                    return getattr(config, attr)

        # Fallback: count ModuleList children
        for _, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 1:
                return len(module)
        return None

    def _resolve_name(self, omnilens_name: str) -> str:
        """Map an omnilens name to the model's native module path.

        If the name is already a valid native module path, use it directly.
        This lets users fall back to raw names when no registry mapping exists.
        """
        if omnilens_name in self._registry:
            return self._registry[omnilens_name]

        # Check if it's a raw module name that exists on the model
        try:
            self._get_module(omnilens_name)
            return omnilens_name
        except AttributeError:
            raise KeyError(
                f"'{omnilens_name}' is not in the registry and is not a valid "
                f"module path. Use model.registry_names() to see available "
                f"names or model.module_names() to see raw module paths."
            )

    def _get_module(self, module_path: str) -> nn.Module:
        """Get a module by its dot-separated path."""
        parts = module_path.split(".")
        current = self.model
        for part in parts:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        return current

    def _register_cache_hooks(
        self, cache: ActivationCache, names: list[str]
    ) -> None:
        """Register forward hooks that store activations in the cache."""
        for name in names:
            native_path = self._resolve_name(name)
            module = self._get_module(native_path)
            omnilens_name = name  # capture for closure

            def make_hook(hook_name: str):
                def hook_fn(mod, input, output):
                    if isinstance(output, tuple):
                        cache[hook_name] = output[0].detach()
                    else:
                        cache[hook_name] = output.detach()

                return hook_fn

            handle = module.register_forward_hook(make_hook(omnilens_name))
            self._hook_handles.append(handle)

    def _register_intervention_hooks(self, hooks: dict[str, HookFn]) -> None:
        """Register forward hooks that can modify activations."""
        for name, hook_fn in hooks.items():
            native_path = self._resolve_name(name)
            module = self._get_module(native_path)
            omnilens_name = name

            def make_hook(hook_name: str, user_fn: HookFn):
                def hook_fn(mod, input, output):
                    if isinstance(output, tuple):
                        modified = user_fn(output[0], hook_name)
                        if modified is not None:
                            return (modified,) + output[1:]
                    else:
                        modified = user_fn(output, hook_name)
                        if modified is not None:
                            return modified
                    return output

                return hook_fn

            handle = module.register_forward_hook(
                make_hook(omnilens_name, hook_fn)
            )
            self._hook_handles.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    @property
    def _device(self) -> torch.device:
        """Get the device of the underlying model."""
        return next(self.model.parameters()).device

    def __repr__(self) -> str:
        arch = self._detect_architecture() or "unknown"
        n_registry = len(self._registry)
        return f"TappedModel(arch={arch}, registry_entries={n_registry})"
