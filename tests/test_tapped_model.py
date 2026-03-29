"""Basic integration test using a tiny model."""

import pytest
import torch
from omnilens import TappedModel
from omnilens.core.cache import ActivationCache
from omnilens.registry.loader import Registry


@pytest.fixture
def small_model():
    """Load a tiny GPT-2 for fast testing."""
    return TappedModel.from_pretrained(
        "sshleifer/tiny-gpt2",
        torch_dtype=torch.float32,
    )


class TestTappedModelInit:
    def test_from_pretrained_loads(self, small_model):
        assert small_model.model is not None
        assert small_model.tokenizer is not None

    def test_repr(self, small_model):
        r = repr(small_model)
        assert "TappedModel" in r

    def test_module_names_nonempty(self, small_model):
        names = small_model.module_names()
        assert len(names) > 0

    def test_print_module_tree(self, small_model, capsys):
        small_model.print_module_tree()
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestRunWithCache:
    def test_returns_logits_and_cache(self, small_model):
        logits, cache = small_model.run_with_cache(
            text="Hello world"
        )
        assert isinstance(logits, torch.Tensor)
        assert isinstance(cache, ActivationCache)
        assert logits.ndim == 3  # (batch, seq, vocab)

    def test_cache_with_raw_module_names(self, small_model):
        """Even without a proper registry, raw module names should work."""
        logits, cache = small_model.run_with_cache(
            text="Hello world",
            names=["transformer.h.0.attn.c_attn"],
        )
        assert "transformer.h.0.attn.c_attn" in cache
        assert isinstance(cache["transformer.h.0.attn.c_attn"], torch.Tensor)

    def test_cache_multiple_points(self, small_model):
        names = [
            "transformer.h.0.attn.c_attn",
            "transformer.h.0.mlp.c_fc",
        ]
        logits, cache = small_model.run_with_cache(
            text="Hello world",
            names=names,
        )
        assert len(cache) == 2
        for name in names:
            assert name in cache

    def test_hooks_cleaned_up(self, small_model):
        small_model.run_with_cache(text="Hello world")
        assert len(small_model._hook_handles) == 0


class TestRunWithHooks:
    def test_intervention_modifies_output(self, small_model):
        # Get baseline logits
        baseline_logits, _ = small_model.run_with_cache(text="Hello world")

        # Intervene by zeroing out MLP output at layer 0
        def zero_out(activation, hook_name):
            return torch.zeros_like(activation)

        modified_logits = small_model.run_with_hooks(
            text="Hello world",
            hooks={"transformer.h.0.mlp.c_fc": zero_out},
        )

        # Outputs should differ
        assert not torch.allclose(baseline_logits, modified_logits)

    def test_hooks_cleaned_up_after_intervention(self, small_model):
        def noop(activation, hook_name):
            return None

        small_model.run_with_hooks(
            text="Hello world",
            hooks={"transformer.h.0.mlp.c_fc": noop},
        )
        assert len(small_model._hook_handles) == 0


class TestRegistry:
    def test_empty_registry_falls_back_to_raw_names(self):
        model = TappedModel.from_pretrained(
            "sshleifer/tiny-gpt2",
            registry={},
            torch_dtype=torch.float32,
        )
        # Should still work with raw module names
        logits, cache = model.run_with_cache(
            text="test",
            names=["transformer.h.0.attn.c_attn"],
        )
        assert "transformer.h.0.attn.c_attn" in cache

    def test_custom_dict_registry(self):
        model = TappedModel.from_pretrained(
            "sshleifer/tiny-gpt2",
            registry={
                "my_custom_name": "transformer.h.0.mlp.c_fc",
            },
            torch_dtype=torch.float32,
        )
        logits, cache = model.run_with_cache(
            text="test",
            names=["my_custom_name"],
        )
        assert "my_custom_name" in cache

    def test_invalid_name_raises(self, small_model):
        with pytest.raises(KeyError, match="not in the registry"):
            small_model.run_with_cache(
                text="test",
                names=["nonexistent.module.path"],
            )
