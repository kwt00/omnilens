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


class TestDerivedAttentionValues:
    def test_cache_qk_logits(self, small_model):
        logits, cache = small_model.run_with_cache(
            text="Hello world",
            names=["layers.0.attention.qk_logits"],
        )
        assert "layers.0.attention.qk_logits" in cache
        tensor = cache["layers.0.attention.qk_logits"]
        assert tensor.ndim == 4  # (batch, heads, seq, seq)

    def test_cache_attention_weights(self, small_model):
        logits, cache = small_model.run_with_cache(
            text="Hello world",
            names=["layers.0.attention.weights"],
        )
        assert "layers.0.attention.weights" in cache
        tensor = cache["layers.0.attention.weights"]
        assert tensor.ndim == 4  # (batch, heads, seq, seq)
        # Weights should sum to ~1 along last dim (softmax)
        sums = tensor.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_cache_weighted_values(self, small_model):
        logits, cache = small_model.run_with_cache(
            text="Hello world",
            names=["layers.0.attention.weighted_values"],
        )
        assert "layers.0.attention.weighted_values" in cache
        tensor = cache["layers.0.attention.weighted_values"]
        assert tensor.ndim == 4  # (batch, heads, seq, head_dim)

    def test_cache_all_derived_at_once(self, small_model):
        logits, cache = small_model.run_with_cache(
            text="Hello world",
            names=[
                "layers.0.attention.qk_logits",
                "layers.0.attention.weights",
                "layers.0.attention.weighted_values",
            ],
        )
        assert len(cache) == 3

    def test_mix_derived_and_module_hooks(self, small_model):
        logits, cache = small_model.run_with_cache(
            text="Hello world",
            names=[
                "layers.0.attention.weights",
                "transformer.h.0.mlp.c_fc",
            ],
        )
        assert "layers.0.attention.weights" in cache
        assert "transformer.h.0.mlp.c_fc" in cache

    def test_intervene_on_attention_weights(self, small_model):
        baseline_logits, _ = small_model.run_with_cache(text="Hello world")

        def zero_attention(activation, hook_name):
            return torch.zeros_like(activation)

        modified_logits = small_model.run_with_hooks(
            text="Hello world",
            hooks={"layers.0.attention.weights": zero_attention},
        )
        assert not torch.allclose(baseline_logits, modified_logits)


class TestSuffixSystem:
    def test_activations_suffix(self, small_model):
        """'.activations' suffix should hook the module output."""
        logits, cache = small_model.run_with_cache(
            text="Hello",
            names=["transformer.h.0.mlp.c_fc.activations"],
        )
        assert "transformer.h.0.mlp.c_fc.activations" in cache
        assert cache["transformer.h.0.mlp.c_fc.activations"].ndim >= 2

    def test_weight_suffix(self, small_model):
        """'.weight' suffix should return the parameter tensor."""
        logits, cache = small_model.run_with_cache(
            text="Hello",
            names=["transformer.h.0.mlp.c_fc.weight"],
        )
        assert "transformer.h.0.mlp.c_fc.weight" in cache
        assert cache["transformer.h.0.mlp.c_fc.weight"].ndim == 2

    def test_bias_suffix(self, small_model):
        """'.bias' suffix should return the bias parameter."""
        logits, cache = small_model.run_with_cache(
            text="Hello",
            names=["transformer.h.0.mlp.c_fc.bias"],
        )
        assert "transformer.h.0.mlp.c_fc.bias" in cache
        assert cache["transformer.h.0.mlp.c_fc.bias"].ndim == 1

    def test_activations_suffix_with_registry_name(self, small_model):
        """Registry names should also accept .activations suffix."""
        logits, cache = small_model.run_with_cache(
            text="Hello",
            names=["layers.0.mlp.up_proj.activations"],
        )
        assert "layers.0.mlp.up_proj.activations" in cache

    def test_weight_suffix_with_registry_name(self, small_model):
        """Registry names should also accept .weight suffix."""
        logits, cache = small_model.run_with_cache(
            text="Hello",
            names=["layers.0.mlp.up_proj.weight"],
        )
        assert "layers.0.mlp.up_proj.weight" in cache
        assert cache["layers.0.mlp.up_proj.weight"].ndim == 2

    def test_cannot_intervene_on_parameters(self, small_model):
        """Intervention on .weight should raise an error."""
        with pytest.raises(ValueError, match="Cannot intervene on parameter"):
            small_model.run_with_hooks(
                text="Hello",
                hooks={"layers.0.mlp.up_proj.weight": lambda a, n: a},
            )

    def test_mix_suffixes(self, small_model):
        """Mix .activations, .weight, and derived names in one call."""
        logits, cache = small_model.run_with_cache(
            text="Hello",
            names=[
                "layers.0.mlp.up_proj.activations",
                "layers.0.mlp.up_proj.weight",
                "layers.0.attention.weights",
            ],
        )
        assert len(cache) == 3


class TestResidualStream:
    def test_residual_input(self, small_model):
        logits, cache = small_model.run_with_cache(
            text="Hello world",
            names=["layers.0.residual.input"],
        )
        assert "layers.0.residual.input" in cache
        assert cache["layers.0.residual.input"].ndim == 3  # (batch, seq, d_model)

    def test_residual_attn_out(self, small_model):
        logits, cache = small_model.run_with_cache(
            text="Hello world",
            names=["layers.0.residual.attn_out"],
        )
        assert "layers.0.residual.attn_out" in cache
        assert cache["layers.0.residual.attn_out"].ndim == 3

    def test_residual_block_out(self, small_model):
        logits, cache = small_model.run_with_cache(
            text="Hello world",
            names=["layers.0.residual.block_out"],
        )
        assert "layers.0.residual.block_out" in cache
        assert cache["layers.0.residual.block_out"].ndim == 3

    def test_all_residual_points(self, small_model):
        logits, cache = small_model.run_with_cache(
            text="Hello world",
            names=[
                "layers.0.residual.input",
                "layers.0.residual.attn_out",
                "layers.0.residual.block_out",
            ],
        )
        assert len(cache) == 3
        # attn_out should differ from input (attention changed it)
        assert not torch.allclose(
            cache["layers.0.residual.input"],
            cache["layers.0.residual.attn_out"],
        )
        # block_out should differ from attn_out (MLP changed it)
        assert not torch.allclose(
            cache["layers.0.residual.attn_out"],
            cache["layers.0.residual.block_out"],
        )

    def test_intervene_on_residual(self, small_model):
        baseline_logits, _ = small_model.run_with_cache(text="Hello world")

        def zero_residual(activation, hook_name):
            return torch.zeros_like(activation)

        modified_logits = small_model.run_with_hooks(
            text="Hello world",
            hooks={"layers.0.residual.attn_out": zero_residual},
        )
        assert not torch.allclose(baseline_logits, modified_logits)
