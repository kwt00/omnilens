"""Tests for SAE — composable sparse autoencoders."""

import pytest
import torch
import torch.nn as nn
from omnilens import TappedModel, SAE
from omnilens.methods.sae import (
    relu_activation,
    topk_activation,
    l1_sparsity,
    no_sparsity,
)


@pytest.fixture
def small_model():
    return TappedModel.from_pretrained(
        "sshleifer/tiny-gpt2", dtype=torch.float32
    )


@pytest.fixture
def random_activations():
    torch.manual_seed(42)
    return torch.randn(16, 10, 32)


class TestFactoryMethods:
    def test_relu(self, random_activations):
        sae = SAE.relu(d_model=32, n_features=64)
        result = sae(random_activations)
        assert result.features.shape == (16, 10, 64)
        assert result.reconstruction.shape == (16, 10, 32)
        assert (result.features >= 0).all()

    def test_topk(self, random_activations):
        sae = SAE.topk(d_model=32, n_features=64, k=8)
        result = sae(random_activations)
        n_active = (result.features > 0).sum(dim=-1)
        assert (n_active <= 8).all()
        assert result.sparsity_loss.item() == 0.0

    def test_jumprelu(self, random_activations):
        sae = SAE.jumprelu(d_model=32, n_features=64, initial_threshold=100.0)
        features = sae.encode(random_activations)
        assert (features == 0).all()  # threshold too high

    def test_gated(self, random_activations):
        sae = SAE.gated(d_model=32, n_features=64)
        result = sae(random_activations)
        assert result.features.shape == (16, 10, 64)
        assert hasattr(sae, "gate")

    def test_tied_weights(self, random_activations):
        sae = SAE.relu(d_model=32, n_features=64, tied_weights=True)
        result = sae(random_activations)
        assert result.reconstruction.shape == (16, 10, 32)

    def test_hidden_dims_single(self, random_activations):
        sae = SAE.relu(d_model=32, n_features=64, hidden_dims=[48])
        result = sae(random_activations)
        assert result.features.shape == (16, 10, 64)
        assert result.reconstruction.shape == (16, 10, 32)

    def test_hidden_dims_multiple(self, random_activations):
        sae = SAE.topk(d_model=32, n_features=64, k=8, hidden_dims=[48, 56])
        result = sae(random_activations)
        assert result.features.shape == (16, 10, 64)
        n_active = (result.features > 0).sum(dim=-1)
        assert (n_active <= 8).all()

    def test_hidden_dims_gradient(self, random_activations):
        sae = SAE.relu(d_model=32, n_features=64, hidden_dims=[48])
        result = sae(random_activations)
        result.loss.backward()
        # All layers should get gradients
        for param in sae.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestCustomComposition:
    def test_custom_encoder_decoder(self, random_activations):
        """User provides their own encoder/decoder modules."""
        sae = SAE(
            encoder=nn.Linear(32, 64),
            decoder=nn.Linear(64, 32, bias=False),
        )
        result = sae(random_activations)
        assert result.features.shape == (16, 10, 64)

    def test_deep_encoder(self, random_activations):
        """Multi-layer encoder."""
        sae = SAE(
            encoder=nn.Sequential(
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            ),
            decoder=nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 32, bias=False),
            ),
        )
        result = sae(random_activations)
        assert result.features.shape == (16, 10, 64)
        assert result.reconstruction.shape == (16, 10, 32)

    def test_custom_activation_fn(self, random_activations):
        """Custom activation function."""
        def my_activation(x):
            return torch.clamp(x, min=0.1)

        sae = SAE(
            encoder=nn.Linear(32, 64),
            decoder=nn.Linear(64, 32, bias=False),
            activation_fn=my_activation,
        )
        features = sae.encode(random_activations)
        assert (features >= 0.1).all()

    def test_custom_sparsity_fn(self, random_activations):
        """Custom sparsity loss."""
        def my_sparsity(features):
            return features.pow(2).mean()  # L2 sparsity

        sae = SAE(
            encoder=nn.Linear(32, 64),
            decoder=nn.Linear(64, 32, bias=False),
            sparsity_fn=my_sparsity,
        )
        result = sae(random_activations)
        assert result.sparsity_loss > 0

    def test_swap_activation_at_runtime(self, random_activations):
        """User can swap activation function after construction."""
        sae = SAE.relu(d_model=32, n_features=64)
        sae.activation_fn = topk_activation(4)
        features = sae.encode(random_activations)
        n_active = (features > 0).sum(dim=-1)
        assert (n_active <= 4).all()


class TestForwardPass:
    def test_loss_is_finite(self, random_activations):
        sae = SAE.relu(d_model=32, n_features=64)
        result = sae(random_activations)
        assert torch.isfinite(result.loss)

    def test_backward_pass(self, random_activations):
        sae = SAE.relu(d_model=32, n_features=64)
        result = sae(random_activations)
        result.loss.backward()
        assert sae.encoder.weight.grad is not None
        assert torch.isfinite(sae.encoder.weight.grad).all()

    def test_encode_decode_roundtrip(self, random_activations):
        sae = SAE.relu(d_model=32, n_features=64)
        features = sae.encode(random_activations)
        reconstruction = sae.decode(features)
        assert reconstruction.shape == random_activations.shape


class TestFit:
    def test_fit_with_tensor(self, small_model):
        sae = SAE.relu(d_model=8, n_features=16)
        torch.manual_seed(42)
        fake_acts = torch.randn(100, 8)
        logs = sae.fit(
            small_model,
            hook_point="transformer.h.0.mlp.c_fc",
            dataset=fake_acts,
            n_steps=10,
            log_every=5,
            batch_size=16,
        )
        assert len(logs) == 2

    def test_fit_with_strings(self, small_model):
        sae = SAE.relu(d_model=8, n_features=16)
        texts = ["Hello world", "The cat sat", "Testing one two three"]
        logs = sae.fit(
            small_model,
            hook_point="transformer.h.0.mlp.c_fc",
            dataset=texts,
            n_steps=5,
            log_every=5,
            batch_size=8,
        )
        assert len(logs) == 1


class TestWithHooks:
    def test_sae_intervention(self, small_model):
        sae = SAE.relu(d_model=8, n_features=16)
        baseline_logits, _ = small_model.run_with_cache(text="Hello world")

        def sae_reconstruct(activation, hook_name):
            features = sae.encode(activation)
            return sae.decode(features)

        modified_logits = small_model.run_with_hooks(
            text="Hello world",
            hooks={"transformer.h.0.mlp.c_fc": sae_reconstruct},
        )
        assert not torch.allclose(baseline_logits, modified_logits)

    def test_feature_ablation(self, small_model):
        sae = SAE.relu(d_model=8, n_features=16)

        def ablate_feature_0(activation, hook_name):
            features = sae.encode(activation)
            features[..., 0] = 0
            return sae.decode(features)

        logits = small_model.run_with_hooks(
            text="Hello world",
            hooks={"transformer.h.0.mlp.c_fc": ablate_feature_0},
        )
        assert logits is not None


class TestRepr:
    def test_repr(self):
        sae = SAE.relu(d_model=32, n_features=64)
        r = repr(sae)
        assert "SAE" in r
        assert "encoder" in r
        assert "decoder" in r
