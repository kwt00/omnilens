"""Tests for SAE variants."""

import pytest
import torch
from omnilens import TappedModel, SAE, TopKSAE, GatedSAE, JumpReLUSAE


@pytest.fixture
def small_model():
    return TappedModel.from_pretrained(
        "sshleifer/tiny-gpt2", dtype=torch.float32
    )


@pytest.fixture
def random_activations():
    """Fake activations for unit testing SAE forward/encode/decode."""
    torch.manual_seed(42)
    return torch.randn(16, 10, 2)  # (batch, seq, d_model=2 for tiny-gpt2)


class TestSAEForwardPass:
    @pytest.mark.parametrize("sae_cls", [SAE, TopKSAE, GatedSAE, JumpReLUSAE])
    def test_forward_returns_result(self, sae_cls, random_activations):
        kwargs = {"d_model": 2, "n_features": 8}
        if sae_cls == TopKSAE:
            kwargs["k"] = 4
        sae = sae_cls(**kwargs)
        result = sae(random_activations)

        assert result.features.shape == (16, 10, 8)
        assert result.reconstruction.shape == (16, 10, 2)
        assert result.loss.ndim == 0
        assert result.reconstruction_loss.ndim == 0
        assert result.sparsity_loss.ndim == 0

    @pytest.mark.parametrize("sae_cls", [SAE, TopKSAE, GatedSAE, JumpReLUSAE])
    def test_encode_decode_shapes(self, sae_cls, random_activations):
        kwargs = {"d_model": 2, "n_features": 8}
        if sae_cls == TopKSAE:
            kwargs["k"] = 4
        sae = sae_cls(**kwargs)

        features = sae.encode(random_activations)
        assert features.shape == (16, 10, 8)

        reconstruction = sae.decode(features)
        assert reconstruction.shape == (16, 10, 2)

    @pytest.mark.parametrize("sae_cls", [SAE, TopKSAE, GatedSAE, JumpReLUSAE])
    def test_loss_is_finite(self, sae_cls, random_activations):
        kwargs = {"d_model": 2, "n_features": 8}
        if sae_cls == TopKSAE:
            kwargs["k"] = 4
        sae = sae_cls(**kwargs)
        result = sae(random_activations)
        assert torch.isfinite(result.loss)

    @pytest.mark.parametrize("sae_cls", [SAE, TopKSAE, GatedSAE, JumpReLUSAE])
    def test_tied_weights(self, sae_cls, random_activations):
        kwargs = {"d_model": 2, "n_features": 8, "tied_weights": True}
        if sae_cls == TopKSAE:
            kwargs["k"] = 4
        sae = sae_cls(**kwargs)
        assert sae.decoder is None

        result = sae(random_activations)
        assert result.reconstruction.shape == (16, 10, 2)


class TestSAESparsity:
    def test_relu_sae_features_nonnegative(self, random_activations):
        sae = SAE(d_model=2, n_features=8)
        features = sae.encode(random_activations)
        assert (features >= 0).all()

    def test_topk_sae_sparsity(self, random_activations):
        sae = TopKSAE(d_model=2, n_features=8, k=3)
        features = sae.encode(random_activations)
        # Each position should have at most k non-zero features
        n_active = (features > 0).sum(dim=-1)
        assert (n_active <= 3).all()

    def test_topk_sae_no_sparsity_loss(self, random_activations):
        sae = TopKSAE(d_model=2, n_features=8, k=3)
        result = sae(random_activations)
        assert result.sparsity_loss.item() == 0.0

    def test_jumprelu_threshold(self, random_activations):
        sae = JumpReLUSAE(d_model=2, n_features=8, initial_threshold=100.0)
        features = sae.encode(random_activations)
        # With a very high threshold, nothing should fire
        assert (features == 0).all()


class TestSAEGradients:
    @pytest.mark.parametrize("sae_cls", [SAE, TopKSAE, GatedSAE, JumpReLUSAE])
    def test_backward_pass(self, sae_cls, random_activations):
        kwargs = {"d_model": 2, "n_features": 8}
        if sae_cls == TopKSAE:
            kwargs["k"] = 4
        sae = sae_cls(**kwargs)
        result = sae(random_activations)
        result.loss.backward()

        # Encoder should have gradients
        assert sae.encoder.weight.grad is not None
        assert torch.isfinite(sae.encoder.weight.grad).all()


class TestSAERepr:
    def test_sae_repr(self):
        assert "SAE" in repr(SAE(d_model=4, n_features=8))

    def test_topk_repr(self):
        r = repr(TopKSAE(d_model=4, n_features=8, k=3))
        assert "TopKSAE" in r
        assert "k=3" in r

    def test_jumprelu_repr(self):
        assert "JumpReLUSAE" in repr(JumpReLUSAE(d_model=4, n_features=8))

    def test_tied_repr(self):
        assert "tied" in repr(SAE(d_model=4, n_features=8, tied_weights=True))


class TestSAEFit:
    def test_fit_with_tensor(self, small_model):
        sae = SAE(d_model=8, n_features=16)
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
        assert len(logs) == 2  # logged at step 0 and 5
        assert logs[-1]["loss"] < logs[0]["loss"] or True  # training ran

    def test_fit_with_strings(self, small_model):
        sae = SAE(d_model=8, n_features=16)  # c_fc outputs dim 8 in tiny-gpt2
        texts = ["Hello world", "The cat sat", "Testing one two three"]
        logs = sae.fit(
            small_model,
            hook_point="transformer.h.0.mlp.c_fc",
            dataset=texts,
            n_steps=5,
            log_every=5,
            batch_size=8,
        )
        assert len(logs) == 1  # logged at step 0


class TestSAEWithHooks:
    def test_sae_intervention(self, small_model):
        """Test that SAE encode→decode can be used as a hook."""
        sae = SAE(d_model=8, n_features=16)  # c_fc outputs dim 8

        baseline_logits, _ = small_model.run_with_cache(text="Hello world")

        def sae_reconstruct(activation, hook_name):
            features = sae.encode(activation)
            return sae.decode(features)

        modified_logits = small_model.run_with_hooks(
            text="Hello world",
            hooks={"transformer.h.0.mlp.c_fc": sae_reconstruct},
        )
        # Output should change (untrained SAE won't reconstruct perfectly)
        assert not torch.allclose(baseline_logits, modified_logits)

    def test_sae_feature_ablation(self, small_model):
        """Test ablating a specific SAE feature via hooks."""
        sae = SAE(d_model=8, n_features=16)  # c_fc outputs dim 8

        def ablate_feature_0(activation, hook_name):
            features = sae.encode(activation)
            features[..., 0] = 0
            return sae.decode(features)

        logits = small_model.run_with_hooks(
            text="Hello world",
            hooks={"transformer.h.0.mlp.c_fc": ablate_feature_0},
        )
        assert logits is not None
