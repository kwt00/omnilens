"""Tests for SAE — composable sparse autoencoders."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from omnilens import TappedModel, SAE


@pytest.fixture
def small_model():
    return TappedModel.from_pretrained(
        "sshleifer/tiny-gpt2", dtype=torch.float32
    )


@pytest.fixture
def random_activations():
    torch.manual_seed(42)
    return torch.randn(16, 10, 32)


class TestConstructor:
    def test_relu(self, random_activations):
        sae = SAE(d_model=32, n_features=64, activation="relu")
        result = sae(random_activations)
        assert result.features.shape == (16, 10, 64)
        assert result.reconstruction.shape == (16, 10, 32)
        assert (result.features >= 0).all()

    def test_topk(self, random_activations):
        sae = SAE(d_model=32, n_features=64, activation="topk", k=8)
        result = sae(random_activations)
        n_active = (result.features > 0).sum(dim=-1)
        assert (n_active <= 8).all()
        assert result.sparsity_loss.item() == 0.0

    def test_jumprelu(self, random_activations):
        sae = SAE(d_model=32, n_features=64, activation="jumprelu", initial_threshold=100.0)
        features = sae.encode(random_activations)
        assert (features == 0).all()

    def test_gated(self, random_activations):
        sae = SAE(d_model=32, n_features=64, activation="gated")
        result = sae(random_activations)
        assert result.features.shape == (16, 10, 64)
        assert hasattr(sae, "gate")

    def test_tied_weights(self, random_activations):
        sae = SAE(d_model=32, n_features=64, activation="relu", tied_weights=True)
        result = sae(random_activations)
        assert result.reconstruction.shape == (16, 10, 32)

    def test_hidden_dims_single(self, random_activations):
        sae = SAE(d_model=32, n_features=64, activation="relu", hidden_dims=[48])
        result = sae(random_activations)
        assert result.features.shape == (16, 10, 64)
        assert result.reconstruction.shape == (16, 10, 32)

    def test_hidden_dims_multiple(self, random_activations):
        sae = SAE(d_model=32, n_features=64, activation="topk", k=8, hidden_dims=[40, 52])
        result = sae(random_activations)
        n_active = (result.features > 0).sum(dim=-1)
        assert (n_active <= 8).all()

    def test_custom_activation(self, random_activations):
        def my_activation(x):
            return torch.clamp(x, min=0.1)

        sae = SAE(d_model=32, n_features=64, activation=my_activation)
        features = sae.encode(random_activations)
        assert (features >= 0.1).all()

    def test_custom_sparsity(self, random_activations):
        def l2_sparsity(features):
            return features.pow(2).mean()

        sae = SAE(d_model=32, n_features=64, activation="relu", sparsity=l2_sparsity)
        result = sae(random_activations)
        assert result.sparsity_loss > 0

    def test_custom_encoder_decoder(self, random_activations):
        sae = SAE(
            d_model=32,
            n_features=64,
            encoder=nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 64)),
            decoder=nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32)),
        )
        result = sae(random_activations)
        assert result.features.shape == (16, 10, 64)

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            SAE(d_model=32, n_features=64, activation="banana")


class TestForwardPass:
    def test_loss_is_finite(self, random_activations):
        sae = SAE(d_model=32, n_features=64)
        result = sae(random_activations)
        assert torch.isfinite(result.loss)

    def test_backward_pass(self, random_activations):
        sae = SAE(d_model=32, n_features=64)
        result = sae(random_activations)
        result.loss.backward()
        assert sae.encoder.weight.grad is not None

    def test_hidden_dims_gradient(self, random_activations):
        sae = SAE(d_model=32, n_features=64, hidden_dims=[48])
        result = sae(random_activations)
        result.loss.backward()
        for param in sae.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_encode_decode_roundtrip(self, random_activations):
        sae = SAE(d_model=32, n_features=64)
        features = sae.encode(random_activations)
        reconstruction = sae.decode(features)
        assert reconstruction.shape == random_activations.shape


class TestHookHelpers:
    def test_hook_ablate(self, small_model):
        sae = SAE(d_model=8, n_features=16)
        baseline, _ = small_model.run_with_cache(text="Hello world")
        modified = small_model.run_with_hooks(
            text="Hello world",
            hooks={"transformer.h.0.mlp.c_fc": sae.hook_ablate([0, 1, 2])},
        )
        assert not torch.allclose(baseline, modified)

    def test_hook_amplify(self, small_model):
        sae = SAE(d_model=8, n_features=16)
        baseline, _ = small_model.run_with_cache(text="Hello world")
        modified = small_model.run_with_hooks(
            text="Hello world",
            hooks={"transformer.h.0.mlp.c_fc": sae.hook_amplify(0, scale=10.0)},
        )
        assert not torch.allclose(baseline, modified)

    def test_hook_clamp(self, small_model):
        sae = SAE(d_model=8, n_features=16)
        logits = small_model.run_with_hooks(
            text="Hello world",
            hooks={"transformer.h.0.mlp.c_fc": sae.hook_clamp(0, value=100.0)},
        )
        assert logits is not None

    def test_hook_reconstruct(self, small_model):
        sae = SAE(d_model=8, n_features=16)
        baseline, _ = small_model.run_with_cache(text="Hello world")
        modified = small_model.run_with_hooks(
            text="Hello world",
            hooks={"transformer.h.0.mlp.c_fc": sae.hook_reconstruct()},
        )
        assert not torch.allclose(baseline, modified)


class TestFit:
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
        assert len(logs) == 2

    def test_fit_with_strings(self, small_model):
        sae = SAE(d_model=8, n_features=16)
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

    def test_fit_on_activations(self):
        sae = SAE(d_model=32, n_features=64)
        torch.manual_seed(42)
        acts = torch.randn(200, 32)
        logs = sae.fit_on_activations(acts, n_steps=10, log_every=5, batch_size=16)
        assert len(logs) == 2


class TestSaveLoad:
    def test_save_and_load(self, tmp_path, random_activations):
        sae = SAE(d_model=32, n_features=64, activation="topk", k=8)
        result_before = sae(random_activations)

        sae.save(tmp_path / "my_sae")

        loaded = SAE.load(tmp_path / "my_sae")
        result_after = loaded(random_activations)

        assert torch.allclose(result_before.features, result_after.features)
        assert torch.allclose(result_before.reconstruction, result_after.reconstruction)

    def test_save_load_with_hidden_dims(self, tmp_path, random_activations):
        sae = SAE(d_model=32, n_features=64, activation="relu", hidden_dims=[48])
        result_before = sae(random_activations)

        sae.save(tmp_path / "deep_sae")
        loaded = SAE.load(tmp_path / "deep_sae")
        result_after = loaded(random_activations)

        assert torch.allclose(result_before.features, result_after.features)

    def test_config_file_contents(self, tmp_path):
        import json
        sae = SAE(d_model=32, n_features=64, activation="topk", k=16, hidden_dims=[48])
        sae.save(tmp_path / "sae")

        with open(tmp_path / "sae" / "config.json") as f:
            config = json.load(f)

        assert config["d_model"] == 32
        assert config["n_features"] == 64
        assert config["activation"] == "topk"
        assert config["k"] == 16
        assert config["hidden_dims"] == [48]


class TestRepr:
    def test_relu_repr(self):
        sae = SAE(d_model=32, n_features=64)
        assert "SAE" in repr(sae)
        assert "relu" in repr(sae)

    def test_topk_repr(self):
        r = repr(SAE(d_model=32, n_features=64, activation="topk", k=8))
        assert "topk" in r
        assert "k=8" in r

    def test_hidden_dims_repr(self):
        r = repr(SAE(d_model=32, n_features=64, hidden_dims=[48]))
        assert "hidden_dims" in r
