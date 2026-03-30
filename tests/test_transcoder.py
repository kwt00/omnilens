"""Tests for Transcoder — sparse MLP replacement."""

import pytest
import torch
import torch.nn as nn
from omnilens import TappedModel, Transcoder


@pytest.fixture
def small_model():
    return TappedModel.from_pretrained(
        "sshleifer/tiny-gpt2", dtype=torch.float32
    )


@pytest.fixture
def random_pairs():
    """Fake (MLP input, MLP output) pairs."""
    torch.manual_seed(42)
    inputs = torch.randn(16, 10, 32)
    outputs = torch.randn(16, 10, 32)
    return inputs, outputs


class TestConstructor:
    def test_relu(self, random_pairs):
        inputs, outputs = random_pairs
        tc = Transcoder(d_input=32, d_output=32, n_features=64, activation="relu")
        result = tc(inputs, outputs)
        assert result.features.shape == (16, 10, 64)
        assert result.prediction.shape == (16, 10, 32)

    def test_topk(self, random_pairs):
        inputs, outputs = random_pairs
        tc = Transcoder(d_input=32, d_output=32, n_features=64, activation="topk", k=8)
        result = tc(inputs, outputs)
        n_active = (result.features > 0).sum(dim=-1)
        assert (n_active <= 8).all()

    def test_gated(self, random_pairs):
        inputs, _ = random_pairs
        tc = Transcoder(d_input=32, d_output=32, n_features=64, activation="gated")
        features = tc.encode(inputs)
        assert features.shape == (16, 10, 64)

    def test_jumprelu(self, random_pairs):
        inputs, _ = random_pairs
        tc = Transcoder(d_input=32, d_output=32, n_features=64, activation="jumprelu", initial_threshold=100.0)
        features = tc.encode(inputs)
        assert (features == 0).all()

    def test_different_input_output_dims(self, random_pairs):
        inputs = torch.randn(16, 10, 32)
        outputs = torch.randn(16, 10, 64)
        tc = Transcoder(d_input=32, d_output=64, n_features=128)
        result = tc(inputs, outputs)
        assert result.prediction.shape == (16, 10, 64)

    def test_hidden_dims(self, random_pairs):
        inputs, outputs = random_pairs
        tc = Transcoder(d_input=32, d_output=32, n_features=64, hidden_dims=[48])
        result = tc(inputs, outputs)
        assert result.features.shape == (16, 10, 64)

    def test_custom_activation(self, random_pairs):
        inputs, _ = random_pairs
        tc = Transcoder(
            d_input=32, d_output=32, n_features=64,
            activation=lambda x: torch.clamp(x, min=0.5),
        )
        features = tc.encode(inputs)
        assert (features >= 0.5).all()

    def test_custom_encoder_decoder(self, random_pairs):
        inputs, outputs = random_pairs
        tc = Transcoder(
            d_input=32, d_output=32, n_features=64,
            encoder=nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 64)),
            decoder=nn.Linear(64, 32, bias=False),
        )
        result = tc(inputs, outputs)
        assert result.prediction.shape == (16, 10, 32)


class TestSkipConnection:
    def test_skip_changes_output(self, random_pairs):
        inputs, outputs = random_pairs
        tc_no_skip = Transcoder(d_input=32, d_output=32, n_features=64, skip=False)
        tc_skip = Transcoder(d_input=32, d_output=32, n_features=64, skip=True)

        pred_no_skip = tc_no_skip(inputs, outputs).prediction
        pred_skip = tc_skip(inputs, outputs).prediction
        assert not torch.allclose(pred_no_skip, pred_skip)

    def test_skip_has_linear(self):
        tc = Transcoder(d_input=32, d_output=32, n_features=64, skip=True)
        assert tc.skip_linear is not None

    def test_no_skip_no_linear(self):
        tc = Transcoder(d_input=32, d_output=32, n_features=64, skip=False)
        assert tc.skip_linear is None


class TestCrossLayer:
    def test_multi_decoder(self):
        tc = Transcoder(
            d_input=32, d_output=32, n_features=64,
            output_layers=[0, 1, 2],
        )
        assert tc.multi_decoder is not None
        assert "0" in tc.multi_decoder
        assert "1" in tc.multi_decoder
        assert "2" in tc.multi_decoder

    def test_multi_decode(self):
        tc = Transcoder(
            d_input=32, d_output=32, n_features=64,
            output_layers=[0, 1, 2],
        )
        torch.manual_seed(42)
        inputs = torch.randn(4, 5, 32)
        features = tc.encode(inputs)

        out_0 = tc.decode(features, layer=0)
        out_1 = tc.decode(features, layer=1)
        assert out_0.shape == (4, 5, 32)
        assert not torch.allclose(out_0, out_1)

    def test_multi_decode_requires_layer(self):
        tc = Transcoder(
            d_input=32, d_output=32, n_features=64,
            output_layers=[0, 1],
        )
        features = torch.randn(4, 5, 64)
        with pytest.raises(ValueError, match="Must specify layer"):
            tc.decode(features)


class TestForwardPass:
    def test_loss_is_finite(self, random_pairs):
        inputs, outputs = random_pairs
        tc = Transcoder(d_input=32, d_output=32, n_features=64)
        result = tc(inputs, outputs)
        assert torch.isfinite(result.loss)

    def test_backward_pass(self, random_pairs):
        inputs, outputs = random_pairs
        tc = Transcoder(d_input=32, d_output=32, n_features=64)
        result = tc(inputs, outputs)
        result.loss.backward()
        assert tc.encoder.weight.grad is not None

    def test_no_target_no_prediction_loss(self, random_pairs):
        inputs, _ = random_pairs
        tc = Transcoder(d_input=32, d_output=32, n_features=64)
        result = tc(inputs)
        assert result.prediction_loss.item() == 0.0


class TestFeatureControl:
    def test_ablate_features(self, random_pairs):
        inputs, _ = random_pairs
        tc = Transcoder(d_input=32, d_output=32, n_features=64)

        features_before = tc.encode(inputs)
        tc.ablate_features([0, 1, 2])
        features_after = tc.encode(inputs)
        tc.restore_features()

        assert (features_after[..., 0] == 0).all()
        assert (features_after[..., 1] == 0).all()
        assert (features_after[..., 2] == 0).all()

    def test_restore_features(self, random_pairs):
        inputs, _ = random_pairs
        tc = Transcoder(d_input=32, d_output=32, n_features=64)

        tc.ablate_features([0, 1, 2])
        tc.restore_features()
        features = tc.encode(inputs)
        # Feature 0 should not necessarily be zero after restore
        # (depends on weights, but ablation should be cleared)
        assert tc._ablated_features == []

    def test_last_features_stored(self, random_pairs):
        inputs, _ = random_pairs
        tc = Transcoder(d_input=32, d_output=32, n_features=64)
        tc.encode(inputs)
        assert tc.last_features is not None
        assert tc.last_features.shape == (16, 10, 64)


class TestAttachDetach:
    def test_attach_changes_output(self, small_model):
        tc = Transcoder(d_input=2, d_output=2, n_features=8)
        baseline, _ = small_model.run_with_cache(text="Hello world")

        tc.attach(small_model, layer=0)
        modified, _ = small_model.run_with_cache(text="Hello world")
        tc.detach(small_model, layer=0)

        assert not torch.allclose(baseline, modified)

    def test_detach_restores_output(self, small_model):
        tc = Transcoder(d_input=2, d_output=2, n_features=8)
        baseline, _ = small_model.run_with_cache(text="Hello world")

        tc.attach(small_model, layer=0)
        tc.detach(small_model, layer=0)
        restored, _ = small_model.run_with_cache(text="Hello world")

        assert torch.allclose(baseline, restored)

    def test_context_manager(self, small_model):
        tc = Transcoder(d_input=2, d_output=2, n_features=8)
        baseline, _ = small_model.run_with_cache(text="Hello world")

        with tc.attached(small_model, layer=0):
            modified, _ = small_model.run_with_cache(text="Hello world")

        restored, _ = small_model.run_with_cache(text="Hello world")

        assert not torch.allclose(baseline, modified)
        assert torch.allclose(baseline, restored)

    def test_ablate_while_attached(self, small_model):
        tc = Transcoder(d_input=2, d_output=2, n_features=8)

        with tc.attached(small_model, layer=0):
            logits_normal, _ = small_model.run_with_cache(text="Hello world")
            tc.ablate_features([0, 1, 2])
            logits_ablated, _ = small_model.run_with_cache(text="Hello world")
            tc.restore_features()

        assert not torch.allclose(logits_normal, logits_ablated)


class TestFit:
    def test_fit_with_tensor(self, small_model):
        tc = Transcoder(d_input=8, d_output=2, n_features=16)
        torch.manual_seed(42)
        fake_in = torch.randn(100, 8)
        fake_out = torch.randn(100, 2)
        logs = tc.fit_on_activations(fake_in, fake_out, n_steps=10, log_every=5, batch_size=16)
        assert len(logs) == 2

    def test_fit_with_strings(self, small_model):
        tc = Transcoder(d_input=2, d_output=8, n_features=16)
        texts = ["Hello world", "The cat sat"]
        logs = tc.fit(
            small_model,
            input_point="transformer.h.0.ln_1",
            output_point="transformer.h.0.mlp.c_fc",
            dataset=texts,
            n_steps=5,
            log_every=5,
            batch_size=8,
        )
        assert len(logs) == 1


class TestSaveLoad:
    def test_save_and_load(self, tmp_path, random_pairs):
        inputs, outputs = random_pairs
        tc = Transcoder(d_input=32, d_output=32, n_features=64, activation="topk", k=8)
        result_before = tc(inputs, outputs)

        tc.save(tmp_path / "my_tc")
        loaded = Transcoder.load(tmp_path / "my_tc")
        result_after = loaded(inputs, outputs)

        assert torch.allclose(result_before.features, result_after.features)
        assert torch.allclose(result_before.prediction, result_after.prediction)

    def test_save_load_skip(self, tmp_path, random_pairs):
        inputs, outputs = random_pairs
        tc = Transcoder(d_input=32, d_output=32, n_features=64, skip=True)
        result_before = tc(inputs, outputs)

        tc.save(tmp_path / "skip_tc")
        loaded = Transcoder.load(tmp_path / "skip_tc")
        result_after = loaded(inputs, outputs)

        assert torch.allclose(result_before.prediction, result_after.prediction)


class TestRepr:
    def test_basic_repr(self):
        r = repr(Transcoder(d_input=32, d_output=32, n_features=64))
        assert "Transcoder" in r

    def test_skip_repr(self):
        r = repr(Transcoder(d_input=32, d_output=32, n_features=64, skip=True))
        assert "skip=True" in r

    def test_output_layers_repr(self):
        r = repr(Transcoder(d_input=32, d_output=32, n_features=64, output_layers=[0, 1, 2]))
        assert "output_layers" in r
