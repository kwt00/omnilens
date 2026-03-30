"""Tests for SteeringVector — directional interventions on activations."""

import pytest
import torch
from omnilens import TappedModel, SteeringVector, SAE, Probe


@pytest.fixture
def small_model():
    return TappedModel.from_pretrained(
        "sshleifer/tiny-gpt2", dtype=torch.float32
    )


class TestFromContrastive:
    def test_basic(self, small_model):
        vec = SteeringVector.from_contrastive(
            small_model,
            hook_point="transformer.h.0.ln_1",
            positive=["I love this", "This is great"],
            negative=["I hate this", "This is terrible"],
        )
        assert vec.direction.ndim == 1
        assert vec.direction.shape[0] == 2  # tiny-gpt2 d_model=2
        assert vec.hook_point == "transformer.h.0.ln_1"

    def test_nonzero_direction(self, small_model):
        vec = SteeringVector.from_contrastive(
            small_model,
            hook_point="transformer.h.0.ln_1",
            positive=["Hello world"],
            negative=["Goodbye world"],
        )
        assert vec.direction.norm() > 0


class TestFromPair:
    def test_basic(self, small_model):
        vec = SteeringVector.from_pair(
            small_model,
            hook_point="transformer.h.0.ln_1",
            positive="I love this",
            negative="I hate this",
        )
        assert vec.direction.ndim == 1

    def test_matches_contrastive_single(self, small_model):
        vec_pair = SteeringVector.from_pair(
            small_model,
            hook_point="transformer.h.0.ln_1",
            positive="Hello",
            negative="Goodbye",
        )
        vec_contrastive = SteeringVector.from_contrastive(
            small_model,
            hook_point="transformer.h.0.ln_1",
            positive=["Hello"],
            negative=["Goodbye"],
        )
        assert torch.allclose(vec_pair.direction, vec_contrastive.direction)


class TestFromProbe:
    def test_basic(self):
        probe = Probe(d_model=16, n_classes=3)
        vec = SteeringVector.from_probe(
            probe, hook_point="layers.0.residual.block_out", class_idx=2
        )
        assert vec.direction.ndim == 1
        assert vec.direction.shape[0] == 16

    def test_class_idx(self):
        probe = Probe(d_model=16, n_classes=3)
        vec0 = SteeringVector.from_probe(probe, hook_point="test", class_idx=0)
        vec1 = SteeringVector.from_probe(probe, hook_point="test", class_idx=1)
        assert not torch.allclose(vec0.direction, vec1.direction)


class TestFromSAEFeature:
    def test_basic(self):
        sae = SAE(d_model=16, n_features=64)
        vec = SteeringVector.from_sae_feature(
            sae, feature=42, hook_point="layers.0.residual.block_out"
        )
        assert vec.direction.ndim == 1
        assert vec.direction.shape[0] == 16

    def test_different_features_different_directions(self):
        sae = SAE(d_model=16, n_features=64)
        vec0 = SteeringVector.from_sae_feature(sae, feature=0, hook_point="test")
        vec1 = SteeringVector.from_sae_feature(sae, feature=1, hook_point="test")
        assert not torch.allclose(vec0.direction, vec1.direction)

    def test_tied_weights(self):
        sae = SAE(d_model=16, n_features=64, tied_weights=True)
        vec = SteeringVector.from_sae_feature(
            sae, feature=10, hook_point="test"
        )
        assert vec.direction.shape[0] == 16


class TestFromRawTensor:
    def test_basic(self):
        direction = torch.randn(32)
        vec = SteeringVector(direction=direction, hook_point="layers.5.residual.block_out")
        assert torch.allclose(vec.direction, direction)


class TestHookApplication:
    def test_hook_changes_output(self, small_model):
        # Use a large random vector to guarantee a measurable difference
        vec = SteeringVector(
            direction=torch.randn(2) * 10.0,
            hook_point="transformer.h.0.ln_1",
        )
        baseline, _ = small_model.run_with_cache(text="Test input")
        modified = small_model.run_with_hooks(
            text="Test input", hooks=vec.hook(scale=1.0)
        )
        assert not torch.allclose(baseline, modified)

    def test_scale_zero_no_change(self, small_model):
        vec = SteeringVector.from_pair(
            small_model,
            hook_point="transformer.h.0.ln_1",
            positive="Hello",
            negative="Goodbye",
        )
        baseline, _ = small_model.run_with_cache(text="Test input")
        modified = small_model.run_with_hooks(
            text="Test input", hooks=vec.hook(scale=0.0)
        )
        assert torch.allclose(baseline, modified)

    def test_negative_scale(self, small_model):
        vec = SteeringVector.from_pair(
            small_model,
            hook_point="transformer.h.0.ln_1",
            positive="Hello",
            negative="Goodbye",
        )
        pos_logits = small_model.run_with_hooks(
            text="Test", hooks=vec.hook(scale=10.0)
        )
        neg_logits = small_model.run_with_hooks(
            text="Test", hooks=vec.hook(scale=-10.0)
        )
        assert not torch.allclose(pos_logits, neg_logits)

    def test_hooks_multiple_layers(self, small_model):
        vec = SteeringVector(
            direction=torch.randn(2),
            hook_point="layers.0.residual.block_out",
        )
        baseline, _ = small_model.run_with_cache(text="Test")
        modified = small_model.run_with_hooks(
            text="Test", hooks=vec.hooks(layers=[0, 1], scale=10.0)
        )
        assert not torch.allclose(baseline, modified)

    def test_hooks_custom_template(self, small_model):
        vec = SteeringVector(
            direction=torch.randn(2),
            hook_point="layers.0.residual.block_out",
        )
        hooks = vec.hooks(
            layers=[0, 1],
            hook_template="layers.{i}.residual.block_out",
        )
        assert "layers.0.residual.block_out" in hooks
        assert "layers.1.residual.block_out" in hooks


class TestUtilities:
    def test_normalize(self):
        vec = SteeringVector(direction=torch.tensor([3.0, 4.0]), hook_point="test")
        normed = vec.normalize()
        assert abs(normed.direction.norm().item() - 1.0) < 1e-6

    def test_cosine_similarity_identical(self):
        vec = SteeringVector(direction=torch.tensor([1.0, 0.0]), hook_point="test")
        assert abs(vec.cosine_similarity(vec) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        vec1 = SteeringVector(direction=torch.tensor([1.0, 0.0]), hook_point="test")
        vec2 = SteeringVector(direction=torch.tensor([0.0, 1.0]), hook_point="test")
        assert abs(vec1.cosine_similarity(vec2)) < 1e-6

    def test_cosine_similarity_opposite(self):
        vec1 = SteeringVector(direction=torch.tensor([1.0, 0.0]), hook_point="test")
        vec2 = SteeringVector(direction=torch.tensor([-1.0, 0.0]), hook_point="test")
        assert abs(vec1.cosine_similarity(vec2) + 1.0) < 1e-6


class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        vec = SteeringVector(
            direction=torch.randn(32),
            hook_point="layers.5.residual.block_out",
        )
        vec.save(tmp_path / "my_vec")
        loaded = SteeringVector.load(tmp_path / "my_vec")

        assert torch.allclose(vec.direction, loaded.direction)
        assert vec.hook_point == loaded.hook_point


class TestRepr:
    def test_repr(self):
        vec = SteeringVector(direction=torch.randn(32), hook_point="layers.5.residual.block_out")
        r = repr(vec)
        assert "SteeringVector" in r
        assert "dims=32" in r
        assert "layers.5" in r
