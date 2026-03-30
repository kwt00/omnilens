"""Tests for Probe — linear/MLP probes on cached activations."""

import pytest
import torch
from omnilens import TappedModel, Probe


@pytest.fixture
def small_model():
    return TappedModel.from_pretrained(
        "sshleifer/tiny-gpt2", dtype=torch.float32
    )


class TestConstructor:
    def test_linear_classification(self):
        probe = Probe(d_model=32, n_classes=3)
        assert probe.task == "classification"
        assert isinstance(probe.network, torch.nn.Linear)

    def test_linear_regression(self):
        probe = Probe(d_model=32, task="regression")
        assert probe.task == "regression"

    def test_n_classes_1_infers_regression(self):
        probe = Probe(d_model=32, n_classes=1)
        assert probe.task == "regression"

    def test_mlp_probe(self):
        probe = Probe(d_model=32, n_classes=3, hidden_dims=[16])
        assert isinstance(probe.network, torch.nn.Sequential)

    def test_deep_mlp_probe(self):
        probe = Probe(d_model=32, n_classes=3, hidden_dims=[64, 16])
        x = torch.randn(4, 32)
        out = probe(x)
        assert out.shape == (4, 3)

    def test_custom_loss_fn(self):
        probe = Probe(d_model=32, n_classes=3, loss_fn=torch.nn.MSELoss())
        assert isinstance(probe.loss_fn, torch.nn.MSELoss)


class TestForwardPass:
    def test_classification_output_shape(self):
        probe = Probe(d_model=32, n_classes=5)
        x = torch.randn(8, 32)
        out = probe(x)
        assert out.shape == (8, 5)

    def test_regression_output_shape(self):
        probe = Probe(d_model=32, task="regression")
        x = torch.randn(8, 32)
        out = probe(x)
        assert out.shape == (8, 1)

    def test_gradient_flow(self):
        probe = Probe(d_model=32, n_classes=3)
        x = torch.randn(8, 32)
        labels = torch.randint(0, 3, (8,))
        logits = probe(x)
        loss = probe.loss_fn(logits, labels)
        loss.backward()
        assert probe.network.weight.grad is not None


class TestFitClassification:
    def test_fit_on_activations(self):
        torch.manual_seed(42)
        # Create separable data
        acts_class0 = torch.randn(50, 16) + torch.tensor([1.0] * 16)
        acts_class1 = torch.randn(50, 16) + torch.tensor([-1.0] * 16)
        acts = torch.cat([acts_class0, acts_class1])
        labels = [0] * 50 + [1] * 50

        probe = Probe(d_model=16, n_classes=2)
        result = probe.fit_on_activations(acts, labels, n_epochs=100, batch_size=32)

        assert "accuracy" in result
        assert probe.accuracy is not None
        assert probe.accuracy > 0.7  # should easily separate

    def test_fit_with_model(self, small_model):
        texts = ["Hello world"] * 10 + ["Goodbye world"] * 10
        labels = [0] * 10 + [1] * 10

        probe = Probe(d_model=2, n_classes=2)
        result = probe.fit(
            small_model,
            hook_point="transformer.h.0.ln_1",
            texts=texts,
            labels=labels,
            n_epochs=10,
        )
        assert "accuracy" in result

    def test_specific_position(self, small_model):
        texts = ["Hello world"] * 10 + ["Goodbye world"] * 10
        labels = [0] * 10 + [1] * 10

        probe = Probe(d_model=2, n_classes=2)
        result = probe.fit(
            small_model,
            hook_point="transformer.h.0.ln_1",
            texts=texts,
            labels=labels,
            position=0,
            n_epochs=10,
        )
        assert "accuracy" in result


class TestFitRegression:
    def test_regression_on_activations(self):
        torch.manual_seed(42)
        acts = torch.randn(100, 16)
        labels = acts[:, 0]  # predict first feature

        probe = Probe(d_model=16, task="regression")
        result = probe.fit_on_activations(acts, labels, n_epochs=100, batch_size=32)

        assert "val_loss" in result
        assert probe.accuracy is None  # no accuracy for regression
        assert probe.train_loss is not None


class TestSweep:
    def test_sweep_layers(self, small_model):
        texts = ["Hello world"] * 10 + ["Goodbye world"] * 10
        labels = [0] * 10 + [1] * 10

        results = Probe.sweep(
            small_model,
            hook_points="layers.{i}.residual.block_out",
            texts=texts,
            labels=labels,
            n_classes=2,
            n_epochs=10,
        )

        assert len(results.accuracies) == 2  # tiny-gpt2 has 2 layers
        assert len(results.probes) == 2
        for name, acc in results.accuracies.items():
            assert 0.0 <= acc <= 1.0

    def test_sweep_specific_points(self, small_model):
        texts = ["Hello world"] * 10 + ["Goodbye world"] * 10
        labels = [0] * 10 + [1] * 10

        results = Probe.sweep(
            small_model,
            hook_points=[
                "transformer.h.0.ln_1",
                "transformer.h.1.ln_1",
            ],
            texts=texts,
            labels=labels,
            n_classes=2,
            n_epochs=10,
        )

        assert len(results.accuracies) == 2


class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        torch.manual_seed(42)
        acts = torch.randn(50, 16) + 1.0
        acts2 = torch.randn(50, 16) - 1.0
        all_acts = torch.cat([acts, acts2])
        labels = [0] * 50 + [1] * 50

        probe = Probe(d_model=16, n_classes=2)
        probe.fit_on_activations(all_acts, labels, n_epochs=50)

        probe.save(tmp_path / "my_probe")
        loaded = Probe.load(tmp_path / "my_probe")

        assert loaded.accuracy == probe.accuracy
        assert loaded.task == probe.task

        # Same predictions
        test_input = torch.randn(4, 16)
        assert torch.allclose(probe(test_input), loaded(test_input))


class TestRepr:
    def test_classification_repr(self):
        probe = Probe(d_model=32, n_classes=3)
        r = repr(probe)
        assert "Probe" in r
        assert "classification" in r
        assert "n_classes=3" in r

    def test_regression_repr(self):
        probe = Probe(d_model=32, task="regression")
        r = repr(probe)
        assert "regression" in r

    def test_accuracy_in_repr(self):
        probe = Probe(d_model=32, n_classes=2)
        probe.accuracy = 0.95
        r = repr(probe)
        assert "0.950" in r

    def test_hidden_dims_in_repr(self):
        probe = Probe(d_model=32, n_classes=2, hidden_dims=[16])
        r = repr(probe)
        assert "hidden_dims" in r
