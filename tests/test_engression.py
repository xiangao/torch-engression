import os
import tempfile

import torch
import pytest
from torch_engression import engression, Engressor
from torch_engression.utils import auto_device


class TestAutoDevice:
    def test_none_auto_detects(self):
        dev = auto_device(None)
        assert isinstance(dev, torch.device)

    def test_string_cpu(self):
        assert auto_device("cpu") == torch.device("cpu")

    def test_string_gpu_alias(self):
        assert auto_device("gpu") == torch.device("cuda")

    def test_string_cuda(self):
        assert auto_device("cuda") == torch.device("cuda")

    def test_device_passthrough(self):
        dev = torch.device("cpu")
        assert auto_device(dev) is dev


class TestEngression:
    """End-to-end tests for the engression() convenience function."""

    def test_basic_fit(self, simple_1d_data):
        x, y = simple_1d_data
        model = engression(x, y, num_epochs=10, hidden_dim=32,
                           noise_dim=16, device="cpu", verbose=False)
        assert model.tr_loss is not None
        assert len(model.tr_loss) == 3

    def test_predict_mean(self, simple_1d_data):
        x, y = simple_1d_data
        model = engression(x, y, num_epochs=10, hidden_dim=32,
                           noise_dim=16, device="cpu", verbose=False)
        pred = model.predict(x[:10], target="mean")
        assert pred.shape == (10, 1)

    def test_predict_quantiles(self, simple_1d_data):
        x, y = simple_1d_data
        model = engression(x, y, num_epochs=10, hidden_dim=32,
                           noise_dim=16, device="cpu", verbose=False)
        q10, q50, q90 = model.predict(x[:10], target=[0.1, 0.5, 0.9],
                                       sample_size=200)
        # Quantiles should be ordered
        assert (q10 <= q90).all()

    def test_sample(self, simple_1d_data):
        x, y = simple_1d_data
        model = engression(x, y, num_epochs=10, hidden_dim=32,
                           noise_dim=16, device="cpu", verbose=False)
        samples = model.sample(x[:5], sample_size=50)
        assert samples.shape == (5, 1, 50)

    def test_eval_loss_types(self, simple_1d_data):
        x, y = simple_1d_data
        model = engression(x, y, num_epochs=10, hidden_dim=32,
                           noise_dim=16, device="cpu", verbose=False)
        l2 = model.eval_loss(x[:50], y[:50], loss_type="l2")
        l1 = model.eval_loss(x[:50], y[:50], loss_type="l1")
        energy = model.eval_loss(x[:50], y[:50], loss_type="energy")
        cor_val = model.eval_loss(x[:50], y[:50], loss_type="cor")
        assert isinstance(l2, float)
        assert isinstance(l1, float)
        assert isinstance(energy, float)
        assert isinstance(cor_val, float)

    def test_no_standardize(self, simple_1d_data):
        x, y = simple_1d_data
        model = engression(x, y, num_epochs=5, hidden_dim=32,
                           noise_dim=16, device="cpu", verbose=False,
                           standardize=False)
        pred = model.predict(x[:5])
        assert pred.shape == (5, 1)

    def test_multivariate(self, multivariate_data):
        x, y = multivariate_data
        model = engression(x, y, num_epochs=5, hidden_dim=32,
                           noise_dim=16, device="cpu", verbose=False)
        pred = model.predict(x[:5])
        assert pred.shape == (5, 2)

    def test_mini_batch(self, simple_1d_data):
        x, y = simple_1d_data
        model = engression(x, y, num_epochs=5, hidden_dim=32,
                           noise_dim=16, batch_size=64, device="cpu",
                           verbose=False)
        pred = model.predict(x[:5])
        assert pred.shape == (5, 1)

    def test_resblock(self, simple_1d_data):
        x, y = simple_1d_data
        model = engression(x, y, num_epochs=5, hidden_dim=32,
                           noise_dim=16, resblock=True, num_layer=4,
                           device="cpu", verbose=False)
        pred = model.predict(x[:5])
        assert pred.shape == (5, 1)

    def test_seed_reproducibility(self, simple_1d_data):
        x, y = simple_1d_data
        m1 = engression(x, y, num_epochs=5, hidden_dim=32, noise_dim=16,
                        device="cpu", verbose=False, seed=123)
        m2 = engression(x, y, num_epochs=5, hidden_dim=32, noise_dim=16,
                        device="cpu", verbose=False, seed=123)
        # Same seed -> same model weights -> same training loss
        assert m1.tr_loss[0] == pytest.approx(m2.tr_loss[0], rel=1e-3)


class TestSaveLoad:
    def test_roundtrip(self, simple_1d_data):
        x, y = simple_1d_data
        model = engression(x, y, num_epochs=5, hidden_dim=32,
                           noise_dim=16, device="cpu", verbose=False, seed=42)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = Engressor.load(path, device="cpu")
            # Hyperparams match
            assert loaded.num_layer == model.num_layer
            assert loaded.hidden_dim == model.hidden_dim
            assert loaded.noise_dim == model.noise_dim
            # Standardization stats match
            assert torch.allclose(loaded.x_mean, model.x_mean)
            assert torch.allclose(loaded.x_std, model.x_std)
            # Training loss preserved
            assert loaded.tr_loss[0] == pytest.approx(model.tr_loss[0])
        finally:
            os.unlink(path)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_save_gpu_load_cpu(self, simple_1d_data):
        x, y = simple_1d_data
        model = engression(x, y, num_epochs=5, hidden_dim=32,
                           noise_dim=16, device="cuda", verbose=False, seed=42)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = Engressor.load(path, device="cpu")
            assert loaded.device == torch.device("cpu")
            pred = loaded.predict(x[:5])
            assert pred.device == torch.device("cpu")
        finally:
            os.unlink(path)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPU:
    def test_auto_detect_gpu(self, simple_1d_data):
        x, y = simple_1d_data
        model = engression(x, y, num_epochs=5, hidden_dim=32,
                           noise_dim=16, verbose=False)
        assert model.device.type == "cuda"

    def test_amp_training(self, simple_1d_data):
        """AMP training converges (no NaN losses)."""
        x, y = simple_1d_data
        model = engression(x, y, num_epochs=20, hidden_dim=64,
                           noise_dim=32, verbose=False)
        assert model.tr_loss[0] > 0  # Loss is positive
        assert not any(v != v for v in model.tr_loss)  # No NaN

    def test_predict_on_gpu(self, simple_1d_data):
        x, y = simple_1d_data
        model = engression(x, y, num_epochs=5, hidden_dim=32,
                           noise_dim=16, verbose=False)
        pred = model.predict(x[:10])
        assert pred.device.type == "cuda"

    def test_sample_on_gpu(self, simple_1d_data):
        x, y = simple_1d_data
        model = engression(x, y, num_epochs=5, hidden_dim=32,
                           noise_dim=16, verbose=False)
        samples = model.sample(x[:10], sample_size=50)
        assert samples.device.type == "cuda"
        assert samples.shape == (10, 1, 50)
