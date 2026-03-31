import torch
import pytest
from torch_engression.models import StoNet, StoLayer, StoResBlock


class TestStoLayer:
    def test_output_shape(self):
        layer = StoLayer(5, 10, noise_dim=20)
        x = torch.randn(32, 5)
        y = layer(x)
        assert y.shape == (32, 10)

    def test_stochastic(self):
        """Two forward passes produce different outputs."""
        layer = StoLayer(5, 10, noise_dim=20)
        x = torch.randn(32, 5)
        y1 = layer(x)
        y2 = layer(x)
        assert not torch.allclose(y1, y2)

    def test_unconditional(self):
        """in_dim=0 generates unconditionally from noise."""
        layer = StoLayer(0, 10, noise_dim=20)
        y = layer(32)  # Pass batch size as int
        assert y.shape == (32, 10)


class TestStoResBlock:
    def test_same_dim(self):
        block = StoResBlock(dim=50, noise_dim=20)
        x = torch.randn(32, 50)
        y = block(x)
        assert y.shape == (32, 50)

    def test_different_dim(self):
        block = StoResBlock(dim=50, out_dim=30, noise_dim=20)
        x = torch.randn(32, 50)
        y = block(x)
        assert y.shape == (32, 30)


class TestStoNet:
    def test_basic_forward(self):
        model = StoNet(5, 1, num_layer=3, hidden_dim=32, noise_dim=16)
        x = torch.randn(64, 5)
        y = model(x)
        assert y.shape == (64, 1)

    def test_multivariate_output(self):
        model = StoNet(5, 3, num_layer=2, hidden_dim=32, noise_dim=16)
        x = torch.randn(64, 5)
        y = model(x)
        assert y.shape == (64, 3)

    def test_resblock(self):
        model = StoNet(5, 1, num_layer=4, resblock=True, hidden_dim=32, noise_dim=16)
        x = torch.randn(64, 5)
        y = model(x)
        assert y.shape == (64, 1)

    def test_sample_shape_expanded(self):
        model = StoNet(5, 2, num_layer=2, hidden_dim=32, noise_dim=16)
        x = torch.randn(10, 5)
        samples = model.sample(x, sample_size=20, expand_dim=True, verbose=False)
        assert samples.shape == (10, 2, 20)

    def test_sample_shape_flat(self):
        model = StoNet(5, 2, num_layer=2, hidden_dim=32, noise_dim=16)
        x = torch.randn(10, 5)
        samples = model.sample(x, sample_size=20, expand_dim=False, verbose=False)
        assert samples.shape == (200, 2)

    def test_predict_mean(self):
        model = StoNet(5, 1, num_layer=2, hidden_dim=32, noise_dim=16)
        x = torch.randn(10, 5)
        pred = model.predict(x, target="mean", sample_size=50)
        assert pred.shape == (10, 1)

    def test_predict_quantile(self):
        model = StoNet(5, 1, num_layer=2, hidden_dim=32, noise_dim=16)
        x = torch.randn(10, 5)
        pred = model.predict(x, target=0.5, sample_size=50)
        assert pred.shape == (10, 1)

    def test_compute_cdf(self):
        model = StoNet(5, 1, num_layer=2, hidden_dim=32, noise_dim=16)
        x = torch.randn(10, 5)
        y = torch.zeros(10, 1)
        cdf = model.compute_cdf(x, y, sample_size=50)
        assert cdf.shape == (10, 1)
        assert (cdf >= 0).all() and (cdf <= 1).all()

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        model = StoNet(5, 1, num_layer=2, hidden_dim=32, noise_dim=16).cuda()
        x = torch.randn(32, 5, device="cuda")
        y = model(x)
        assert y.device.type == "cuda"
        assert y.shape == (32, 1)

    def test_no_noise_all_layer(self):
        """noise_all_layer=False only injects noise at first layer."""
        model = StoNet(5, 1, num_layer=3, noise_all_layer=False, hidden_dim=32, noise_dim=16)
        x = torch.randn(32, 5)
        y = model(x)
        assert y.shape == (32, 1)
