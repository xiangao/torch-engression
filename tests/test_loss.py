import torch
import pytest
from torch_engression.loss import energy_loss_two_sample, energy_loss


class TestEnergyLossTwoSample:
    """Tests for the two-sample energy loss used in training."""

    def test_identical_samples_zero_variance(self):
        """If x == xp, the variance term s2 should be zero."""
        x0 = torch.randn(50, 2)
        x = torch.randn(50, 2)
        loss, s1, s2 = energy_loss_two_sample(x0, x, x, verbose=True)
        assert s2.item() == pytest.approx(0.0, abs=1e-6)

    def test_perfect_match_zero_loss(self):
        """If model samples equal true samples, s1 == s2/2 so loss ≈ 0."""
        x0 = torch.randn(50, 2)
        loss = energy_loss_two_sample(x0, x0, x0, verbose=False)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_shape(self):
        """Verbose returns 3-element tensor, non-verbose returns scalar."""
        x0 = torch.randn(50, 2)
        x = torch.randn(50, 2)
        xp = torch.randn(50, 2)
        result_v = energy_loss_two_sample(x0, x, xp, verbose=True)
        result_s = energy_loss_two_sample(x0, x, xp, verbose=False)
        assert result_v.shape == (3,)
        assert result_s.shape == ()

    def test_1d_input(self):
        """Works with 1D tensors (auto-vectorized)."""
        x0 = torch.randn(50)
        x = torch.randn(50)
        xp = torch.randn(50)
        loss = energy_loss_two_sample(x0, x, xp, verbose=False)
        assert loss.shape == ()

    def test_beta_parameter(self):
        """Non-integer beta uses epsilon for stability."""
        x0 = torch.randn(50, 2)
        x = torch.randn(50, 2)
        xp = torch.randn(50, 2)
        loss_b1 = energy_loss_two_sample(x0, x, xp, beta=1, verbose=False)
        loss_b05 = energy_loss_two_sample(x0, x, xp, beta=0.5, verbose=False)
        # Both should be finite
        assert torch.isfinite(loss_b1)
        assert torch.isfinite(loss_b05)

    def test_weighted(self):
        """Observation weights are applied."""
        x0 = torch.randn(50, 2)
        x = torch.randn(50, 2)
        xp = torch.randn(50, 2)
        w = torch.ones(50) / 50
        loss_uniform = energy_loss_two_sample(x0, x, xp, weights=w, verbose=False)
        loss_default = energy_loss_two_sample(x0, x, xp, verbose=False)
        assert loss_uniform.item() == pytest.approx(loss_default.item(), rel=1e-5)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Works on CUDA."""
        x0 = torch.randn(50, 2, device="cuda")
        x = torch.randn(50, 2, device="cuda")
        xp = torch.randn(50, 2, device="cuda")
        loss = energy_loss_two_sample(x0, x, xp, verbose=False)
        assert loss.device.type == "cuda"
        assert torch.isfinite(loss)


class TestEnergyLoss:
    """Tests for the multi-sample energy loss used in evaluation."""

    def test_basic(self):
        """Basic energy loss computation."""
        x_true = torch.randn(50, 2)
        x_est = torch.randn(100, 2)  # 2 samples of 50
        result = energy_loss(x_true, x_est, verbose=True)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()

    def test_list_input(self):
        """Accepts list of sample tensors."""
        x_true = torch.randn(50, 2)
        x_est = [torch.randn(50, 2) for _ in range(5)]
        result = energy_loss(x_true, x_est, verbose=False)
        assert torch.isfinite(result)
