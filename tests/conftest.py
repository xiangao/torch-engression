import pytest
import torch


@pytest.fixture(scope="session")
def seed_torch():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def simple_1d_data():
    """Simple 1D regression: y = x^2 + noise."""
    torch.manual_seed(42)
    n = 500
    x = torch.randn(n, 1)
    y = x ** 2 + 0.3 * torch.randn(n, 1)
    return x, y


@pytest.fixture
def multivariate_data():
    """Multivariate regression: y = [x1^2, x2]."""
    torch.manual_seed(42)
    n = 500
    x = torch.randn(n, 3)
    y = torch.cat([x[:, 0:1] ** 2, x[:, 1:2]], dim=1) + 0.2 * torch.randn(n, 2)
    return x, y


@pytest.fixture
def classification_data():
    """Binary classification data."""
    torch.manual_seed(42)
    n = 500
    x = torch.randn(n, 2)
    logits = x[:, 0:1] + 0.5 * x[:, 1:2]
    probs = torch.sigmoid(logits)
    y = torch.bernoulli(probs)
    return x, y
