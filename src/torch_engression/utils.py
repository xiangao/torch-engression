import os
import warnings

import torch


def auto_device(device=None):
    """Auto-detect the best available device.

    Priority: CUDA > MPS > CPU. Accepts string shortcuts like "gpu".

    Args:
        device: None (auto-detect), str ("cpu", "cuda", "gpu", "mps", "cuda:1"),
                or torch.device.

    Returns:
        torch.device
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            warnings.warn(
                "MPS detected. Some operations (torch.cdist, quantile) may have "
                "numerical issues on MPS. Consider using CPU if results look wrong."
            )
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(device, str):
        if device == "gpu":
            device = "cuda"
        return torch.device(device)
    return device


def vectorize(x):
    """Ensure tensor is 2D: (n, d).

    Args:
        x (torch.Tensor): input data of any shape.

    Returns:
        torch.Tensor: data of shape (n, d).
    """
    if len(x.shape) == 1:
        return x.unsqueeze(1)
    if len(x.shape) == 2:
        return x
    return x.reshape(x.shape[0], -1)


def cor(x, y):
    """Pearson correlation between two tensors.

    Args:
        x (torch.Tensor): input data.
        y (torch.Tensor): input data.

    Returns:
        torch.Tensor: scalar correlation.
    """
    x = vectorize(x)
    y = vectorize(y)
    x = x - x.mean(0)
    y = y - y.mean(0)
    return ((x * y).mean()) / (x.std(unbiased=False) * y.std(unbiased=False))


def make_folder(name):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(name):
        os.makedirs(name)


def set_seed(seed):
    """Set random seeds for reproducibility.

    Args:
        seed (int): random seed.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
