import torch
from torch.linalg import vector_norm

from .utils import vectorize


def _compute_norm(tensor, p, dim):
    """Compute norm, with MPS fallback.

    Args:
        tensor (torch.Tensor): input tensor.
        p (float): norm order.
        dim (int): dimension along which to compute norm.

    Returns:
        torch.Tensor: computed norm.
    """
    if tensor.device.type == "mps":
        return torch.norm(tensor, p=p, dim=dim)
    return vector_norm(tensor, ord=p, dim=dim)


def energy_loss(x_true, x_est, beta=1, verbose=True):
    """Energy score loss for evaluation (multi-sample).

    Args:
        x_true (torch.Tensor): samples from true distribution, shape (n, d).
        x_est: list of tensors or stacked tensor (n*m, d) with m samples per point.
        beta (float): power parameter. Default 1.
        verbose (bool): if True, return (loss, s1, s2) tuple.

    Returns:
        torch.Tensor or tuple: energy loss, optionally with components.
    """
    EPS = 0 if float(beta).is_integer() else 1e-5
    x_true = vectorize(x_true).unsqueeze(1)
    if not isinstance(x_est, list):
        x_est = list(torch.split(x_est, x_true.shape[0], dim=0))
    m = len(x_est)
    x_est = [vectorize(x_est[i]).unsqueeze(1) for i in range(m)]
    x_est = torch.cat(x_est, dim=1)

    s1 = (_compute_norm(x_est - x_true, 2, dim=2) + EPS).pow(beta).mean()
    s2 = (torch.cdist(x_est, x_est, 2) + EPS).pow(beta).mean() * m / (m - 1)
    if verbose:
        return torch.cat([(s1 - s2 / 2).reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    return s1 - s2 / 2


def energy_loss_two_sample(x0, x, xp, x0p=None, beta=1, verbose=True, weights=None):
    """Energy score loss for training (two-sample estimator).

    More efficient than energy_loss — uses only two forward passes instead of m.

    Args:
        x0 (torch.Tensor): sample from true distribution.
        x (torch.Tensor): first sample from estimated distribution.
        xp (torch.Tensor): second sample from estimated distribution.
        x0p (torch.Tensor, optional): second sample from true distribution.
        beta (float): power parameter. Default 1.
        verbose (bool): if True, return (loss, s1, s2) tuple.
        weights (torch.Tensor, optional): observation weights.

    Returns:
        torch.Tensor or tuple: energy loss, optionally with components.
    """
    EPS = 0 if float(beta).is_integer() else 1e-5
    x0 = vectorize(x0)
    x = vectorize(x)
    xp = vectorize(xp)
    if weights is None:
        weights = 1 / x0.size(0)
    if x0p is None:
        s1 = (
            ((_compute_norm(x - x0, 2, dim=1) + EPS).pow(beta) * weights).sum() / 2
            + ((_compute_norm(xp - x0, 2, dim=1) + EPS).pow(beta) * weights).sum() / 2
        )
        s2 = ((_compute_norm(x - xp, 2, dim=1) + EPS).pow(beta) * weights).sum()
        loss = s1 - s2 / 2
    else:
        x0p = vectorize(x0p)
        s1 = (
            (_compute_norm(x - x0, 2, dim=1) + EPS).pow(beta).sum()
            + (_compute_norm(xp - x0, 2, dim=1) + EPS).pow(beta).sum()
            + (_compute_norm(x - x0p, 2, dim=1) + EPS).pow(beta).sum()
            + (_compute_norm(xp - x0p, 2, dim=1) + EPS).pow(beta).sum()
        ) / 4
        s2 = (_compute_norm(x - xp, 2, dim=1) + EPS).pow(beta).sum()
        s3 = (_compute_norm(x0 - x0p, 2, dim=1) + EPS).pow(beta).sum()
        loss = s1 - s2 / 2 - s3 / 2
    if verbose:
        return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    return loss
