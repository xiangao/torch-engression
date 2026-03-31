import torch
import torch.nn as nn
import numpy as np


def preanm_simulator(true_function="softplus", n=10000, x_lower=0, x_upper=2,
                     noise_std=1, noise_dist="gaussian", train=True, device=None):
    """Simulate data from a pre-additive noise model (pre-ANM).

    Generates Y = g*(X + eps), where g* is a known function and eps is noise.
    Useful for testing engression models.

    Args:
        true_function (str or callable): true function g*. Default "softplus".
            Choices: "softplus", "cubic", "square", "log", or a callable.
        n (int): sample size. Default 10000.
        x_lower (float): lower bound of training support. Default 0.
        x_upper (float): upper bound of training support. Default 2.
        noise_std (float): noise standard deviation. Default 1.
        noise_dist (str): noise distribution. "gaussian" or "uniform". Default "gaussian".
        train (bool): if True, return (x, y). If False, return (x, y_median, y_mean).
        device: torch device. Default None (auto-detect).

    Returns:
        tuple of torch.Tensors.
    """
    from ..utils import auto_device

    if device is None:
        device = torch.device("cpu")  # Simulator runs on CPU by default
    elif isinstance(device, str):
        device = torch.device(device)

    if isinstance(true_function, str):
        if true_function == "softplus":
            true_function = lambda x: nn.Softplus()(x)
        elif true_function == "cubic":
            true_function = lambda x: x.pow(3) / 3
        elif true_function == "square":
            true_function = lambda x: (nn.functional.relu(x)).pow(2) / 2
        elif true_function == "log":
            true_function = lambda x: (x / 3 + np.log(3) - 2 / 3) * (x <= 2) + (torch.log(1 + x * (x > 2))) * (x > 2)

    if train:
        x = torch.rand(n, 1) * (x_upper - x_lower) + x_lower
        if noise_dist == "gaussian":
            eps = torch.randn(n, 1) * noise_std
        else:
            eps = (torch.rand(n, 1) - 0.5) * noise_std * np.sqrt(12)
        xn = x + eps
        y = true_function(xn)
        return x.to(device), y.to(device)
    else:
        x_eval = torch.linspace(x_lower, x_upper, n).unsqueeze(1)
        y_eval_med = true_function(x_eval)
        gen_sample_size = 10000
        x_rep = torch.repeat_interleave(x_eval, (gen_sample_size * torch.ones(n)).long(), dim=0)
        x_rep = x_rep + torch.randn(x_rep.size(0), 1) * noise_std
        y_eval_mean = true_function(x_rep)
        y_eval_mean = list(torch.split(y_eval_mean, gen_sample_size))
        y_eval_mean = torch.cat([y_eval_mean[i].mean().unsqueeze(0) for i in range(n)], dim=0).unsqueeze(1)
        return x_eval.to(device), y_eval_med.to(device), y_eval_mean.to(device)
