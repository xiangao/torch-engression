import warnings

import torch
import torch.nn as nn

from .data.loader import make_dataloader


def get_act_func(name):
    """Get activation function by name."""
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "softmax":
        return nn.Softmax(dim=1)
    elif name == "elu":
        return nn.ELU(inplace=True)
    elif name == "softplus":
        return nn.Softplus()
    return None


class StoLayer(nn.Module):
    """Stochastic layer: concatenates random noise with input before linear transform.

    This is the building block of engression's generative architecture.
    Each forward pass produces a different output due to the injected noise.

    Args:
        in_dim (int): input dimension.
        out_dim (int): output dimension.
        noise_dim (int): noise dimension to concatenate. Default 100.
        add_bn (bool): add batch normalization. Default False.
        out_act (str): output activation. Default None.
        noise_std (float): std of injected noise. Default 1.
    """

    def __init__(self, in_dim, out_dim, noise_dim=100, add_bn=False,
                 out_act=None, noise_std=1, verbose=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        self.verbose = verbose

        layer = [nn.Linear(in_dim + noise_dim, out_dim)]
        if add_bn:
            layer += [nn.BatchNorm1d(out_dim)]
        self.layer = nn.Sequential(*layer)
        if out_act == "softmax" and out_dim == 1:
            out_act = "sigmoid"
        self.out_act = get_act_func(out_act)

    def forward(self, x):
        device = next(self.layer.parameters()).device
        if isinstance(x, int):
            # Unconditional generation: x is the batch size
            assert self.in_dim == 0
            out = torch.randn(x, self.noise_dim, device=device) * self.noise_std
        else:
            if x.size(1) < self.in_dim and self.verbose:
                print("Warning: covariate dimension does not match input dimension; "
                      "filling remaining with noise.")
            eps = torch.randn(
                x.size(0), self.noise_dim + self.in_dim - x.size(1), device=device
            ) * self.noise_std
            out = torch.cat([x, eps], dim=1)
        out = self.layer(out)
        if self.out_act is not None:
            out = self.out_act(out)
        return out


class StoResBlock(nn.Module):
    """Stochastic residual block with skip connection.

    Two-layer block with noise injection at each layer, plus a skip connection.

    Args:
        dim (int): input dimension. Default 100.
        hidden_dim (int): hidden dimension. Default None (= dim).
        out_dim (int): output dimension. Default None (= dim).
        noise_dim (int): noise dimension. Default 100.
        add_bn (bool): add batch normalization. Default False.
        out_act (str): output activation. Default None.
        noise_std (float): std of injected noise. Default 1.
    """

    def __init__(self, dim=100, hidden_dim=None, out_dim=None, noise_dim=100,
                 add_bn=False, out_act=None, noise_std=1):
        super().__init__()
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        if hidden_dim is None:
            hidden_dim = dim
        if out_dim is None:
            out_dim = dim

        self.layer1 = [nn.Linear(dim + noise_dim, hidden_dim)]
        if add_bn:
            self.layer1.append(nn.BatchNorm1d(hidden_dim))
        self.layer1.append(nn.ReLU())
        self.layer1 = nn.Sequential(*self.layer1)

        self.layer2 = nn.Linear(hidden_dim + noise_dim, out_dim)
        if add_bn and out_act == "relu":
            self.layer2 = nn.Sequential(*[self.layer2, nn.BatchNorm1d(out_dim)])

        if out_dim != dim:
            self.layer3 = nn.Linear(dim, out_dim)

        self.dim = dim
        self.out_dim = out_dim
        if out_act == "softmax" and out_dim == 1:
            out_act = "sigmoid"
        self.out_act = get_act_func(out_act)

    def forward(self, x):
        if self.noise_dim > 0:
            eps = torch.randn(x.size(0), self.noise_dim, device=x.device) * self.noise_std
            out = self.layer1(torch.cat([x, eps], dim=1))
            eps = torch.randn(x.size(0), self.noise_dim, device=x.device) * self.noise_std
            out = self.layer2(torch.cat([out, eps], dim=1))
        else:
            out = self.layer2(self.layer1(x))
        if self.out_dim != self.dim:
            out = out + self.layer3(x)
        else:
            out += x
        if self.out_act is not None:
            out = self.out_act(out)
        return out


class StoNetBase(nn.Module):
    """Base class for stochastic networks with sampling/prediction methods."""

    def __init__(self, forward_sampling=True):
        super().__init__()
        self.sampling_func = self.forward if forward_sampling else self.sampling_func

    @torch.no_grad()
    def predict(self, x, target=["mean"], sample_size=100):
        """Point prediction via Monte Carlo sampling.

        Args:
            x (torch.Tensor): input data.
            target: "mean", "median", float (quantile), or list of these.
            sample_size (int): number of samples per input. Default 100.

        Returns:
            torch.Tensor or list of torch.Tensor.
        """
        samples = self.sample(x=x, sample_size=sample_size, expand_dim=True)
        if not isinstance(target, list):
            target = [target]
        results = []
        extremes = []
        for t in target:
            if t == "mean":
                results.append(samples.mean(dim=-1))
            else:
                if t == "median":
                    t = 0.5
                assert isinstance(t, float)
                original_device = samples.device
                if original_device.type == "mps":
                    quantile_result = samples.cpu().quantile(t, dim=-1).to(original_device)
                else:
                    quantile_result = samples.quantile(t, dim=-1)
                results.append(quantile_result)
                if min(t, 1 - t) * sample_size < 10:
                    extremes.append(t)

        if len(extremes) > 0:
            print(f"Warning: quantile estimates at {extremes} with sample_size={sample_size} "
                  "could be inaccurate. Increase sample_size.")

        return results[0] if len(results) == 1 else results

    @torch.no_grad()
    def compute_cdf(self, x, y, sample_size=100):
        """Compute P(Y <= y | X=x) via Monte Carlo.

        Args:
            x (torch.Tensor): covariates (n, d_x).
            y (torch.Tensor): values at which to evaluate CDF (n, d_y).
            sample_size (int): number of samples. Default 100.

        Returns:
            torch.Tensor: estimated CDF values (n, d_y).
        """
        samples = self.sample(x=x, sample_size=sample_size, expand_dim=True)
        return (samples <= y.unsqueeze(-1)).float().mean(dim=-1)

    def sample_onebatch(self, x, sample_size=100, expand_dim=True,
                        require_grad=False, chunk_size=None):
        """Sample responses for one batch of inputs.

        Uses chunked sampling to control memory usage. Instead of creating
        x.repeat(sample_size, 1) all at once, processes in chunks.

        Args:
            x (torch.Tensor): input (n, d_x).
            sample_size (int): samples per input. Default 100.
            expand_dim (bool): if True, return (n, d_y, S). Else (n*S, d_y).
            require_grad (bool): keep gradients. Default False.
            chunk_size (int): max samples per chunk. Default None (= sample_size).

        Returns:
            torch.Tensor.
        """
        data_size = x.size(0)
        if chunk_size is None:
            chunk_size = sample_size

        all_samples = []
        remaining = sample_size
        while remaining > 0:
            cs = min(remaining, chunk_size)
            x_rep = x.repeat(cs, 1)
            if not require_grad:
                with torch.no_grad():
                    samples = self.sampling_func(x_rep).detach()
            else:
                samples = self.sampling_func(x_rep)
            all_samples.append(samples)
            remaining -= cs

        samples = torch.cat(all_samples, dim=0)

        if not expand_dim:
            return samples

        # Zero-copy reshape: (n*S, d) -> (S, n, d) -> (n, d, S)
        samples = samples.view(sample_size, data_size, -1).permute(1, 2, 0)
        return samples

    def sample_batch(self, x, sample_size=100, expand_dim=True,
                     batch_size=None, chunk_size=None):
        """Sample with mini-batches over data points (for large n)."""
        if batch_size is not None and batch_size < x.shape[0]:
            test_loader = make_dataloader(x, batch_size=batch_size, shuffle=False)
            samples = []
            for (x_batch,) in test_loader:
                samples.append(self.sample_onebatch(x_batch, sample_size, expand_dim,
                                                    chunk_size=chunk_size))
            samples = torch.cat(samples, dim=0)
        else:
            samples = self.sample_onebatch(x, sample_size, expand_dim, chunk_size=chunk_size)
        return samples

    def sample(self, x, sample_size=100, expand_dim=True, verbose=True, chunk_size=50):
        """Sample with adaptive batch size on OOM.

        Args:
            x (torch.Tensor): input (n, d_x).
            sample_size (int): samples per input. Default 100.
            expand_dim (bool): return (n, d_y, S) if True. Default True.
            verbose (bool): print OOM warnings. Default True.
            chunk_size (int): max samples per forward pass chunk. Default 50.

        Returns:
            torch.Tensor.
        """
        batch_size = x.shape[0]
        while True:
            try:
                samples = self.sample_batch(x, sample_size, expand_dim, batch_size,
                                            chunk_size=chunk_size)
                break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size = batch_size // 2
                    if batch_size == 0:
                        raise
                    if verbose:
                        print(f"OOM; reducing batch size to {batch_size}")
                    torch.cuda.empty_cache()
                else:
                    raise
        return samples


class StoNet(StoNetBase):
    """Stochastic neural network for distributional regression.

    Maps input x to a distribution over y by injecting noise at each layer.
    Two forward passes with the same x produce different outputs, enabling
    distributional learning via energy score loss.

    Args:
        in_dim (int): input dimension.
        out_dim (int): output dimension.
        num_layer (int): number of layers. Default 2.
        hidden_dim (int): neurons per hidden layer. Default 100.
        noise_dim (int): noise dimension per layer. Default 100.
        add_bn (bool): batch normalization. Default False.
        out_act (str): output activation. Default None.
        resblock (bool): use residual blocks. Default False.
        noise_all_layer (bool): inject noise at all layers. Default True.
        compile_model (bool): apply torch.compile. Default False.
    """

    def __init__(self, in_dim, out_dim, num_layer=2, hidden_dim=100,
                 noise_dim=100, add_bn=False, out_act=None, resblock=False,
                 noise_all_layer=True, out_bias=True, verbose=True,
                 forward_sampling=True, compile_model=False):
        super().__init__(forward_sampling=forward_sampling)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.add_bn = add_bn
        self.noise_all_layer = noise_all_layer
        self.out_bias = out_bias
        if out_act == "softmax" and out_dim == 1:
            out_act = "sigmoid"
        self.out_act = get_act_func(out_act)

        self.num_blocks = None
        if resblock:
            if num_layer % 2 != 0:
                num_layer += 1
                if verbose:
                    print(f"Residual blocks require even layers. Changed to {num_layer}.")
            num_blocks = num_layer // 2
            self.num_blocks = num_blocks
        self.resblock = resblock
        self.num_layer = num_layer

        if self.resblock:
            if self.num_blocks == 1:
                self.net = StoResBlock(dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                                      noise_dim=noise_dim, add_bn=add_bn, out_act=out_act)
            else:
                self.input_layer = StoResBlock(
                    dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
                    noise_dim=noise_dim, add_bn=add_bn, out_act="relu")
                nd = 0 if not noise_all_layer else noise_dim
                self.inter_layer = nn.Sequential(
                    *[StoResBlock(dim=hidden_dim, noise_dim=nd, add_bn=add_bn,
                                 out_act="relu")] * (self.num_blocks - 2))
                self.out_layer = StoResBlock(
                    dim=hidden_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                    noise_dim=nd, add_bn=add_bn, out_act=out_act)
        else:
            self.input_layer = StoLayer(
                in_dim=in_dim, out_dim=hidden_dim, noise_dim=noise_dim,
                add_bn=add_bn, out_act="relu", verbose=verbose)
            nd = 0 if not noise_all_layer else noise_dim
            self.inter_layer = nn.Sequential(
                *[StoLayer(in_dim=hidden_dim, out_dim=hidden_dim, noise_dim=nd,
                           add_bn=add_bn, out_act="relu")] * (num_layer - 2))
            self.out_layer = nn.Linear(hidden_dim, out_dim, bias=out_bias)
            if self.out_act is not None:
                self.out_layer = nn.Sequential(self.out_layer, self.out_act)

        # Optional torch.compile (opt-in)
        if compile_model and hasattr(torch, "compile"):
            try:
                self.forward = torch.compile(
                    self.forward, mode="reduce-overhead", dynamic=True)
            except Exception:
                warnings.warn("torch.compile failed, falling back to eager mode.")

    def forward(self, x):
        if self.num_blocks == 1:
            return self.net(x)
        return self.out_layer(self.inter_layer(self.input_layer(x)))
