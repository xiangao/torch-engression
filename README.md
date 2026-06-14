# torch-engression

[![docs](https://img.shields.io/badge/docs-site-blue.svg)](https://xiangao.github.io/torch-engression/)

`torch-engression` is a PyTorch implementation of
[engression](https://github.com/xwshen51/engression) from Shen and Meinshausen
(2024). It fits a stochastic neural network using the energy score, so the
object of interest is the conditional distribution `P(Y | X)`, not only a
conditional mean.

## Installation

```bash
pip install torch-engression
```

Or from source:

```bash
git clone https://github.com/xiangao/torch-engression
cd torch-engression
pip install -e .
```

Requires PyTorch >= 2.0. For GPU support, install PyTorch with CUDA.

## Quick Start

```python
import torch
from torch_engression import engression

# Generate data
x = torch.randn(10000, 5)
y = x[:, 0:1] ** 2 + 0.5 * torch.randn(10000, 1)

# Fit model (auto-detects GPU)
model = engression(x, y, num_epochs=200)

# Point predictions
y_mean = model.predict(x, target="mean")
y_median = model.predict(x, target="median")

# Quantiles
q10, q90 = model.predict(x, target=[0.1, 0.9], sample_size=500)

# Full distributional samples
samples = model.sample(x, sample_size=100)  # (n, d_y, 100)

# Evaluate
l2 = model.eval_loss(x, y, loss_type="l2")

# Save / load
model.save("model.pt")
model = Engressor.load("model.pt")
```

## GPU Acceleration

torch-engression automatically detects and uses the best available device (CUDA > MPS > CPU). No code changes needed:

```python
# Auto-detect (default)
model = engression(x, y)

# Explicit device
model = engression(x, y, device="cuda")
model = engression(x, y, device="cpu")
```

Auto-detection probes the GPU before selecting it. A CUDA device can be *visible*
to PyTorch yet unable to run kernels — for example a card whose compute capability
the installed PyTorch wheel was not built for (`cudaErrorNoKernelImageForDevice`).
In that case `device=None` warns once and falls back to CPU rather than crashing.
Passing `device="cuda"` explicitly still forces the GPU.

### Mixed Precision Training

On CUDA devices, training automatically uses mixed precision (AMP):
- Forward passes run in FP16 for faster matrix multiplications
- Energy loss computed in FP32 to avoid numerical issues
- GradScaler handles gradient scaling

### torch.compile (opt-in)

For long training runs, enable `torch.compile` for additional speedup:

```python
model = engression(x, y, compile_model=True)
```

## Benchmark

Training time on GTX 1080 Ti (100 epochs, 3-layer StoNet, hidden_dim=100):

| n | CPU (s) | GPU (s) | Speedup |
|---|---------|---------|---------|
| 1,000 | 1.4 | 1.2 | 1.2x |
| 5,000 | 2.0 | 1.0 | 2.1x |
| 10,000 | 3.6 | 1.2 | 3.1x |
| 50,000 | 17.6 | 1.3 | 13.4x |
| 100,000 | 38.1 | 2.5 | **15.1x** |

These timings are meant as a scale check, not as a universal benchmark. See
`nb/benchmark.ipynb` for the script.

## Documentation & examples

Full documentation: **<https://xiangao.github.io/torch-engression/>**

| Page | Description |
|------|-------------|
| [Benchmark notebook](https://github.com/xiangao/torch-engression/blob/master/nb/benchmark.ipynb) | End-to-end runtime and distributional-fit benchmark |
| [Examples page](https://xiangao.github.io/torch-engression/examples/) | Notebook and generated benchmark figure links |

## API Reference

### `engression(x, y, **kwargs)`

Convenience function that creates an `Engressor`, trains it, and returns it.

**Key parameters:**
- `num_layer` (int, default 2): number of layers
- `hidden_dim` (int, default 100): neurons per layer
- `noise_dim` (int, default 100): noise dimension per layer
- `lr` (float, default 0.0001): learning rate
- `num_epochs` (int, default 500): training epochs
- `batch_size` (int, default None): mini-batch size (None = full batch)
- `device` (str/None, default None): "cpu", "cuda", "gpu", or None (auto)
- `standardize` (bool, default True): standardize data internally
- `compile_model` (bool, default False): apply torch.compile
- `seed` (int, default None): random seed for reproducibility

### `Engressor`

- `.predict(x, target="mean", sample_size=100)` — point predictions
- `.sample(x, sample_size=100)` — distributional samples
- `.eval_loss(x, y, loss_type="l2")` — evaluate (l2, l1, energy, cor)
- `.plot(x_te, y_te, ...)` — visualization
- `.save(path)` / `Engressor.load(path)` — serialization
- `.summary()` — print model info

## References

- Shen, X. & Meinshausen, N. (2024). Engression: Extrapolation through the Lens of Distributional Regression. *JMLR*.
- Original Python package: [engression](https://github.com/xwshen51/engression)
- GPU patterns from: [torchonometrics](https://github.com/apoorvalal/torchonometrics)
