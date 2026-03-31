# torch-engression

GPU-accelerated distributional regression via energy scores.

## Project Structure

```
torch-engression/
├── src/torch_engression/
│   ├── __init__.py          # Package exports: engression(), Engressor
│   ├── engression.py        # Engressor class + engression() convenience function
│   ├── models.py            # StoNet, StoLayer, StoResBlock, StoNetBase
│   ├── loss.py              # energy_loss, energy_loss_two_sample
│   ├── utils.py             # auto_device, vectorize, cor, set_seed
│   └── data/
│       ├── loader.py        # make_dataloader
│       └── simulator.py     # preanm_simulator (test data generation)
├── tests/                   # pytest suite (45 tests)
├── nb/benchmark.ipynb       # CPU vs GPU benchmark
└── pyproject.toml           # hatchling build
```

## Key Design Decisions

- **Auto device detection**: `device=None` → CUDA > MPS > CPU (from torchonometrics)
- **AMP scope**: Forward passes only in FP16. Energy loss in FP32 (catastrophic cancellation risk in s1 - s2/2)
- **torch.compile**: Opt-in (`compile_model=False` default). Dynamic noise injection works with dynamo but adds warmup cost
- **Parameter names**: Match original engression exactly (e.g., `num_layer` not `num_layers`)
- **Chunked sampling**: Default chunk_size=50 to avoid OOM on large sample_size
- **MPS**: Auto-detected with warning about torch.cdist issues

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

## Development

```bash
pip install -e ".[dev]"
```

## Origins

- Core algorithms copied from: `/home/xao/projects/claude/frengression/engression/engression-python/engression/`
- GPU patterns from: `/home/xao/projects/claude/torchonometrics/`
- This is a standalone package (not modifying the originals)
