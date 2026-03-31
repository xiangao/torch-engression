import torch
from torch.utils.data import TensorDataset, DataLoader


def make_dataloader(x, y=None, batch_size=128, shuffle=True, num_workers=0, pin_memory=False):
    """Create a DataLoader from tensors.

    Args:
        x (torch.Tensor): predictor data.
        y (torch.Tensor, optional): response data.
        batch_size (int): batch size. Default 128.
        shuffle (bool): whether to shuffle. Default True.
        num_workers (int): number of worker processes. Default 0.
        pin_memory (bool): pin memory for faster CUDA transfers. Default False.

    Returns:
        DataLoader
    """
    if y is None:
        dataset = TensorDataset(x)
    else:
        dataset = TensorDataset(x, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
