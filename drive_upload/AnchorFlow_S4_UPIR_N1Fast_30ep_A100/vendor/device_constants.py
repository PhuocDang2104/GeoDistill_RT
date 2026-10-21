"""Device-only constant creation, including inside CUDA graph capture."""
import torch


def device_constant(reference, value):
    if isinstance(value, (list, tuple)):
        return torch.stack([reference.new_full((), x) for x in value])
    return reference.new_full((), value)
