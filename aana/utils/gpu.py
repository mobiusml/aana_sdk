def get_gpu_memory(gpu: int = 0) -> int:
    """Get the total memory of a GPU in bytes."""
    import torch

    return torch.cuda.get_device_properties(gpu).total_memory
