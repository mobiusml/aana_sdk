__all__ = ["get_gpu_memory"]


def get_gpu_memory(gpu: int = 0) -> int:
    """Get the total memory of a GPU in bytes.

    Args:
        gpu (int): the GPU index. Defaults to 0.

    Returns:
        int: the total memory of the GPU in bytes
    """
    import torch

    return torch.cuda.get_device_properties(gpu).total_memory
