import torch


def is_gpu_available() -> bool:
    """
    Check if a GPU is available.

    Returns:
        bool: True if a GPU is available, False otherwise.
    """

    return torch.cuda.is_available()
