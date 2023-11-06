def is_gpu_available() -> bool:
    """
    Check if a GPU is available.

    Returns:
        bool: True if a GPU is available, False otherwise.
    """
    import torch

    # TODO: find the way to check if GPU is available without importing torch
    return torch.cuda.is_available()
