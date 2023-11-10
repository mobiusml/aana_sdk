import rapidfuzz

from aana.tests.const import ALLOWED_LEVENSTEIN_ERROR_RATE


def is_gpu_available() -> bool:
    """
    Check if a GPU is available.

    Returns:
        bool: True if a GPU is available, False otherwise.
    """
    import torch

    # TODO: find the way to check if GPU is available without importing torch
    return torch.cuda.is_available()


def compare_texts(expected_text: str, text: str):
    """
    Compare two texts using Levenshtein distance.
    The error rate is allowed to be less than ALLOWED_LEVENSTEIN_ERROR_RATE.

    Args:
        expected_text (str): the expected text
        text (str): the actual text

    Raises:
        AssertionError: if the error rate is too high
    """
    dist = rapidfuzz.distance.Levenshtein.distance(text, expected_text)
    assert dist < len(expected_text) * ALLOWED_LEVENSTEIN_ERROR_RATE, (
        expected_text,
        text,
    )
