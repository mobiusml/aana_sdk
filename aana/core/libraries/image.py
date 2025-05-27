from pathlib import Path

import numpy as np


class AbstractImageLibrary:
    """Abstract class for image libraries."""

    @classmethod
    def read_file(cls, path: Path) -> np.ndarray:
        """Read a file using the image library.

        Args:
            path (Path): The path of the file to read.

        Returns:
            np.ndarray: The file as a numpy array.
        """
        raise NotImplementedError

    @classmethod
    def read_from_bytes(cls, content: bytes) -> np.ndarray:
        """Read bytes using the image library.

        Args:
            content (bytes): The content of the file to read.

        Returns:
            np.ndarray: The file as a numpy array.
        """
        raise NotImplementedError

    @classmethod
    def write_file(
        cls,
        path: Path,
        img: np.ndarray,
        format: str = "bmp",  # noqa: A002
        quality: int = 95,
        compression: int = 3,
    ):
        """Write an image to disk in BMP, PNG or JPEG format.

        Args:
            path (Path): Base path (extension will be enforced).
            img (np.ndarray): RGB image array.
            format (str): One of "bmp", "png", "jpeg" (or "jpg").
            quality (int): JPEG quality (0-100; higher is better). Only used if format is JPEG.
            compression (int): PNG compression level (0-9; higher is smaller). Only used if format is PNG.
        """
        raise NotImplementedError

    @classmethod
    def write_to_bytes(cls, img: np.ndarray) -> bytes:
        """Write bytes using the image library.

        Args:
            img (np.ndarray): The image to write.

        Returns:
            bytes: The image as bytes.
        """
        raise NotImplementedError
