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
    def write_file(cls, path: Path, img: np.ndarray):
        """Write a file using the image library.

        Args:
            path (Path): The path of the file to write.
            img (np.ndarray): The image to write.
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
