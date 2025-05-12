from pathlib import Path

import cv2
import numpy as np

from aana.core.libraries.image import AbstractImageLibrary

__all__ = ["OpenCVWrapper"]


class OpenCVWrapper(AbstractImageLibrary):
    """Wrapper class for OpenCV functions."""

    @classmethod
    def read_file(cls, path: Path) -> np.ndarray:
        """Read a file using OpenCV.

        Args:
            path (Path): The path of the file to read.

        Returns:
            np.ndarray: The file as a numpy array in RGB format.
        """
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @classmethod
    def read_from_bytes(cls, content: bytes) -> np.ndarray:
        """Read bytes using OpenCV.

        Args:
            content (bytes): The content of the file to read.

        Returns:
            np.ndarray: The file as a numpy array in RGB format.
        """
        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

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
            quality (int): JPEG quality (0–100; higher is better). Only used if format is JPEG.
            compression (int): PNG compression level (0–9; higher is smaller). Only used if format is PNG.
        """
        fmt = format.lower()
        if fmt not in ("jpeg", "jpg", "png", "bmp"):
            raise ValueError(f"Unsupported format '{format}'. Choose bmp, png or jpeg.")  # noqa: TRY003

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        params = []
        if fmt in ("jpeg", "jpg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif fmt == "png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression]

        success = cv2.imwrite(str(path), img_bgr, params)
        if not success:
            raise OSError(f"Failed to write image to '{path}'")  # noqa: TRY003

    @classmethod
    def write_to_bytes(
        cls,
        img: np.ndarray,
        format: str = "jpeg",  # noqa: A002
        quality: int = 95,
        compression: int = 3,
    ) -> bytes:
        """Write bytes using OpenCV.

        Args:
            img (np.ndarray): The image to write.
            format (str): The format to use for encoding. Default is "jpeg".
            quality (int): The quality to use for encoding. Default is 95.
            compression (int): The compression level to use for encoding. Default is 3.

        Returns:
            bytes: The image as bytes.
        """
        fmt = format.lower()
        if fmt not in ("jpeg", "jpg", "png", "bmp"):
            raise ValueError(f"Unsupported format '{format}'. Choose bmp, png or jpeg.")  # noqa: TRY003

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        params = []
        if fmt in ("jpeg", "jpg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif fmt == "png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression]

        success, buffer = cv2.imencode(f".{fmt}", img_bgr, params)
        if not success:
            raise OSError("Failed to write image to bytes")  # noqa: TRY003
        return buffer.tobytes()
