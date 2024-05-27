from pathlib import Path

import cv2
import numpy as np

from aana.core.libraries.image import AbstractImageLibrary


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
    def write_file(cls, path: Path, img: np.ndarray):
        """Write a file using OpenCV.

        Args:
            path (Path): The path of the file to write.
            img (np.ndarray): The image to write.
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), img)

    @classmethod
    def write_to_bytes(cls, img: np.ndarray) -> bytes:
        """Write bytes using OpenCV.

        Args:
            img (np.ndarray): The image to write.

        Returns:
            bytes: The image as bytes.
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".bmp", img)
        return buffer.tobytes()
