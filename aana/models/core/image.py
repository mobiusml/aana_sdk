from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Optional, Type
import uuid
import cv2

import numpy as np
from aana.configs.settings import settings

from aana.exceptions.general import ImageReadingException
from aana.utils.general import download_file


class AbstractImageLibrary:
    """
    Abstract class for image libraries.
    """

    @classmethod
    def read_file(cls, path: Path) -> np.ndarray:
        """
        Read a file using the image library.

        Args:
            path (Path): The path of the file to read.

        Returns:
            np.ndarray: The file as a numpy array.
        """
        raise NotImplementedError

    @classmethod
    def read_bytes(cls, content: bytes) -> np.ndarray:
        """
        Read bytes using the image library.

        Args:
            content (bytes): The content of the file to read.

        Returns:
            np.ndarray: The file as a numpy array.
        """
        raise NotImplementedError

    @classmethod
    def write_file(cls, path: Path, img: np.ndarray):
        """
        Write a file using the image library.

        Args:
            path (Path): The path of the file to write.
            img (np.ndarray): The image to write.
        """
        raise NotImplementedError

    @classmethod
    def write_bytes(cls, img: np.ndarray) -> bytes:
        """
        Write bytes using the image library.

        Args:
            img (np.ndarray): The image to write.

        Returns:
            bytes: The image as bytes.
        """
        raise NotImplementedError


class OpenCVWrapper(AbstractImageLibrary):
    """
    Wrapper class for OpenCV functions.
    """

    @classmethod
    def read_file(cls, path: Path) -> np.ndarray:
        """
        Read a file using OpenCV.

        Args:
            path (Path): The path of the file to read.

        Returns:
            np.ndarray: The file as a numpy array in RGB format.
        """
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @classmethod
    def read_bytes(cls, content: bytes) -> np.ndarray:
        """
        Read bytes using OpenCV.

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
        """
        Write a file using OpenCV.

        Args:
            path (Path): The path of the file to write.
            img (np.ndarray): The image to write.
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), img)

    @classmethod
    def write_bytes(cls, img: np.ndarray) -> bytes:
        """
        Write bytes using OpenCV.

        Args:
            img (np.ndarray): The image to write.

        Returns:
            bytes: The image as bytes.
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".bmp", img)
        return buffer.tobytes()


@dataclass
class Image:
    path: Optional[Path] = None  # The file path of the image.
    url: Optional[str] = None  # The URL of the image.
    content: Optional[
        bytes
    ] = None  # The content of the image in bytes (image file as bytes).
    numpy: Optional[np.ndarray] = None  # The image as a numpy array.
    image_id: Optional[str] = field(
        default_factory=lambda: str(uuid.uuid4())
    )  # The ID of the image, generated automatically
    save_on_disc: bool = True  # Whether to save the image on disc or not
    image_lib: Type[
        AbstractImageLibrary
    ] = OpenCVWrapper  # The image library to use, TODO: add support for PIL and allow to choose the library
    is_saved: bool = False  # Whether the image is saved on disc by the class or not (used for cleanup)

    def __post_init__(self):
        """
        Post-initialization method.

        Performs checks:
        - Checks that path is a Path object.
        - Checks that at least one of 'path', 'url', 'content' or 'numpy' is provided.
        - Checks if path exists if provided.

        Saves the image on disk if needed.
        """
        # check that path is a Path object
        if self.path:
            assert isinstance(self.path, Path)
        # check that at least one of 'path', 'url', 'content' or 'numpy' is provided
        if not any(
            [
                self.path is not None,
                self.url is not None,
                self.content is not None,
                self.numpy is not None,
            ]
        ):
            raise ValueError(
                "At least one of 'path', 'url', 'content' or 'numpy' must be provided."
            )

        # check if path exists if provided
        if self.path and not self.path.exists():
            raise FileNotFoundError(f"Image file not found: {self.path}")

        if self.save_on_disc:
            self.save_image()

    def save_image(self):
        """
        Save the image on disc.
        If the image is already available on disc, do nothing.
        If the image represented as a byte string, save it on disc.
        If the image is represented as a URL, download it and save it on disc.
        If the image is represented as a numpy array, convert it to BMP and save it on disc.

        First check if the image is already available on disc, then content, then url, then numpy
        to avoid unnecessary operations (e.g. downloading the image or converting it to BMP).

        Raises:
            ValueError: If none of 'path', 'url', 'content' or 'numpy' is provided.
        """
        if self.path:
            return

        file_dir = settings.tmp_data_dir / "images"
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / (self.image_id + ".bmp")

        if self.content:
            self.save_from_content(file_path)
        elif self.numpy is not None:
            self.save_from_numpy(file_path)
        elif self.url:
            self.save_from_url(file_path)
        else:
            raise ValueError(
                "At least one of 'path', 'url', 'content' or 'numpy' must be provided."
            )
        self.path = file_path
        self.is_saved = True

    def save_from_content(self, file_path: Path):
        """
        Save the image from content on disc.

        Args:
            file_path (Path): The path of the file to write.
        """
        assert self.content is not None
        with open(file_path, "wb") as f:
            f.write(self.content)

    def save_from_numpy(self, file_path: Path):
        """
        Save the image from numpy on disc.

        Args:
            file_path (Path): The path of the file to write.
        """
        assert self.numpy is not None
        self.image_lib.write_file(file_path, self.numpy)

    def save_from_url(self, file_path: Path):
        """
        Save the image from URL on disc.

        Args:
            file_path (Path): The path of the file to write.
        """
        assert self.url is not None
        content = download_file(self.url)
        file_path.write_bytes(content)

    def get_numpy(self) -> np.ndarray:
        """
        Load the image as a numpy array.

        Returns:
            np.ndarray: The image as a numpy array.

        Raises:
            ValueError: If none of 'path', 'url', 'content' or 'numpy' is provided.
            ImageReadingException: If there is an error reading the image.
        """
        if self.numpy is not None:
            return self.numpy
        elif self.path:
            self.load_numpy_from_path()
        elif self.url:
            self.load_numpy_from_url()
        elif self.content:
            self.load_numpy_from_content()
        else:
            raise ValueError(
                "At least one of 'path', 'url', 'content' or 'numpy' must be provided."
            )
        assert self.numpy is not None
        return self.numpy

    def load_numpy_from_path(self):
        """
        Load the image as a numpy array from a path.

        Raises:
            ImageReadingException: If there is an error reading the image.
        """
        assert self.path is not None
        try:
            self.numpy = self.image_lib.read_file(self.path)
        except Exception as e:
            raise ImageReadingException(self) from e

    def load_numpy_from_image_bytes(self, img_bytes: bytes):
        """
        Load the image as a numpy array from image bytes (downloaded from URL or read from file).

        Raises:
            ImageReadingException: If there is an error reading the image.
        """
        try:
            self.numpy = self.image_lib.read_bytes(img_bytes)
        except Exception as e:
            raise ImageReadingException(self) from e

    def load_numpy_from_url(self):
        """
        Load the image as a numpy array from a URL.

        Raises:
            ImageReadingException: If there is an error reading the image.
        """
        assert self.url is not None
        content: bytes = download_file(self.url)
        self.load_numpy_from_image_bytes(content)

    def load_numpy_from_content(self):
        """
        Load the image as a numpy array from content.

        Raises:
            ImageReadingException: If there is an error reading the image.
        """
        assert self.content is not None
        self.load_numpy_from_image_bytes(self.content)

    def get_content(self) -> bytes:
        """
        Get the content of the image as bytes.

        Returns:
            bytes: The content of the image as bytes.

        Raises:
            ValueError: If none of 'path', 'url', 'content' or 'numpy' is provided.
        """
        if self.content:
            return self.content
        elif self.path:
            self.load_content_from_path()
        elif self.url:
            self.load_content_from_url()
        elif self.numpy is not None:
            self.load_content_from_numpy()
        else:
            raise ValueError(
                "At least one of 'path', 'url', 'content' or 'numpy' must be provided."
            )
        assert self.content is not None
        return self.content

    def load_content_from_numpy(self):
        """
        Load the content of the image from numpy.
        """
        assert self.numpy is not None
        self.content = self.image_lib.write_bytes(self.numpy)

    def load_content_from_path(self):
        """
        Load the content of the image from the path.

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        assert self.path is not None
        with open(self.path, "rb") as f:
            self.content = f.read()

    def load_content_from_url(self):
        """
        Load the content of the image from the URL using requests.

        Raises:
            DownloadException: If there is an error downloading the image.
        """
        assert self.url is not None
        self.content = download_file(self.url)

    def __repr__(self) -> str:
        """
        Get the representation of the image.

        Use md5 hash for the content of the image if it is available.

        For numpy array, use the shape of the array with the md5 hash of the array if it is available.

        Returns:
            str: The representation of the image.
        """
        content_hash = hashlib.md5(self.content).hexdigest() if self.content else None
        if self.numpy is not None:
            assert self.numpy is not None
            numpy_hash = hashlib.md5(self.numpy.tobytes()).hexdigest()
            numpy_repr = f"ndarray(shape={self.numpy.shape}, dtype={self.numpy.dtype}, md5={numpy_hash})"
        else:
            numpy_repr = None
        return (
            f"Image(path={self.path}, "
            f"url={self.url}, "
            f"content={content_hash}, "
            f"numpy={numpy_repr}, "
            f"image_id={self.image_id})"
        )

    def __str__(self) -> str:
        """
        Get the string representation of the image.

        Returns:
            str: The string representation of the image.
        """
        return self.__repr__()

    def cleanup(self):
        """
        Cleanup the image.

        Remove the image from disc if it was saved by the class.
        """
        # Remove the image from disc if it was saved by the class
        if self.is_saved and self.path:
            self.path.unlink(missing_ok=True)
