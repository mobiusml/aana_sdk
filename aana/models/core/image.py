import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aana.configs.settings import settings
from aana.exceptions.io import ImageReadingException
from aana.extern.opencv import OpenCVWrapper
from aana.models.core.media import Media
from aana.models.libs.image import AbstractImageLibrary
from aana.utils.download import download_file


@dataclass
class Image(Media):
    """A class representing an image.

    At least one of 'path', 'url', 'content' or 'numpy' must be provided.
    If 'save_on_disk' is True, the image will be saved on disk automatically.

    Attributes:
        path (Path): The file path of the image.
        url (str): The URL of the image.
        content (bytes): The content of the image in bytes (image file as bytes).
        numpy (np.ndarray): The image as a numpy array.
        media_id (MediaId): The ID of the image, generated automatically if not provided.
    """

    media_dir: Path | None = settings.image_dir
    numpy: np.ndarray | None = None  # The image as a numpy array.
    image_lib: type[
        AbstractImageLibrary
    ] = OpenCVWrapper  # The image library to use, TODO: add support for PIL and allow to choose the library

    def validate(self):
        """Validate the image."""
        # validate the parent class
        super().validate()

        # check that at least one of 'path', 'url', 'content' or 'numpy' is provided
        if not any(
            [
                self.path is not None,
                self.url is not None,
                self.content is not None,
                self.numpy is not None,
            ]
        ):
            raise ValueError(  # noqa: TRY003
                "At least one of 'path', 'url', 'content' or 'numpy' must be provided."
            )

    def save(self):
        """Save the image on disk.

        If the image is already available on disk, do nothing.
        If the image represented as a byte string, save it on disk.
        If the image is represented as a URL, download it and save it on disk.
        If the image is represented as a numpy array, convert it to BMP and save it on disk.

        First check if the image is already available on disk, then content, then url, then numpy
        to avoid unnecessary operations (e.g. downloading the image or converting it to BMP).

        Raises:
            ValueError: If none of 'path', 'url', 'content' or 'numpy' is provided.
        """
        if self.path:
            return

        image_dir = settings.image_dir
        image_dir.mkdir(parents=True, exist_ok=True)
        file_path = image_dir / (self.media_id + ".bmp")

        if self.content:
            self.save_from_content(file_path)
        elif self.numpy is not None:
            self.save_from_numpy(file_path)
        elif self.url:
            self.save_from_url(file_path)
        else:
            raise ValueError(  # noqa: TRY003
                "At least one of 'path', 'url', 'content' or 'numpy' must be provided."
            )
        self.path = file_path
        self.is_saved = True

    def save_from_numpy(self, file_path: Path):
        """Save the image from numpy on disk.

        Args:
            file_path (Path): The path of the file to write.
        """
        assert self.numpy is not None  # noqa: S101 TODO
        self.image_lib.write_file(file_path, self.numpy)

    def get_numpy(self) -> np.ndarray:
        """Load the image as a numpy array.

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
            raise ValueError(  # noqa: TRY003
                "At least one of 'path', 'url', 'content' or 'numpy' must be provided."
            )
        assert self.numpy is not None  # noqa: S101
        return self.numpy

    def load_numpy_from_path(self):
        """Load the image as a numpy array from a path.

        Raises:
            ImageReadingException: If there is an error reading the image.
        """
        assert self.path is not None  # noqa: S101
        try:
            self.numpy = self.image_lib.read_file(self.path)
        except Exception as e:
            raise ImageReadingException(self) from e

    def load_numpy_from_image_bytes(self, img_bytes: bytes):
        """Load the image as a numpy array from image bytes (downloaded from URL or read from file).

        Raises:
            ImageReadingException: If there is an error reading the image.
        """
        try:
            self.numpy = self.image_lib.read_from_bytes(img_bytes)
        except Exception as e:
            raise ImageReadingException(self) from e

    def load_numpy_from_url(self):
        """Load the image as a numpy array from a URL.

        Raises:
            ImageReadingException: If there is an error reading the image.
        """
        assert self.url is not None  # noqa: S101
        content: bytes = download_file(self.url)
        self.load_numpy_from_image_bytes(content)

    def load_numpy_from_content(self):
        """Load the image as a numpy array from content.

        Raises:
            ImageReadingException: If there is an error reading the image.
        """
        assert self.content is not None  # noqa: S101
        self.load_numpy_from_image_bytes(self.content)

    def get_content(self) -> bytes:
        """Get the content of the image as bytes.

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
            raise ValueError(  # noqa: TRY003
                "At least one of 'path', 'url', 'content' or 'numpy' must be provided."
            )
        assert self.content is not None  # noqa: S101
        return self.content

    def load_content_from_numpy(self):
        """Load the content of the image from numpy."""
        assert self.numpy is not None  # noqa: S101
        self.content = self.image_lib.write_to_bytes(self.numpy)

    def __repr__(self) -> str:
        """Get the representation of the image.

        Use md5 hash for the content of the image if it is available.

        For numpy array, use the shape of the array with the md5 hash of the array if it is available.

        Returns:
            str: The representation of the image.
        """
        content_hash = (
            hashlib.md5(self.content, usedforsecurity=False).hexdigest()
            if self.content
            else None
        )
        if self.numpy is not None:
            assert self.numpy is not None  # noqa: S101
            numpy_hash = hashlib.md5(
                self.numpy.tobytes(), usedforsecurity=False
            ).hexdigest()
            numpy_repr = f"ndarray(shape={self.numpy.shape}, dtype={self.numpy.dtype}, md5={numpy_hash})"
        else:
            numpy_repr = None
        return (
            f"Image(path={self.path}, "
            f"url={self.url}, "
            f"content={content_hash}, "
            f"numpy={numpy_repr}, "
            f"media_id={self.media_id})"
        )
