import base64
import hashlib
import io
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import PIL.Image
from pydantic import (
    AfterValidator,
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from typing_extensions import Self

from aana.configs.settings import settings
from aana.core.libraries.image import AbstractImageLibrary
from aana.core.models.base import BaseListModel
from aana.core.models.media import Media, MediaId
from aana.exceptions.io import ImageReadingException
from aana.exceptions.runtime import UploadedFileNotFound
from aana.integrations.external.opencv import OpenCVWrapper
from aana.utils.download import download_file


@dataclass
class Image(Media):
    """A class representing an image.

    At least one of `path`, `url`, `content` or `numpy` must be provided.
    If `save_on_disk` is True, the image will be saved on disk automatically.

    Attributes:
        path (Path): The file path of the image.
        url (str): The URL of the image.
        content (bytes): The content of the image in bytes (image file as bytes).
        numpy (np.ndarray): The image as a numpy array.
        media_id (MediaId): The ID of the image, generated automatically if not provided.
        format (str): The format of the image to save from numpy input (e.g., 'jpeg', 'png', 'bmp'). Default is 'bmp'.
            Ignored if the image is provided as a path, url, or content. Only used for numpy input.
    """

    media_dir: Path | None = settings.image_dir
    numpy: np.ndarray | None = None  # The image as a numpy array.
    image_lib: type[AbstractImageLibrary] = (
        OpenCVWrapper  # The image library to use, TODO: add support for PIL and allow to choose the library
    )
    format: str = "bmp"

    def _validate(self):
        """Validate the image."""
        # validate the parent class
        super()._validate()

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
        If the image is represented as a numpy array, convert it to the specified format and save it on disk.

        First check if the image is already available on disk, then content, then url, then numpy
        to avoid unnecessary operations (e.g. downloading the image or converting it to BMP).

        Raises:
            ValueError: If none of 'path', 'url', 'content' or 'numpy' is provided.
        """
        if self.path:
            return

        image_dir = settings.image_dir
        image_dir.mkdir(parents=True, exist_ok=True)
        file_path = image_dir / (self.media_id + f".{self.format.lower()}")

        if self.content:
            self._save_from_content(file_path)
        elif self.numpy is not None:
            self._save_from_numpy(file_path)
        elif self.url:
            self._save_from_url(file_path)
        else:
            raise ValueError(  # noqa: TRY003
                "At least one of 'path', 'url', 'content' or 'numpy' must be provided."
            )
        self.path = file_path
        self.is_saved = True

    def _save_from_numpy(self, file_path: Path):
        """Save the image from numpy on disk.

        Args:
            file_path (Path): The path of the file to write.
        """
        assert self.numpy is not None  # noqa: S101 TODO
        self.image_lib.write_file(file_path, self.numpy, self.format)

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
            self._load_numpy_from_path()
        elif self.url:
            self._load_numpy_from_url()
        elif self.content:
            self._load_numpy_from_content()
        else:
            raise ValueError(  # noqa: TRY003
                "At least one of 'path', 'url', 'content' or 'numpy' must be provided."
            )
        assert self.numpy is not None  # noqa: S101
        return self.numpy

    def get_pil_image(self) -> PIL.Image:
        """Get the image as a PIL image.

        Returns:
            PIL.Image: The image as a PIL image.
        """
        return PIL.Image.fromarray(self.get_numpy())

    def get_base64(self, format: str | None = None) -> str:  # noqa: A002
        """Get the image as a base64-encoded string.

        Args:
            format (str | None): The format to encode the image in (e.g., 'jpeg', 'png', 'bmp').
                                 If None, uses self.format.

        Returns:
            str: The base64-encoded image.
        """
        if format is None or (format == self.format and self.is_saved):
            image_data = self.get_content()
            base64_encoded_image = base64.b64encode(image_data)
            return base64_encoded_image.decode("utf-8")
        else:
            buf = self.image_lib.write_to_bytes(self.get_numpy(), format)
            base64_encoded_image = base64.b64encode(buf)
            return base64_encoded_image.decode("utf-8")

    def get_base64_url(self, format: str | None = None) -> str:  # noqa: A002
        """Get the image as a base64-encoded string with a data URL.

        Args:
            format (str | None): The format to encode the image in (e.g., 'jpeg', 'png', 'bmp').
                                 If None, uses self.format.

        Returns:
            str: The base64-encoded image with a data URL.
        """
        if format is None:
            format = self.format  # noqa: A001
        base64_image = self.get_base64(format)
        return f"data:image/{format};base64,{base64_image}"

    def _load_numpy_from_path(self):
        """Load the image as a numpy array from a path.

        Raises:
            ImageReadingException: If there is an error reading the image.
        """
        assert self.path is not None  # noqa: S101
        try:
            self.numpy = self.image_lib.read_file(self.path)
        except Exception as e:
            raise ImageReadingException(self) from e

    def _load_numpy_from_image_bytes(self, img_bytes: bytes):
        """Load the image as a numpy array from image bytes (downloaded from URL or read from file).

        Raises:
            ImageReadingException: If there is an error reading the image.
        """
        try:
            self.numpy = self.image_lib.read_from_bytes(img_bytes)
        except Exception as e:
            raise ImageReadingException(self) from e

    def _load_numpy_from_url(self):
        """Load the image as a numpy array from a URL.

        Raises:
            ImageReadingException: If there is an error reading the image.
        """
        assert self.url is not None  # noqa: S101
        content: bytes = download_file(self.url)
        self._load_numpy_from_image_bytes(content)

    def _load_numpy_from_content(self):
        """Load the image as a numpy array from content.

        Raises:
            ImageReadingException: If there is an error reading the image.
        """
        assert self.content is not None  # noqa: S101
        self._load_numpy_from_image_bytes(self.content)

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
            self._load_content_from_path()
        elif self.url:
            self._load_content_from_url()
        elif self.numpy is not None:
            self._load_content_from_numpy()
        else:
            raise ValueError(  # noqa: TRY003
                "At least one of 'path', 'url', 'content' or 'numpy' must be provided."
            )
        assert self.content is not None  # noqa: S101
        return self.content

    def _load_content_from_numpy(self):
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


class ImageInput(BaseModel):
    """An image input.

    Exactly one of 'path', 'url', or 'content' must be provided.

    If 'content' or 'numpy' is set to 'file',
    the image will be loaded from the files uploaded to the endpoint.


    Attributes:
        path (str): the file path of the image
        url (AnyUrl): the URL of the image
        content (bytes): the content of the image in bytes
        numpy (bytes): the image as a numpy array
    """

    path: str | None = Field(None, description="The file path of the image.")
    url: Annotated[
        AnyUrl | None,
        AfterValidator(lambda x: str(x) if x else None),
        Field(None, description="The URL of the image."),
    ]
    content: str | None = Field(
        None,
        description=(
            "The name of the file uploaded to the endpoint. The image will be loaded from the file automatically."
        ),
    )
    numpy: str | None = Field(
        None,
        description="The name of the file uploaded to the endpoint. The image will be loaded from the file automatically.",
    )
    media_id: MediaId = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The ID of the image. If not provided, it will be generated automatically.",
    )
    _file: bytes | None = None

    def set_files(self, files: dict[str, bytes]):
        """Set the files for the image.

        Args:
            files (dict[str, bytes]): the files uploaded to the endpoint

        Raises:
            UploadedFileNotFound: if the file isn't found
        """
        file_name = self.content or self.numpy
        if file_name:
            if file_name not in files:
                raise UploadedFileNotFound(filename=file_name)
            self._file = files[file_name]

    @model_validator(mode="after")
    def check_only_one_field(self) -> Self:
        """Check that exactly one of 'path', 'url', 'content' or 'numpy' is provided.

        Raises:
            ValueError: if not exactly one of 'path', 'url', 'content' or 'numpy' is provided

        Returns:
            Self: the instance
        """
        count = sum(
            value is not None
            for value in [self.path, self.url, self.content, self.numpy]
        )
        if count != 1:
            raise ValueError(  # noqa: TRY003
                "Exactly one of 'path', 'url', 'content' or 'numpy' must be provided."
            )
        return self

    def convert_input_to_object(self) -> Image:
        """Convert the image input to an image object.

        Returns:
            Image: the image object corresponding to the image input

        Raises:
            UploadedFileNotFound: if the file isn't found
            ValueError: if the file isn't found
        """
        file_name = self.content or self.numpy
        if file_name and not self._file:
            raise UploadedFileNotFound(filename=file_name)

        content = None
        numpy = None
        if self._file and self.content:
            content = self._file
        elif self._file and self.numpy:
            file_bytes = self._file
            try:
                numpy = np.load(io.BytesIO(file_bytes), allow_pickle=False)
            except ValueError as e:
                raise ValueError("The numpy file isn't valid.") from e  # noqa: TRY003

        return Image(
            path=Path(self.path) if self.path else None,
            url=self.url,
            content=content,
            numpy=numpy,
            media_id=self.media_id,
        )

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "An image. \n"
                "Exactly one of 'path', 'url', 'content' or 'numpy' must be provided. \n"
                "If 'path' is provided, the image will be loaded from the path. \n"
                "If 'url' is provided, the image will be downloaded from the url. \n"
                "The 'content' and 'numpy' will be loaded automatically "
                "if files are uploaded to the endpoint and the corresponding field is set to the file name."
            ),
            "examples": [
                {"url": "https://example.com/image_12345.jpg", "media_id": "12345"},
                {"path": "/path/to/image_12345.jpg", "media_id": "12345"},
            ],
        },
        validate_assignment=True,
    )


class ImageInputList(BaseListModel):
    """A pydantic model for a list of images to be used as input.

    Only used for the requests, DO NOT use it for anything else.
    Convert it to a list of image objects with convert_input_to_object().
    """

    root: list[ImageInput]

    @model_validator(mode="after")
    def check_non_empty(self) -> Self:
        """Check that the list of images isn't empty.

        Raises:
            ValueError: if the list of images is empty

        Returns:
            Self: the instance
        """
        if len(self.root) == 0:
            raise ValueError("The list of images must not be empty.")  # noqa: TRY003
        return self

    def set_files(self, files: dict[str, bytes]):
        """Set the files for the images.

        Args:
            files (dict[str, bytes]): the files uploaded to the endpoint

        Raises:
            UploadedFileNotFound: if the file isn't found
        """
        for image in self.root:
            image.set_files(files)

    def convert_input_to_object(self) -> list[Image]:
        """Convert the list of image inputs to a list of image objects.

        Returns:
            List[Image]: the list of image objects corresponding to the image inputs
        """
        return [image.convert_input_to_object() for image in self.root]

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A list of images. \n"
                "Exactly one of 'path', 'url', 'content' or 'numpy' must be provided for each image. \n"
                "If 'path' is provided, the image will be loaded from the path. \n"
                "If 'url' is provided, the image will be downloaded from the url. \n"
                "The 'content' and 'numpy' will be loaded automatically "
                "if files are uploaded to the endpoint and the corresponding field is set to the file name."
            )
        },
    )
