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
    ValidationError,
    model_validator,
)
from pydantic_core import InitErrorDetails
from typing_extensions import Self

from aana.configs.settings import settings
from aana.core.libraries.image import AbstractImageLibrary
from aana.core.models.base import BaseListModel
from aana.core.models.media import Media, MediaId
from aana.exceptions.io import ImageReadingException
from aana.integrations.external.opencv import OpenCVWrapper
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

    def get_pil_image(self) -> PIL.Image:
        """Get the image as a PIL image.

        Returns:
            PIL.Image: The image as a PIL image.
        """
        return PIL.Image.fromarray(self.get_numpy())

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
    content: bytes | None = Field(
        None,
        description=(
            "The content of the image in bytes. "
            "Set this field to 'file' to upload files to the endpoint."
        ),
    )
    numpy: bytes | None = Field(
        None,
        description=(
            "The image as a numpy array. "
            "Set this field to 'file' to upload files to the endpoint."
        ),
    )
    media_id: MediaId = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The ID of the image. If not provided, it will be generated automatically.",
    )

    def set_file(self, file: bytes):
        """Sets the instance internal file data.

        If 'content' or 'numpy' is set to 'file',
        the image will be loaded from the file uploaded to the endpoint.

        set_file() should be called after the files are uploaded to the endpoint.

        Args:
            file (bytes): the file uploaded to the endpoint

        Raises:
            ValueError: if the content or numpy isn't set to 'file'
        """
        if self.content == b"file":
            self.content = file
        elif self.numpy == b"file":
            self.numpy = file
        else:
            raise ValueError(  # noqa: TRY003
                "The content or numpy of the image must be 'file' to set files."
            )

    def set_files(self, files: list[bytes]):
        """Set the files for the image.

        Args:
            files (List[bytes]): the files uploaded to the endpoint

        Raises:
            ValidationError: if the number of images and files aren't the same
        """
        if len(files) != 1:
            raise ValidationError.from_exception_data(
                title=self.__class__.__name__,
                line_errors=[
                    InitErrorDetails(
                        loc=("images",),
                        type="value_error",
                        ctx={
                            "error": ValueError(
                                "The number of images and files must be the same."
                            )
                        },
                        input=None,
                    )
                ],
            )
        self.set_file(files[0])

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
            ValueError: if the numpy file isn't set
        """
        if self.numpy and self.numpy != b"file":
            try:
                numpy = np.load(io.BytesIO(self.numpy), allow_pickle=False)
            except ValueError:
                raise ValueError("The numpy file isn't valid.")  # noqa: TRY003, TRY200, B904 TODO
        elif self.numpy == b"file":
            raise ValueError("The numpy file isn't set. Call set_files() to set it.")  # noqa: TRY003
        else:
            numpy = None

        return Image(
            path=Path(self.path) if self.path else None,
            url=self.url,
            content=self.content,
            numpy=numpy,
            media_id=self.media_id,
        )

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "An image. \n"
                "Exactly one of 'path', 'url', or 'content' must be provided. \n"
                "If 'path' is provided, the image will be loaded from the path. \n"
                "If 'url' is provided, the image will be downloaded from the url. \n"
                "The 'content' will be loaded automatically "
                "if files are uploaded to the endpoint (should be set to 'file' for that)."
            )
        },
        validate_assignment=True,
        file_upload=True,
        file_upload_description="Upload image file.",
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

    def set_files(self, files: list[bytes]):
        """Set the files for the images.

        Args:
            files (List[bytes]): the files uploaded to the endpoint

        Raises:
            ValidationError: if the number of images and files aren't the same
        """
        if len(self.root) != len(files):
            error = ValueError("The number of images and files must be the same.")
            # raise ValidationError(error,
            raise error
        for image, file in zip(self.root, files, strict=False):
            image.set_file(file)

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
                "Exactly one of 'path', 'url', or 'content' must be provided for each image. \n"
                "If 'path' is provided, the image will be loaded from the path. \n"
                "If 'url' is provided, the image will be downloaded from the url. \n"
                "The 'content' will be loaded automatically "
                "if files are uploaded to the endpoint (should be set to 'file' for that)."
            )
        },
        file_upload=True,
        file_upload_description="Upload image files.",
    )
