import io
from pathlib import Path

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails
from typing_extensions import Self

from aana.models.core.image import Image
from aana.models.pydantic.base import BaseListModel
from aana.models.pydantic.media_id import MediaId


class ImageInput(BaseModel):
    """An image input.

    Exactly one of 'path', 'url', or 'content' must be provided.

    If 'content' or 'numpy' is set to 'file',
    the image will be loaded from the files uploaded to the endpoint.


    Attributes:
        path (str): the file path of the image
        url (str): the URL of the image
        content (bytes): the content of the image in bytes
        numpy (bytes): the image as a numpy array
    """

    path: str | None = Field(None, description="The file path of the image.")
    url: str | None = Field(
        None, description="The URL of the image."
    )  # TODO: validate url
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
        default_factory=lambda: MediaId.random(),
        description="The ID of the image. If not provided, it will be generated automatically.",
    )

    @field_validator("media_id")
    @classmethod
    def media_id_must_not_be_empty(cls, media_id):
        """Validates that the media_id is not an empty string.

        Args:
            media_id (MediaId): The value of the media_id field.

        Raises:
            ValueError: If the media_id is an empty string.

        Returns:
            str: The non-empty media_id value.
        """
        if media_id == "":
            raise ValueError("media_id cannot be an empty string")  # noqa: TRY003
        return media_id

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
