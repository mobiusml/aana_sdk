import uuid
from pathlib import Path
from typing import Annotated, TypedDict

from haystack.dataclasses import ByteStream
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

from aana.core.models.media import MediaId
from aana.utils.download import download_file


class PathResult(TypedDict):
    """Represents a path result describing a file on disk."""

    path: Path


class FileInput(BaseModel):
    """A file input.

    Exactly one of 'path', 'url', or 'content' must be provided.

    If 'content' is set to 'file',
    the image will be loaded from the files uploaded to the endpoint.


    Attributes:
        path (str): the file path
        url (AnyUrl): the URL
        content (bytes): the content as bytes
    """

    path: str | None = Field(None, description="The file path.")
    url: Annotated[
        AnyUrl | None,
        AfterValidator(lambda x: str(x) if x else None),
        Field(None, description="The URL."),
    ]
    content: bytes | None = Field(
        None,
        description=(
            "The content as bytes. "
            "Set this field to 'file' to upload files to the endpoint."
        ),
    )
    media_id: MediaId = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The ID of the file. If not provided, it will be generated automatically.",
    )

    def set_file(self, file: bytes):
        """Sets the instance internal file data.

        If 'content' is set to 'file',
        the image will be loaded from the file uploaded to the endpoint.

        set_file() should be called after the files are uploaded to the endpoint.

        Args:
            file (bytes): the file uploaded to the endpoint

        Raises:
            ValueError: if the content or numpy isn't set to 'file'
        """
        if self.content == b"file":
            self.content = file
        else:
            raise ValueError(  # noqa: TRY003
                "The content or numpy of the image must be 'file' to set files."
            )

    def set_files(self, files: list[bytes]):
        """Set the files as bytes content.

        Args:
            files (List[bytes]): the files uploaded to the endpoint

        Raises:
            ValidationError: if the number file inputs and files aren't the same
        """
        if len(files) != 1:
            raise ValueError(  # noqa: TRY003
                "The number of file inputs and files must be the same."
            )
        self.set_file(files[0])

    @model_validator(mode="after")
    def check_only_one_field(self) -> Self:
        """Check that exactly one of 'path', 'url', or 'content' is provided.

        Raises:
            ValueError: if not exactly one of 'path', 'url', or 'content' is provided

        Returns:
            Self: the instance
        """
        count = sum(value is not None for value in [self.path, self.url, self.content])
        if count != 1:
            raise ValueError(  # noqa: TRY003
                "Exactly one of 'path', 'url', 'content' or 'numpy' must be provided."
            )
        return self

    def to_haystack_byte_stream(self) -> ByteStream:
        """Convert the instance to a ByteStream from Haystack.

        Returns:
            ByteStream: the ByteStream
        """
        if self.path:
            return ByteStream.from_file_path(Path(self.path))
        elif self.content:
            return ByteStream(data=self.content)
        elif self.url:
            content = download_file(self.url)
            return ByteStream(data=content)

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A file input. \n"
                "Exactly one of 'path', 'url', or 'content' must be provided. \n"
                "If 'path' is provided, the image will be loaded from the path. \n"
                "If 'url' is provided, the image will be downloaded from the url. \n"
                "The 'content' will be loaded automatically "
                "if files are uploaded to the endpoint (should be set to 'file' for that)."
            )
        },
        validate_assignment=True,
        file_upload=True,
        file_upload_description="Upload file.",
    )
