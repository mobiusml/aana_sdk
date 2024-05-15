import uuid
from pathlib import Path

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

from aana.api.models.base import BaseListModel
from aana.api.models.media_id import MediaId
from aana.models.core.video import Video


class VideoInput(BaseModel):
    """A video input.

    Exactly one of 'path', 'url', or 'content' must be provided.

    If 'content' is set to 'file',
    the video will be loaded from the files uploaded to the endpoint.

    Attributes:
        media_id (MediaId): the ID of the video. If not provided, it will be generated automatically.
        path (str): the file path of the video
        url (str): the URL of the video (supports YouTube videos)
        content (bytes): the content of the video in bytes
    """

    path: str | None = Field(None, description="The file path of the video.")
    url: str | None = Field(
        None, description="The URL of the video (supports YouTube videos)."
    )
    content: bytes | None = Field(
        None,
        description=(
            "The content of the video in bytes. "
            "Set this field to 'file' to upload files to the endpoint."
        ),
    )
    media_id: MediaId = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The ID of the video. If not provided, it will be generated automatically.",
    )

    @field_validator("url")
    @classmethod
    def check_url(cls, url: str) -> str:
        """Check that the URL is valid and supported.

        Right now, we support normal URLs and youtube URLs.

        Args:
            url (str): the URL

        Returns:
            str: the valid URL

        Raises:
            ValueError: if the URL is invalid or unsupported
        """
        # TODO: implement the youtube URL validation
        return url

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
                "Exactly one of 'path', 'url', or 'content' must be provided."
            )
        return self

    def set_file(self, file: bytes):
        """Sets the file.

        If 'content' is set to 'file',
        the video will be loaded from the file uploaded to the endpoint.

        set_file() should be called after the files are uploaded to the endpoint.

        Args:
            file (bytes): the file uploaded to the endpoint

        Raises:
            ValueError: if the content isn't set to 'file'
        """
        if self.content == b"file":
            self.content = file
        else:
            raise ValueError("The content of the video must be 'file' to set files.")  # noqa: TRY003

    def set_files(self, files: list[bytes]):
        """Set the files for the video.

        Args:
            files (List[bytes]): the files uploaded to the endpoint

        Raises:
            ValidationError: if the number of files isn't 1
        """
        if len(files) != 1:
            raise ValidationError.from_exception_data(
                title=self.__class__.__name__,
                line_errors=[
                    InitErrorDetails(
                        loc=("video",),
                        type="value_error",
                        ctx={
                            "error": ValueError(
                                "The number of videos and files must be the same."
                            )
                        },
                        input=None,
                    )
                ],
            )
        self.set_file(files[0])

    def convert_input_to_object(self) -> Video:
        """Convert the video input to a video object.

        Returns:
            Video: the video object corresponding to the video input
        """
        if self.content == b"file":
            raise ValueError(  # noqa: TRY003
                "The content of the video isn't set. Please upload files and call set_files()."
            )
        return Video(
            path=Path(self.path) if self.path is not None else None,
            url=self.url,
            content=self.content,
            media_id=self.media_id,
        )

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A video. \n"
                "Exactly one of 'path', 'url', or 'content' must be provided. \n"
                "If 'path' is provided, the video will be loaded from the path. \n"
                "If 'url' is provided, the video will be downloaded from the url. \n"
                "The 'content' will be loaded automatically "
                "if files are uploaded to the endpoint (should be set to 'file' for that)."
            )
        },
        validate_assignment=True,
        file_upload=True,
        file_upload_description="Upload video file.",
    )


class VideoInputList(BaseListModel):
    """A pydantic model for a list of video inputs.

    Only used for the requests, DO NOT use it for anything else.

    Convert it to a list of video objects with convert_input_to_object().
    """

    root: list[VideoInput]

    @model_validator(mode="after")
    def check_non_empty(self) -> Self:
        """Check that the list of videos isn't empty.

        Raises:
            ValueError: if the list of videos is empty

        Returns:
            Self: the instance
        """
        if len(self.root) == 0:
            raise ValueError("The list of videos must not be empty.")  # noqa: TRY003
        return self

    def set_files(self, files: list[bytes]):
        """Set the files for the videos.

        Args:
            files (List[bytes]): the files uploaded to the endpoint

        Raises:
            ValidationError: if the number of videos and files aren't the same
        """
        if len(self.root) != len(files):
            raise ValidationError.from_exception_data(
                title=self.__class__.__name__,
                line_errors=[
                    InitErrorDetails(
                        loc=("videos",),
                        type="value_error",
                        ctx={
                            "error": ValueError(
                                "The number of videos and files must be the same."
                            )
                        },
                        input=None,
                    )
                ],
            )
        for video, file in zip(self.root, files, strict=False):
            video.set_file(file)

    def convert_input_to_object(self) -> list[VideoInput]:
        """Convert the VideoInputList to a list of video inputs.

        Returns:
            List[VideoInput]: the list of video inputs
        """
        return self.root

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A list of videos. \n"
                "Exactly one of 'path', 'url', or 'content' must be provided for each video. \n"
                "If 'path' is provided, the video will be loaded from the path. \n"
                "If 'url' is provided, the video will be downloaded from the url. \n"
                "The 'content' will be loaded automatically "
                "if files are uploaded to the endpoint (should be set to 'file' for that)."
            )
        },
        file_upload=True,
        file_upload_description="Upload video files.",
    )
