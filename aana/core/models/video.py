import hashlib  # noqa: I001
from dataclasses import dataclass
from pathlib import Path
import torch, decord  # noqa: F401  # See https://github.com/dmlc/decord/issues/263
from decord import DECORDError
from typing import Annotated
from aana.configs.settings import settings
from aana.exceptions.io import VideoReadingException
from aana.core.models.media import Media
import uuid

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

from aana.core.models.base import BaseListModel
from aana.core.models.media import MediaId


@dataclass
class Video(Media):
    """A class representing a video.

    At least one of 'path', 'url', or 'content' must be provided.
    If 'save_on_disk' is True, the video will be saved on disk automatically.

    Attributes:
        path (Path): the path to the video file
        url (str): the URL of the video
        content (bytes): the content of the video in bytes
        media_id (MediaId): the ID of the video. If not provided, it will be generated automatically.
        title (str): the title of the video
        description (str): the description of the video
        media_dir (Path): the directory to save the video in
    """

    title: str = ""
    description: str = ""
    media_dir: Path | None = settings.video_dir

    def validate(self):
        """Validate the video.

        Raises:
            ValueError: if none of 'path', 'url', or 'content' is provided
            VideoReadingException: if the video is not valid
        """
        # validate the parent class
        super().validate()

        # check that at least one of 'path', 'url' or 'content' is provided
        if not any(
            [
                self.path is not None,
                self.url is not None,
                self.content is not None,
            ]
        ):
            raise ValueError(  # noqa: TRY003
                "At least one of 'path', 'url' or 'content' must be provided."
            )

        # check that the video is valid
        if self.path and not self.is_video():
            raise VideoReadingException(video=self)

    def is_video(self) -> bool:
        """Checks if it's a valid video."""
        if not self.path:
            return False

        try:
            decord.VideoReader(str(self.path))
        except DECORDError:
            try:
                decord.AudioReader(str(self.path))
            except DECORDError:
                return False
        return True

    def save_from_url(self, file_path):
        """Save the media from the URL.

        Args:
            file_path (Path): the path to save the media to

        Raises:
            DownloadError: if the media can't be downloaded
            VideoReadingException: if the media is not a valid video
        """
        super().save_from_url(file_path)
        # check that the file is a video
        if not self.is_video():
            raise VideoReadingException(video=self)

    def __repr__(self) -> str:
        """Get the representation of the video.

        Use md5 hash for the content of the video if it is available.

        Returns:
            str: the representation of the video
        """
        content_hash = (
            hashlib.md5(self.content, usedforsecurity=False).hexdigest()
            if self.content
            else None
        )
        return (
            f"Video(path={self.path}, "
            f"url={self.url}, "
            f"content={content_hash}, "
            f"media_id={self.media_id}, "
            f"title={self.title}, "
            f"description={self.description})"
        )


class VideoMetadata(BaseModel):
    """Metadata of a video.

    Attributes:
        title (str): the title of the video
        description (str): the description of the video
    """

    title: str = Field(None, description="The title of the video.")
    description: str = Field(None, description="The description of the video.")
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Metadata of a video.",
        }
    )

class VideoStatus(BaseModel):
    """Metadata of a video.

    Attributes:
        status (str): the title of the video
        description (str): the description of the video
    """

    status: str = Field(None, description="Current processing status of video.")
    model_config = ConfigDict(
        json_schema_extra={
            "status": "Current processing status of video.",
        }
    )

class VideoParams(BaseModel):
    """A pydantic model for video parameters.

    Attributes:
        extract_fps (float): the number of frames to extract per second
        fast_mode_enabled (bool): whether to use fast mode (keyframes only)
    """

    extract_fps: float = Field(
        default=3.0,
        gt=0.0,
        description=(
            "The number of frames to extract per second. "
            "Can be smaller than 1. For example, 0.5 means 1 frame every 2 seconds."
        ),
    )
    fast_mode_enabled: bool = Field(
        default=True,
        description=(
            "Whether to use fast mode (keyframes only). "
            "extract_fps will be ignored if this is set to True."
        ),
    )
    model_config = ConfigDict(
        json_schema_extra={"description": "Video parameters."}, validate_assignment=True
    )


class VideoInput(BaseModel):
    """A video input.

    Exactly one of 'path', 'url', or 'content' must be provided.

    If 'content' is set to 'file',
    the video will be loaded from the files uploaded to the endpoint.

    Attributes:
        media_id (MediaId): the ID of the video. If not provided, it will be generated automatically.
        path (str): the file path of the video
        url (AnyUrl): the URL of the video (supports YouTube videos)
        content (bytes): the content of the video in bytes
    """

    path: str | None = Field(None, description="The file path of the video.")
    url: Annotated[
        AnyUrl | None,
        AfterValidator(lambda x: str(x) if x else None),
        Field(None, description="The URL of the video (supports YouTube videos)."),
    ]
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
