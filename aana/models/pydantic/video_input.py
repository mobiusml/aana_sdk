from pathlib import Path
from typing import List, Optional
import uuid
from pydantic import BaseModel, Field, ValidationError, root_validator
from pydantic.error_wrappers import ErrorWrapper
from aana.models.core.video import Video

from aana.models.pydantic.base import BaseListModel


class VideoInput(BaseModel):
    """
    A video input.

    Exactly one of 'path', 'url', or 'content' must be provided.

    If 'content' is set to 'file',
    the video will be loaded from the files uploaded to the endpoint.

    Attributes:
        video_id (str): the ID of the video. If not provided, it will be generated automatically.
        path (str): the file path of the video
        url (str): the URL of the video
        content (bytes): the content of the video in bytes
    """

    path: Optional[str] = Field(None, description="The file path of the video.")
    url: Optional[str] = Field(None, description="The URL of the video.")
    content: Optional[bytes] = Field(
        None,
        description=(
            "The content of the video in bytes. "
            "Set this field to 'file' to upload files to the endpoint."
        ),
    )
    video_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The ID of the video. If not provided, it will be generated automatically.",
    )

    @root_validator
    def check_only_one_field(cls, values):
        """
        Check that exactly one of 'path', 'url', or 'content' is provided.

        Args:
            values (Dict): the values of the fields

        Returns:
            Dict: the values of the fields

        Raises:
            ValueError: if not exactly one of 'path', 'url', or 'content' is provided
        """
        count = sum(
            value is not None
            for key, value in values.items()
            if key in ["path", "url", "content"]
        )
        if count != 1:
            raise ValueError(
                "Exactly one of 'path', 'url', or 'content' must be provided."
            )
        return values

    def set_file(self, file: bytes):
        """
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
            raise ValueError("The content of the video must be 'file' to set files.")

    def set_files(self, files: List[bytes]):
        """
        Set the files for the video.

        Args:
            files (List[bytes]): the files uploaded to the endpoint

        Raises:
            ValidationError: if the number of files isn't 1
        """
        if len(files) != 1:
            error = ErrorWrapper(
                ValueError("The number of videos and files must be the same."),
                loc=("video",),
            )
            raise ValidationError([error], self.__class__)
        self.set_file(files[0])

    def convert_input_to_object(self) -> Video:
        """
        Convert the video input to a video object.

        Returns:
            Video: the video object corresponding to the video input
        """
        if self.content == b"file":
            raise ValueError(
                "The content of the video isn't set. Please upload files and call set_files()."
            )
        return Video(
            path=Path(self.path) if self.path is not None else None,
            url=self.url,
            content=self.content,
            video_id=self.video_id,
        )

    class Config:
        schema_extra = {
            "description": (
                "A video. \n"
                "Exactly one of 'path', 'url', or 'content' must be provided. \n"
                "If 'path' is provided, the video will be loaded from the path. \n"
                "If 'url' is provided, the video will be downloaded from the url. \n"
                "The 'content' will be loaded automatically "
                "if files are uploaded to the endpoint (should be set to 'file' for that)."
            )
        }
        validate_assignment = True
        file_upload = True
        file_upload_description = "Upload video file."


class VideoInputList(BaseListModel):
    """
    A pydantic model for a list of video inputs.

    Only used for the requests, DO NOT use it for anything else.

    Convert it to a list of video objects with convert_input_to_object().
    """

    __root__: List[VideoInput]

    def set_files(self, files: List[bytes]):
        """
        Set the files for the videos.

        Args:
            files (List[bytes]): the files uploaded to the endpoint

        Raises:
            ValidationError: if the number of videos and files aren't the same
        """

        if len(self.__root__) != len(files):
            error = ErrorWrapper(
                ValueError("The number of videos and files must be the same."),
                loc=("videos",),
            )
            raise ValidationError([error], self.__class__)
        for video, file in zip(self.__root__, files):
            video.set_file(file)

    def convert_input_to_object(self) -> List[Video]:
        """
        Convert the list of video inputs to a list of video objects.

        Returns:
            List[Video]: the list of video objects corresponding to the video inputs
        """
        return [video.convert_input_to_object() for video in self.__root__]

    class Config:
        schema_extra = {
            "description": (
                "A list of videos. \n"
                "Exactly one of 'path', 'url', or 'content' must be provided for each video. \n"
                "If 'path' is provided, the video will be loaded from the path. \n"
                "If 'url' is provided, the video will be downloaded from the url. \n"
                "The 'content' will be loaded automatically "
                "if files are uploaded to the endpoint (should be set to 'file' for that)."
            )
        }
        file_upload = True
        file_upload_description = "Upload video files."