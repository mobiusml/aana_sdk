import hashlib  # noqa: I001
from dataclasses import dataclass
from pathlib import Path
import torch, decord  # noqa: F401  # See https://github.com/dmlc/decord/issues/263
from decord import DECORDError

from aana.configs.settings import settings
from aana.exceptions.general import VideoReadingException
from aana.models.core.media import Media


@dataclass
class Video(Media):
    """A class representing a video.

    At least one of 'path', 'url', or 'content' must be provided.
    If 'save_on_disk' is True, the video will be saved on disk automatically.

    Attributes:
        path (Path): the path to the video file
        url (str): the URL of the video
        content (bytes): the content of the video in bytes
        media_id (str): the ID of the video. If not provided, it will be generated automatically.
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
