from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Optional
import uuid

from aana.configs.settings import settings
from aana.utils.general import download_file


@dataclass
class Video:
    """
    A class representing a video.

    At least one of 'path', 'url', or 'content' must be provided.
    If 'save_on_disk' is True, the video will be saved on disk automatically.

    Attributes:
        path (Path): the path to the video file
        url (str): the URL of the video
        content (bytes): the content of the video in bytes
        video_id (str): the ID of the video. If not provided, it will be generated automatically.
    """

    path: Optional[Path] = None
    url: Optional[str] = None
    content: Optional[bytes] = None
    video_id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))
    save_on_disk: bool = True  # Whether to save the image on disc or not
    saved: bool = False  # Whether the video is saved on disc by the class or not (used for cleanup)

    def __post_init__(self):
        """
        Post-initialization.

        Perform checks and save the video on disc if needed.
        """
        # check that at least one of 'path', 'url', or 'content' is provided
        if not any(
            [
                self.path is not None,
                self.url is not None,
                self.content is not None,
            ]
        ):
            raise ValueError(
                "At least one of 'path', 'url', or 'content' must be provided."
            )

        # check if path exists if provided
        if self.path and not self.path.exists():
            raise FileNotFoundError(f"Video file '{self.path}' does not exist.")

        if self.save_on_disk:
            self.save()

    def save(self):
        """
        Save the video on disc.
        If the video is already available on disc, do nothing.
        If the video represented as a byte string, save it on disc.
        If the video is represented as a URL, download it and save it on disc.

        Raises:
            ValueError: if at least one of 'path', 'url', or 'content' is not provided
        """
        if self.path:
            return

        file_dir = settings.tmp_data_dir / "videos"
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / (self.video_id + ".mp4")

        if self.content:
            self.save_from_content(file_path)
        elif self.url:
            self.save_from_url(file_path)
        else:
            raise ValueError(
                "At least one of 'path', 'url', or 'content' must be provided."
            )
        self.path = file_path
        self.saved = True

    def save_from_bytes(self, file_path: Path, content: bytes):
        """
        Save the video from bytes.

        Args:
            file_path (Path): the path to save the video to
            content (bytes): the content of the video
        """
        file_path.write_bytes(content)

    def save_from_content(self, file_path: Path):
        """
        Save the video from the content.

        Args:
            file_path (Path): the path to save the video to
        """
        assert self.content is not None
        self.save_from_bytes(file_path, self.content)

    def save_from_url(self, file_path):
        """
        Save the video from the URL.

        Args:
            file_path (Path): the path to save the video to

        Raises:
            DownloadError: if the video can't be downloaded
        """
        content: bytes = download_file(self.url)
        self.save_from_bytes(file_path, content)

    def get_content(self) -> bytes:
        """
        Get the content of the video as bytes.

        Returns:
            bytes: the content of the video

        Raises:
            ValueError: if at least one of 'path', 'url', or 'content' is not provided
        """
        if self.content:
            return self.content
        elif self.path:
            self.content = self.load_content_from_path()
            return self.content
        elif self.url:
            self.content = self.load_content_from_url()
            return self.content
        else:
            raise ValueError(
                "At least one of 'path', 'url', or 'content' must be provided."
            )

    def load_content_from_path(self) -> bytes:
        """
        Load the content of the video from the path.

        Returns:
            bytes: the content of the video
        """
        assert self.path is not None
        content: bytes = self.path.read_bytes()
        return content

    def load_content_from_url(self) -> bytes:
        """
        Load the content of the video from the URL.

        Returns:
            bytes: the content of the video

        Raises:
            DownloadError: if the video can't be downloaded
        """
        assert self.url is not None
        content: bytes = download_file(self.url)
        return content

    def __repr__(self) -> str:
        """
        Get the representation of the video.

        Use md5 hash for the content of the video if it is available.

        Returns:
            str: the representation of the video
        """
        content_hash = hashlib.md5(self.content).hexdigest() if self.content else None
        return (
            f"Video(path={self.path}, "
            f"url={self.url}, "
            f"content={content_hash}, "
            f"video_id={self.video_id})"
        )

    def __str__(self) -> str:
        """
        Get the string representation of the video.

        Returns:
            str: the string representation of the video
        """
        return self.__repr__()

    def cleanup(self):
        """
        Cleanup the video.

        If the video is saved on disc by the class, delete it.
        """
        if self.saved and self.path:
            self.path.unlink(missing_ok=True)
