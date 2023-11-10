from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Optional
import uuid

from aana.configs.settings import settings
from aana.utils.general import download_file


@dataclass
class Media:
    """
    A base class representing a media.

    It is used to represent images, medias, and audio files.

    At least one of 'path', 'url', or 'content' must be provided.
    If 'save_on_disk' is True, the media will be saved on disk automatically.

    Attributes:
        path (Path): the path to the media file
        url (str): the URL of the media
        content (bytes): the content of the media in bytes
        media_id (str): the ID of the media. If not provided, it will be generated automatically.
    """

    path: Optional[Path] = None
    url: Optional[str] = None
    content: Optional[bytes] = None
    media_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    save_on_disk: bool = True
    is_saved: bool = False

    def validate(self):
        """
        Validate the media.
        """
        # check that path is a Path object
        if self.path:
            assert isinstance(self.path, Path)

        # check if path exists if provided
        if self.path and not self.path.exists():
            raise FileNotFoundError(f"File '{self.path}' does not exist.")

    def __post_init__(self):
        """
        Post-initialization.

        Perform checks and save the media on disk if needed.
        """
        self.validate()

        if self.save_on_disk:
            self.save()

    def save(self):
        """
        Save the media on disk.
        If the media is already available on disk, do nothing.
        If the media represented as a byte string, save it on disk
        If the media is represented as a URL, download it and save it on disk.

        Raises:
            ValueError: if at least one of 'path', 'url', or 'content' is not provided
        """
        if self.path:
            return

        file_dir = settings.tmp_data_dir / "medias"
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / (self.media_id + ".mp4")

        if self.content:
            self.save_from_content(file_path)
        elif self.url:
            self.save_from_url(file_path)
        else:
            raise ValueError(
                "At least one of 'path', 'url', or 'content' must be provided."
            )
        self.path = file_path
        self.is_saved = True

    def save_from_bytes(self, file_path: Path, content: bytes):
        """
        Save the media from bytes.

        Args:
            file_path (Path): the path to save the media to
            content (bytes): the content of the media
        """
        file_path.write_bytes(content)

    def save_from_content(self, file_path: Path):
        """
        Save the media from the content.

        Args:
            file_path (Path): the path to save the media to
        """
        assert self.content is not None
        self.save_from_bytes(file_path, self.content)

    def save_from_url(self, file_path):
        """
        Save the media from the URL.

        Args:
            file_path (Path): the path to save the media to

        Raises:
            DownloadError: if the media can't be downloaded
        """
        assert self.url is not None
        content: bytes = download_file(self.url)
        self.save_from_bytes(file_path, content)

    def get_content(self) -> bytes:
        """
        Get the content of the media as bytes.

        Returns:
            bytes: the content of the media

        Raises:
            ValueError: if at least one of 'path', 'url', or 'content' is not provided
        """
        if self.content:
            return self.content
        elif self.path:
            self.load_content_from_path()
        elif self.url:
            self.load_content_from_url()
        else:
            raise ValueError(
                "At least one of 'path', 'url', or 'content' must be provided."
            )
        assert self.content is not None
        return self.content

    def load_content_from_path(self):
        """
        Load the content of the media from the path.
        """
        assert self.path is not None
        self.content = self.path.read_bytes()

    def load_content_from_url(self):
        """
        Load the content of the media from the URL.

        Raises:
            DownloadError: if the media can't be downloaded
        """
        assert self.url is not None
        self.content = download_file(self.url)

    def __repr__(self) -> str:
        """
        Get the representation of the media.

        Use md5 hash for the content of the media if it is available.

        Returns:
            str: the representation of the media
        """
        content_hash = hashlib.md5(self.content).hexdigest() if self.content else None
        return (
            f"Media(path={self.path}, "
            f"url={self.url}, "
            f"content={content_hash}, "
            f"media_id={self.media_id})"
        )

    def __str__(self) -> str:
        """
        Get the string representation of the media.

        Returns:
            str: the string representation of the media
        """
        return self.__repr__()

    def cleanup(self):
        """
        Cleanup the media.

        If the media is saved on disk by the class, delete it.
        """
        if self.is_saved and self.path:
            self.path.unlink(missing_ok=True)
