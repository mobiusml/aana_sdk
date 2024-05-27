import hashlib  # noqa: I001
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from aana.configs.settings import settings
from aana.exceptions.io import AudioReadingException
from aana.integrations.external.av import pyAVWrapper
from aana.integrations.external.decord import is_audio
from aana.core.models.media import Media
from aana.core.libraries.audio import AbstractAudioLibrary
from aana.utils.download import download_file


@dataclass
class Audio(Media):
    """A class representing an audio.

    At least one of 'path', 'url', or 'content' must be provided.
    If 'save_on_disk' is True, the audio will be saved on disk automatically.

    Attributes:
        path (Path): the path to the audio file
        url (str): the URL of the audio
        content (bytes): the content of the audio in bytes
        media_id (str): the ID of the audio. If not provided, it will be generated automatically.
        title (str): the title of the audio
        description (str): the description of the audio
        audio_dir (Path): the directory to save the audio in
    """

    title: str = ""
    description: str = ""
    audio_dir: Path | None = settings.audio_dir
    numpy: np.ndarray | None = None
    audio_lib: type[AbstractAudioLibrary] = pyAVWrapper

    def validate(self):
        """Validate the audio.

        Raises:
            ValueError: if none of 'path', 'url', or 'content' is provided
            AudioReadingException: if the audio is not valid
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

        # check that the audio is valid
        if self.path and not self.is_audio():
            raise AudioReadingException(audio=self)

    def is_audio(self) -> bool:
        """Checks if it's a valid audio."""
        if not self.path:
            return False
        return is_audio(self.path)

    def save(self):
        """Save the audio on disk.

        If the audio is already available on disk, do nothing.
        If the audio represented as a byte string, save it on disk.
        If the audio is represented as a URL, download it and save it on disk.
        If the audio is represented as a numpy array, convert it to audio wav and save it on disk.

        First check if the audio is already available on disk, then content, then url, then numpy
        to avoid unnecessary operations (e.g. downloading the audio or converting it to wav).

        Raises:
            ValueError: If none of 'path', 'url', 'content' or 'numpy' is provided.
        """
        if self.path:
            return

        audio_dir = settings.audio_dir
        audio_dir.mkdir(parents=True, exist_ok=True)
        file_path = audio_dir / (self.media_id + ".wav")
        if self.content is not None:
            self.save_from_audio_content(file_path)
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

    def save_from_audio_content(self, file_path: Path):
        """Save the audio from the content.

        Args:
            file_path (Path): the path to save the audio to
        """
        assert self.content is not None  # noqa: S101
        self.audio_lib.write_audio_bytes(file_path, self.content)

    def save_from_audio_url(self, file_path):
        """Save the audio from the URL.

        Args:
            file_path (Path): the path to save the audio to

        Raises:
            DownloadError: if the audio can't be downloaded
        """
        assert self.url is not None  # noqa: S101
        content: bytes = download_file(self.url)
        self.audio_lib.write_audio_bytes(file_path, content)

    def save_from_numpy(self, file_path: Path):
        """Save the audio from numpy on disk.

        Args:
            file_path (Path): The path of the file to write.
        """
        assert self.numpy is not None  # noqa: S101 TODO
        self.audio_lib.write_file(file_path, self.numpy)

    def get_numpy(self) -> np.ndarray:
        """Load the audio as a numpy array.

        Returns:
            np.ndarray: The audio as a numpy array.

        Raises:
            ValueError: If none of 'path', 'url', 'content' or 'numpy' is provided.
            AudioReadingException: If there is an error reading the audio.
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

    def load_numpy_from_path(self):
        """Load the audio as a numpy array from a path.

        Raises:
            AudioReadingException: If there is an error reading the audio.
        """
        assert self.path is not None  # noqa: S101
        try:
            self.numpy = self.audio_lib.read_file(self.path)
        except Exception as e:
            raise AudioReadingException(self) from e

    def load_numpy_from_audio_bytes(self, audio_bytes: bytes):
        """Load the image as a numpy array from image bytes (downloaded from URL or read from file).

        Raises:
            ImageReadingException: If there is an error reading the image.
        """
        try:
            self.numpy = self.audio_lib.read_from_bytes(audio_bytes)
        except Exception as e:
            raise AudioReadingException(self) from e

    def load_numpy_from_url(self):
        """Load the audio as a numpy array from a URL.

        Raises:
            AudioReadingException: If there is an error reading the audio.
        """
        assert self.url is not None  # noqa: S101
        content: bytes = download_file(self.url)
        self.load_numpy_from_audio_bytes(content)

    def load_numpy_from_content(self):
        """Load the image as a numpy array from content.

        Raises:
            ImageReadingException: If there is an error reading the image.
        """
        assert self.content is not None  # noqa: S101
        self.load_numpy_from_audio_bytes(self.content)

    def __repr__(self) -> str:
        """Get the representation of the audio.

        Use md5 hash for the content of the audio if it is available.

        Returns:
            str: the representation of the audio
        """
        content_hash = (
            hashlib.md5(self.content, usedforsecurity=False).hexdigest()
            if self.content
            else None
        )
        return (
            f"Audio(path={self.path}, "
            f"url={self.url}, "
            f"content={content_hash}, "
            f"media_id={self.media_id}, "
            f"title={self.title}, "
            f"description={self.description})"
        )
