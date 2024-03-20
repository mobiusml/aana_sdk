import hashlib  # noqa: I001
from dataclasses import dataclass
import io, gc
import itertools
from pathlib import Path
from collections.abc import Generator
import torch, decord  # noqa: F401  # See https://github.com/dmlc/decord/issues/263
from decord import DECORDError
import numpy as np
import wave
from aana.configs.settings import settings
from aana.exceptions.general import AudioReadingException
from aana.models.core.media import Media
from aana.utils.general import download_file
import av


class AbstractAudioLibrary:
    """Abstract class for audio libraries."""

    @classmethod
    def read_file(cls, path: Path) -> np.ndarray:
        """Read an audio file from path and return as numpy audio array.

        Args:
            path (Path): The path of the file to read.

        Returns:
            np.ndarray: The file as a numpy array.
        """
        raise NotImplementedError

    @classmethod
    def read_from_bytes(cls, content: bytes) -> np.ndarray:
        """Read bytes using the image library.

        Args:
            content (bytes): The content of the file to read.

        Returns:
            np.ndarray: The file as a numpy array.
        """
        raise NotImplementedError

    @classmethod
    def write_file(cls, path: Path, audio: np.ndarray):
        """Write a file using the audio library.

        Args:
            path (Path): The path of the file to write.
            audio (np.ndarray): The audio to write.
        """
        raise NotImplementedError

    @classmethod
    def write_to_bytes(cls, audio: np.ndarray) -> bytes:
        """Write bytes using the audio library.

        Args:
            audio (np.ndarray): The audio to write.

        Returns:
            bytes: The audio as bytes.
        """
        raise NotImplementedError

    @classmethod
    def write_audio_bytes(cls, path: Path, audio: bytes, sample_rate: int = 16000):
        """Write a file to wav from the normalized audio bytes.

        Args:
            path (Path): The path of the file to write.
            audio (bytes): The audio to in 16-bit PCM byte write.
            sample_rate (int): The sample rate of the audio.
        """
        raise NotImplementedError


class pyAVWrapper(AbstractAudioLibrary):
    """Class for audio handling using PyAV library."""

    @classmethod
    def read_file(cls, path: Path, sample_rate: int = 16000) -> np.ndarray:
        """Read an audio file from path and return it as a numpy array.

        Args:
            path (Path): The path of the file to read.
            sample_rate (int): sample rate of the audio, default is 16000.

        Returns:
            np.ndarray: The audio file as a numpy array.
        """
        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=sample_rate,
        )

        raw_buffer = io.BytesIO()
        dtype = None

        with av.open(str(path), mode="r", metadata_errors="ignore") as container:
            frames = container.decode(audio=0)
            frames = ignore_invalid_frames(frames)
            frames = group_frames(frames, 500000)
            frames = resample_frames(frames, resampler)

            for frame in frames:
                array = frame.to_ndarray()
                dtype = array.dtype
                raw_buffer.write(array)

        # It appears that some objects related to the resampler are not freed
        # unless the garbage collector is manually run.
        del resampler
        gc.collect()

        audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)
        # Convert s16 back to f32.
        audio = audio.astype(np.float32) / 32768.0
        return audio

    @classmethod
    def read_from_bytes(cls, content: bytes, sample_rate: int = 16000) -> np.ndarray:
        """Read audio bytes and return as a numpy array.

        Args:
            content (bytes): The content of the file to read.
            sample_rate (int): sample rate of the audio, default is 16000.

        Returns:
            np.ndarray: The file as a numpy array.
        """
        # Create an in-memory file-like object
        content_io = io.BytesIO(content)

        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=sample_rate,
        )

        raw_buffer = io.BytesIO()
        dtype = None

        with av.open(content_io, mode="r", metadata_errors="ignore") as container:
            frames = container.decode(audio=0)
            frames = ignore_invalid_frames(frames)
            frames = group_frames(frames, 500000)
            frames = resample_frames(frames, resampler)

            for frame in frames:
                array = frame.to_ndarray()
                dtype = array.dtype
                raw_buffer.write(array)

        # It appears that some objects related to the resampler are not freed
        # unless the garbage collector is manually run.
        del resampler
        gc.collect()

        audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)
        # Convert s16 back to f32.
        audio = audio.astype(np.float32) / 32768.0
        return audio

    @classmethod
    def write_file(cls, path: Path, audio: np.ndarray, sample_rate: int = 16000):
        """Write an audio file in wav format to the path from numpy array.

        Args:
            path (Path): The path of the file to write.
            audio (np.ndarray): The audio to write.
            sample_rate (int): The sample rate of the audio to save, default is 16000.
        """
        audio = (audio * 32768.0).astype(np.int16)
        # Create an AV container
        container = av.open(str(path), "w", format="wav")
        # Add an audio stream
        stream = container.add_stream("pcm_s16le", rate=sample_rate)
        stream.channels = 1
        # Write audio frames to the stream
        for frame in av.AudioFrame.from_ndarray(
            audio, format="s16", layout="mono", rate=sample_rate
        ):
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        container.close()

    @classmethod
    def write_to_bytes(cls, audio: np.ndarray) -> bytes:
        """Write bytes using the audio library from numpy array.

        Args:
            audio (np.ndarray): The audio to write.

        Returns:
            bytes: The audio as bytes.
        """
        frame = av.AudioFrame(format="s16", layout="mono", samples=len(audio))
        frame.planes[0].update(audio.astype(np.int16).tobytes())
        return frame.planes[0].to_bytes()

    @classmethod
    def write_audio_bytes(cls, path: Path, audio: bytes, sample_rate: int = 16000):
        """Write an audio file in wav format to path from the normalized audio bytes.

        Args:
            path (Path): The path of the file to write.
            audio (bytes): The audio to in 16-bit PCM byte write.
            sample_rate (int): The sample rate of the audio, default is 16000.
        """
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono audio
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)  # Sample rate
            wav_file.writeframes(audio)


def ignore_invalid_frames(frames: Generator) -> Generator:
    """Filter out invalid frames from the input generator.

    Args:
        frames (Generator): The input generator of frames.

    Yields:
        av.audio.frame.AudioFrame: Valid audio frames.

    Raises:
        StopIteration: When the input generator is exhausted.
    """
    iterator = iter(frames)

    while True:
        try:
            yield next(iterator)
        except StopIteration:  # noqa: PERF203
            break
        except av.error.InvalidDataError:
            continue


def group_frames(frames: Generator, num_samples: int | None = None) -> Generator:
    """Group audio frames and yield groups of frames based on the specified number of samples.

    Args:
        frames (Generator): The input generator of audio frames.
        num_samples (int | None): The target number of samples for each group.

    Yields:
        av.audio.frame.AudioFrame: Grouped audio frames.
    """
    fifo = av.audio.fifo.AudioFifo()

    for frame in frames:
        frame.pts = None  # Ignore timestamp check.
        fifo.write(frame)

        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()

    if fifo.samples > 0:
        yield fifo.read()


def resample_frames(frames: Generator, resampler) -> Generator:
    """Resample audio frames using the provided resampler.

    Args:
        frames (Generator): The input generator of audio frames.
        resampler: The audio resampler.

    Yields:
        av.audio.frame.AudioFrame: Resampled audio frames.
    """
    # Add None to flush the resampler.
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)


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
            raise AudioReadingException(video=self)

    def is_audio(self) -> bool:
        """Checks if it's a valid audio."""
        if not self.path:
            return False
        try:
            decord.AudioReader(str(self.path))
        except DECORDError:
            return False
        return True

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
