from pathlib import Path

import numpy as np


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
