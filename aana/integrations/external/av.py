import gc
import io
import itertools
import wave
from collections.abc import Generator
from pathlib import Path
from typing import TypedDict

import av
import numpy as np

from aana.core.libraries.audio import AbstractAudioLibrary
from aana.core.models.image import Image
from aana.core.models.stream import StreamInput
from aana.exceptions.io import StreamReadingException


class FramesDict(TypedDict):
    """Represents a set of frames with ids, timestamps."""

    frames: list[Image]
    timestamps: list[float]
    frame_ids: list[int]


def load_audio(file: Path | None, sample_rate: int = 16000) -> bytes:
    """Open an audio file and read as mono waveform, resampling as necessary.

    Args:
        file (Path): The audio/video file to open.
        sample_rate (int): The sample rate to resample the audio if necessary.

    Returns:
        bytes: The content of the audio as bytes.

    Raises:
        RuntimeError: if ffmpeg fails to convert and load the audio.
    """
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=sample_rate,
    )

    raw_buffer = io.BytesIO()

    # Try loading audio and check for empty audio in one shot.
    try:
        with av.open(str(file), mode="r", metadata_errors="ignore") as container:
            # check for empty audio
            if container.streams.audio == tuple():
                return b""

            frames = container.decode(audio=0)
            frames = ignore_invalid_frames(frames)
            frames = group_frames(frames, 500000)
            frames = resample_frames(frames, resampler)

            for frame in frames:
                array = frame.to_ndarray()
                raw_buffer.write(array)

        # It appears that some objects related to the resampler are not freed
        # unless the garbage collector is manually run.
        del resampler
        gc.collect()

        return raw_buffer.getvalue()

    except Exception as e:
        raise RuntimeError(f"{e!s}") from e


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


def fetch_stream_frames(
    stream_input: StreamInput, batch_size: int = 2
) -> Generator[FramesDict, None, None]:
    """Generate frames from a video using decord.

    Args:
        stream_input (StreamInput): the video stream to fetch frames from
        batch_size (int): the number of frames to yield at each iteration
    Yields:
        FramesDict: a dictionary containing the extracted frames, frame ids, timestamps, and duration for each batch
    """
    stream_url = stream_input.url
    channel = stream_input.channel_number
    extraction_fps = stream_input.extract_fps

    try:
        stream_container = av.open(stream_url)
    except Exception as e:
        raise StreamReadingException(stream_url) from e

    available_streams = [s for s in stream_container.streams if s.type == "video"]

    # Check the stream channel be valid
    if len(available_streams) == 0 or channel >= len(available_streams):
        raise StreamReadingException(
            stream_url,
            msg=f"selected channel does not exist: {channel + 1} from {len(available_streams)}",
        )
    video_stream = available_streams[channel]

    avg_rate = float(video_stream.average_rate)

    if extraction_fps > avg_rate:
        extraction_fps = avg_rate

    frame_rate = int(avg_rate / extraction_fps)

    # read frames from the stream
    frame_number = 0
    batch_frames = []
    batch_timestamps = []
    num_batches = 0

    for packet in stream_container.demux(video_stream):
        for frame in packet.decode():
            if frame_number % frame_rate == 0:
                img = Image(numpy=frame.to_rgb().to_ndarray())
                packet_timestamp = float(frame.pts * frame.time_base)  # in seconds
                batch_frames.append(img)
                batch_timestamps.append(packet_timestamp)
            frame_number += 1
            if len(batch_frames) == batch_size:
                num_batches += 1
                yield FramesDict(
                    frames=batch_frames,
                    frame_ids=list(
                        range(num_batches * batch_size, (num_batches + 1) * batch_size)
                    ),
                    timestamps=batch_timestamps,
                )
                batch_frames = []
                batch_timestamps = []


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
