from collections.abc import Generator
from pathlib import Path

import decord
import numpy as np
from decord import DECORDError
from typing_extensions import TypedDict

from aana.core.models.image import Image
from aana.core.models.video import Video, VideoParams
from aana.exceptions.io import VideoReadingException


class FramesDict(TypedDict):
    """Represents a set of frames with ids, timestamps and total duration."""

    frames: list[Image]
    timestamps: list[float]
    duration: float
    frame_ids: list[int]


def extract_frames(video: Video, params: VideoParams) -> FramesDict:
    """Extract frames from a video using decord.

    Args:
        video (Video): the video to extract frames from
        params (VideoParams): the parameters of the video extraction

    Returns:
        FramesDict: a dictionary containing the extracted frames, frame_ids, timestamps, and duration
    """
    device = decord.cpu(0)
    num_threads = 1  # TODO: see if we can use more threads

    num_fps: float = params.extract_fps
    try:
        video_reader = decord.VideoReader(
            str(video.path), ctx=device, num_threads=num_threads
        )
    except DECORDError as video_reader_exception:
        try:
            audio_reader = decord.AudioReader(str(video.path), ctx=device)
            return FramesDict(
                frames=[],
                timestamps=[],
                duration=audio_reader.duration(),
                frame_ids=[],
            )
        except DECORDError:
            raise VideoReadingException(video) from video_reader_exception

    video_fps = video_reader.get_avg_fps()
    num_frames = len(video_reader)
    duration = num_frames / video_fps

    if params.fast_mode_enabled:
        indexes = video_reader.get_key_indices()
    else:
        # num_fps can be smaller than 1 (e.g. 0.5 means 1 frame every 2 seconds)
        indexes = np.arange(0, num_frames, int(video_fps / num_fps))
    timestamps = video_reader.get_frame_timestamp(indexes)[:, 0].tolist()

    frames_array = video_reader.get_batch(indexes).asnumpy()
    frames = []
    for frame_id, frame in enumerate(frames_array):
        img = Image(numpy=frame, media_id=f"{video.media_id}_frame_{frame_id}")
        frames.append(img)

    return FramesDict(
        frames=frames,
        timestamps=timestamps,
        duration=duration,
        frame_ids=list(range(len(frames))),
    )


def get_video_duration(video: Video) -> float:
    """Extract video duration using decord.

    Args:
        video (Video): the video to get its duration

    Returns:
        float: duration of the video

    Raises:
        VideoReadingException: if the file is not readable or a valid multimedia file
    """
    device = decord.cpu(0)
    try:
        video_reader = decord.VideoReader(str(video.path), ctx=device, num_threads=1)
    except DECORDError as video_reader_exception:
        try:
            audio_reader = decord.AudioReader(str(video.path), ctx=device)
            return audio_reader.duration()
        except DECORDError:
            raise VideoReadingException(video) from video_reader_exception

    video_fps = video_reader.get_avg_fps()
    num_frames = len(video_reader)
    duration = num_frames / video_fps
    return duration


def generate_frames(
    video: Video, params: VideoParams, batch_size: int = 8
) -> Generator[FramesDict, None, None]:
    """Generate frames from a video using decord.

    Args:
        video (Video): the video to extract frames from
        params (VideoParams): the parameters of the video extraction
        batch_size (int): the number of frames to yield at each iteration

    Yields:
        FramesDict: a dictionary containing the extracted frames, frame ids, timestamps,
                    and duration for each batch
    Raises:
        VideoReadingException: if the file is not readable or a valid multimedia file
    """
    device = decord.cpu(0)
    num_threads = 1  # TODO: see if we can use more threads

    num_fps: float = params.extract_fps
    is_audio_only = False
    try:
        video_reader = decord.VideoReader(
            str(video.path), ctx=device, num_threads=num_threads
        )
    except DECORDError as video_reader_exception:
        try:
            audio_reader = decord.AudioReader(str(video.path), ctx=device)
            is_audio_only = True
            yield FramesDict(
                frames=[],
                timestamps=[],
                duration=audio_reader.duration(),
                frame_ids=[],
            )

        except DECORDError:
            raise VideoReadingException(video) from video_reader_exception

    if is_audio_only:
        return

    video_fps = video_reader.get_avg_fps()
    num_frames = len(video_reader)
    duration = num_frames / video_fps

    if params.fast_mode_enabled:
        indexes = video_reader.get_key_indices()
    else:
        # num_fps can be smaller than 1 (e.g. 0.5 means 1 frame every 2 seconds)
        indexes = np.arange(0, num_frames, int(video_fps / num_fps))
    timestamps = video_reader.get_frame_timestamp(indexes)[:, 0].tolist()

    for i in range(0, len(indexes), batch_size):
        batch = indexes[i : i + batch_size]
        batch_frames_array = video_reader.get_batch(batch).asnumpy()
        batch_frames = []
        for frame_id, frame in enumerate(batch_frames_array):
            img = Image(numpy=frame, media_id=f"{video.media_id}_frame_{frame_id}")
            batch_frames.append(img)

        batch_timestamps = timestamps[i : i + batch_size]
        yield FramesDict(
            frames=batch_frames,
            frame_ids=list(range(len(batch_frames))),
            timestamps=batch_timestamps,
            duration=duration,
        )


def is_audio(path: Path) -> bool:
    """Checks if it's a valid audio."""
    try:
        decord.AudioReader(str(path))
    except DECORDError:
        return False
    return True
