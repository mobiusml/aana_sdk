import decord
import numpy as np
from aana.exceptions.general import VideoReadingException
from aana.models.core.image import Image
from aana.models.core.video import Video
from aana.models.pydantic.video_params import VideoParams
from typing import List, TypedDict


class FramesDict(TypedDict):
    frames: List[Image]
    timestamps: List[float]
    duration: float


def extract_frames_decord(video: Video, params: VideoParams) -> FramesDict:
    """
    Extract frames from a video using decord.

    Args:
        video (Video): the video to extract frames from
        params (VideoParams): the parameters of the video extraction

    Returns:
        FramesDict: a dictionary containing the extracted frames, timestamps, and duration
    """
    device = decord.cpu(0)
    num_threads = 1  # TODO: see if we can use more threads

    num_fps: float = params.extract_fps
    try:
        video_reader = decord.VideoReader(
            str(video.path), ctx=device, num_threads=num_threads
        )
    except Exception as e:
        raise VideoReadingException(video) from e

    video_fps = video_reader.get_avg_fps()
    num_frames = len(video_reader)
    duration = num_frames / video_fps

    if params.fast_mode_enabled:
        indexes = video_reader.get_key_indices()
    else:
        # num_fps can be smaller than 1 (e.g. 0.5 means 1 frame every 2 seconds)
        indexes = np.arange(0, num_frames, int(video_fps / num_fps))
    timestamps = video_reader.get_frame_timestamp(indexes)[:, 0].tolist()
    timestamps.append(duration)

    frames_array = video_reader.get_batch(indexes).asnumpy()
    frames = []
    for _, frame in enumerate(frames_array):
        img = Image(numpy=frame)
        frames.append(img)

    return FramesDict(frames=frames, timestamps=timestamps, duration=duration)
