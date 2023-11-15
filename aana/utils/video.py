from pathlib import Path
import decord
import numpy as np
import yt_dlp
from yt_dlp.utils import DownloadError
from typing import List, TypedDict
from aana.configs.settings import settings
from aana.exceptions.general import DownloadException, VideoReadingException
from aana.models.core.image import Image
from aana.models.core.video import Video
from aana.models.core.video_source import VideoSource
from aana.models.pydantic.video_input import VideoInput
from aana.models.pydantic.video_params import VideoParams


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


def download_video(video_input: VideoInput) -> Video:
    """
    Downloads videos for a VideoInput object.

    Args:
        video_input (VideoInput): the video input to download

    Returns:
        Video: the video object
    """
    if video_input.url is not None:
        video_source: VideoSource = VideoSource.from_url(video_input.url)
        if video_source == VideoSource.YOUTUBE:
            youtube_video_dir = settings.youtube_video_dir

            ydl_options = {
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
                "outtmpl": f"{youtube_video_dir}/%(id)s.%(ext)s",
                "extract_flat": True,
                "hls_prefer_native": True,
                "extractor_args": {"youtube": {"skip": ["hls", "dash"]}},
            }
            try:
                with yt_dlp.YoutubeDL(ydl_options) as ydl:
                    info = ydl.extract_info(video_input.url, download=False)
                    path = Path(ydl.prepare_filename(info))
                    if not path.exists():
                        ydl.download([video_input.url])
                    if not path.exists():
                        raise DownloadException(video_input.url)
                    return Video(path=path)
            except DownloadError as e:
                raise DownloadException(video_input.url) from e
        elif video_source == VideoSource.AUTO:
            video = Video(url=video_input.url, save_on_disk=True)
            return video
        else:
            raise NotImplementedError(f"Download for {video_source} not implemented")
    else:
        return video_input.convert_input_to_object()
