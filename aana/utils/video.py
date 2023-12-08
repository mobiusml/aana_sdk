import pickle  # noqa: I001
from collections import defaultdict
from collections.abc import Generator
from math import floor
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch, decord  # See https://github.com/dmlc/decord/issues/263  # noqa: F401
import yt_dlp
from yt_dlp.utils import DownloadError

from aana.configs.settings import settings
from aana.exceptions.general import DownloadException, VideoReadingException
from aana.models.core.image import Image
from aana.models.core.video import Video
from aana.models.core.video_source import VideoSource
from aana.models.pydantic.asr_output import (
    AsrSegments,
    AsrTranscription,
    AsrTranscriptionInfo,
)
from aana.models.pydantic.video_input import VideoInput
from aana.models.pydantic.video_params import VideoParams


class FramesDict(TypedDict):
    """Represents a set of frames with timestamps and total duration."""

    frames: list[Image]
    timestamps: list[float]
    duration: float


def extract_frames_decord(video: Video, params: VideoParams) -> FramesDict:
    """Extract frames from a video using decord.

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

    frames_array = video_reader.get_batch(indexes).asnumpy()
    frames = []
    for _, frame in enumerate(frames_array):
        img = Image(numpy=frame)
        frames.append(img)

    return FramesDict(frames=frames, timestamps=timestamps, duration=duration)


def generate_frames_decord(
    video: Video, params: VideoParams, batch_size: int = 8
) -> Generator[FramesDict, None, None]:
    """Generate frames from a video using decord.

    Args:
        video (Video): the video to extract frames from
        params (VideoParams): the parameters of the video extraction
        batch_size (int): the number of frames to yield at each iteration

    Yields:
        FramesDict: a dictionary containing the extracted frames, timestamps,
                    and duration for each batch
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

    for i in range(0, len(indexes), batch_size):
        batch = indexes[i : i + batch_size]
        batch_frames_array = video_reader.get_batch(batch).asnumpy()
        batch_frames = []
        for _, frame in enumerate(batch_frames_array):
            img = Image(numpy=frame)
            batch_frames.append(img)

        batch_timestamps = timestamps[i : i + batch_size]
        yield FramesDict(
            frames=batch_frames, timestamps=batch_timestamps, duration=duration
        )


def download_video(video_input: VideoInput | Video) -> Video:
    """Downloads videos for a VideoInput object.

    Args:
        video_input (VideoInput): the video input to download

    Returns:
        Video: the video object
    """
    if isinstance(video_input, Video):
        return video_input
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
                    title = info.get("title", "")
                    description = info.get("description", "")
                    path = Path(ydl.prepare_filename(info))
                    if not path.exists():
                        ydl.download([video_input.url])
                    if not path.exists():
                        raise DownloadException(video_input.url)
                    return Video(
                        path=path,
                        media_id=video_input.media_id,
                        title=title,
                        description=description,
                    )
            except DownloadError as e:
                raise DownloadException(video_input.url) from e
        elif video_source == VideoSource.AUTO:
            video = Video(
                url=video_input.url, save_on_disk=True, media_id=video_input.media_id
            )
            return video
        else:
            raise NotImplementedError(f"Download for {video_source} not implemented")
    else:
        return video_input.convert_input_to_object()


def save_transcription(
    transcription: AsrTranscription,
    transcription_info: AsrTranscriptionInfo,
    segments: AsrSegments,
    video: Video,
):
    """Saves the transcription to a file.

    Args:
        transcription (AsrTranscription): the transcription to save
        transcription_info (AsrTranscriptionInfo): the transcription info to save
        segments (AsrSegments): the segments to save
        media_id (str): the id of the media
    """
    print(video)
    print(transcription)
    media_id = video.media_id
    output_dir = settings.tmp_data_dir / "transcriptions"
    output_dir.mkdir(parents=True, exist_ok=True)
    # dump the transcription to a file as json
    output_path = Path(output_dir) / f"{media_id}.pkl"
    with output_path.open("wb") as f:
        pickle.dump(
            {
                "transcription": transcription,
                "transcription_info": transcription_info,
                "segments": segments,
            },
            f,
        )

    # output_path = Path(output_dir) / f"{media_id}.txt"
    # output_path.write_text(transcription.text)
    return {
        "path": str(output_path),
    }


def save_video_captions(captions: list[str], timestamps: list[float], video: Video):
    """Saves the captions to a file.

    Args:
        captions (list[str]): the captions to save
        timestamps (list[float]): the timestamps to save
        video (Video): the video to save the captions for
    """
    media_id = video.media_id
    output_dir = settings.tmp_data_dir / "captions"
    output_dir.mkdir(parents=True, exist_ok=True)
    # dump the transcription to a file as json
    output_path = Path(output_dir) / f"{media_id}.pkl"
    with output_path.open("wb") as f:
        pickle.dump(
            {
                "captions": captions,
                "timestamps": timestamps,
            },
            f,
        )

    # output_path = Path(output_dir) / f"{media_id}.txt"
    # output_path.write_text(transcription.text)
    return {
        "path": str(output_path),
    }


def save_video_metadata(video: Video):
    """Saves the metadata of the video to a file.

    Args:
        video (Video): the video to save the metadata for

    Returns:
        dict: dictionary containing the path to the saved metadata
    """
    media_id = video.media_id
    output_dir = settings.tmp_data_dir / "video_metadata"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_dir) / f"{media_id}.pkl"
    with output_path.open("wb") as f:
        pickle.dump(
            {
                "title": video.title,
                "description": video.description,
            },
            f,
        )
    return {
        "path": str(output_path),
    }


def generate_combined_timeline(
    video: Video,
    transcription_segments: AsrSegments,
    captions: list[str],
    caption_timestamps: list[float],
    chunk_size: float = 10.0,
):
    """Generates a combined timeline from the ASR segments and the captions.

    Args:
        video (Video): the video
        transcription_segments (AsrSegments): the ASR segments
        captions (list[str]): the captions
        caption_timestamps (list[float]): the timestamps for the captions
        chunk_size (float, optional): the chunk size for the combined timeline in seconds. Defaults to 10.0.

    Returns:
        list[str]: the combined timeline
    """
    media_id = video.media_id

    timeline_dict: defaultdict[int, dict[str, list[str]]] = defaultdict(
        lambda: {"transcription": [], "captions": []}
    )
    for segment in transcription_segments:
        segment_start = segment.time_interval.start
        chunk_index = floor(segment_start / chunk_size)
        timeline_dict[chunk_index]["transcription"].append(segment.text)

    if len(captions) != len(caption_timestamps):
        raise ValueError(  # noqa: TRY003
            f"Length of captions ({len(captions)}) and timestamps ({len(caption_timestamps)}) do not match"
        )

    for timestamp, caption in zip(caption_timestamps, captions, strict=True):
        chunk_index = floor(timestamp / chunk_size)
        timeline_dict[chunk_index]["captions"].append(caption)

    num_chunks = max(timeline_dict.keys()) + 1

    timeline = [
        {
            "start_time": chunk_index * chunk_size,
            "end_time": (chunk_index + 1) * chunk_size,
            "audio_transcript": "\n".join(timeline_dict[chunk_index]["transcription"]),
            "visual_caption": "\n".join(timeline_dict[chunk_index]["captions"]),
        }
        for chunk_index in range(num_chunks)
    ]

    # save the timeline to a file
    output_dir = settings.tmp_data_dir / "timelines"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"{media_id}.pkl"

    with output_path.open("wb") as f:
        pickle.dump(timeline, f)

    return {
        "path": str(output_path),
        "timeline": timeline,
    }
