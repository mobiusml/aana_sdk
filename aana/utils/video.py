import hashlib  # noqa: I001
import json
import pickle
from collections import defaultdict
from collections.abc import Generator
from math import floor
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch, decord  # noqa: F401  # See https://github.com/dmlc/decord/issues/263
import yt_dlp
from yt_dlp.utils import DownloadError

from aana.configs.settings import settings
from aana.exceptions.general import (
    DownloadException,
    MediaIdNotFoundException,
    VideoReadingException,
)
from aana.models.core.image import Image
from aana.models.core.video import Video
from aana.models.pydantic.asr_output import (
    AsrSegments,
    AsrTranscription,
    AsrTranscriptionInfo,
)
from aana.models.pydantic.chat_message import ChatDialog, ChatMessage
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
        video_dir = settings.video_dir
        url_hash = hashlib.md5(
            video_input.url.encode(), usedforsecurity=False
        ).hexdigest()

        # we use yt_dlp to download the video
        # it works not only for youtube videos, but also for other websites and direct links
        ydl_options = {
            "outtmpl": f"{video_dir}/{url_hash}.%(ext)s",
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
                    url=video_input.url,
                    media_id=video_input.media_id,
                    title=title,
                    description=description,
                )
        except DownloadError as e:
            # removes the yt-dlp request to file an issue
            error_message = e.msg.split(";")[0]
            raise DownloadException(url=video_input.url, msg=error_message) from e
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
        video (Video): the video to save the transcription for
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


def load_video_metadata(media_id: str):
    """Loads the metadata of the video from a file.

    Args:
        media_id: the id of the video

    Returns:
        dict: dictionary containing the metadata
    """
    output_dir = settings.tmp_data_dir / "video_metadata"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_dir) / f"{media_id}.pkl"
    if not output_path.exists():
        raise MediaIdNotFoundException(media_id)
    with output_path.open("rb") as f:
        metadata = pickle.load(f)  # noqa: S301
    return metadata


def load_video_timeline(media_id: str):
    """Loads the timeline of the video from a file.

    Args:
        media_id: the id of the video

    Returns:
        dict: dictionary containing the timeline
    """
    output_dir = settings.tmp_data_dir / "timelines"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_dir) / f"{media_id}.pkl"
    with output_path.open("rb") as f:
        timeline = pickle.load(f)  # noqa: S301
    return timeline


def generate_dialog(
    metadata: dict,
    timeline: list[dict],
    question: str,
    max_timeline_len: int | None = 1024,
) -> ChatDialog:
    """Generates a dialog from the metadata and timeline of a video.

    Args:
        metadata (dict): the metadata of the video
        timeline (list[dict]): the timeline of the video
        question (str): the question to ask
        max_timeline_len (int, optional): the maximum length of the timeline in tokens.
                                          Defaults to 1024.
                                          If the timeline is longer than this, it will be truncated.
                                          If None, the timeline will not be truncated.

    Returns:
        ChatDialog: the generated dialog
    """
    system_prompt_preamble = """You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while ensuring safety. You will be provided with a script in json format for a video containing information from visual captions and audio transcripts. Each entry in the script follows the format:

    {{
    "start_time":"start_time_in_seconds",
    "end_time": "end_time_in_seconds",
    "audio_transcript": "the_transcript_from_automatic_speech_recognition_system",
    "visual_caption": "the_caption_of_the_visuals_using_computer_vision_system"
    }}
    Note that the audio_transcript can sometimes be empty.

    Ensure you do not introduce any new named entities in your output and maintain the utmost factual accuracy in your responses.

    In the addition you will be provided with title and description of video extracted.
    """
    instruction = (
        "Provide a short and concise answer to the following user's question. "
        "Avoid mentioning any details about the script in JSON format. "
        "For example, a good response would be: 'Based on the analysis, "
        "here are the most relevant/useful/aesthetic moments.' "
        "A less effective response would be: "
        "'Based on the provided visual caption/audio transcript, "
        "here are the most relevant/useful/aesthetic moments. The user question is "
    )

    user_prompt_template = (
        "{instruction}"
        "Given the timeline of audio and visual activities in the video below "
        "I want to find out the following: {question}"
        "The timeline is: "
        "{timeline_json}"
        "\n"
        "The title of the video is {video_title}"
        "\n"
        "The description of the video is {video_description}"
    )

    timeline_json = json.dumps(timeline, indent=4, separators=(",", ": "))
    # truncate the timeline if it is too long
    timeline_tokens = (
        timeline_json.split()
    )  # not an accurate count of tokens, but good enough
    if max_timeline_len is not None and len(timeline_tokens) > max_timeline_len:
        timeline_json = " ".join(timeline_tokens[:max_timeline_len])

    messages = []
    messages.append(ChatMessage(content=system_prompt_preamble, role="system"))
    messages.append(
        ChatMessage(
            content=user_prompt_template.format(
                instruction=instruction,
                question=question,
                timeline_json=timeline_json,
                video_title=metadata["title"],
                video_description=metadata["description"],
            ),
            role="user",
        )
    )

    dialog = ChatDialog(messages=messages)
    return dialog
