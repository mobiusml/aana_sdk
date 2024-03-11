import gc
import hashlib
import io
import itertools
import json
from collections import defaultdict
from collections.abc import Generator
from math import floor
from pathlib import Path
from typing import Optional, TypedDict

import av
import decord
import numpy as np
import torch  # noqa: F401  # See https://github.com/dmlc/decord/issues/263
import yt_dlp
from decord import DECORDError
from yt_dlp.utils import DownloadError

from aana.configs.settings import settings
from aana.exceptions.general import (
    DownloadException,
    VideoReadingException,
)
from aana.models.core.audio import Audio
from aana.models.core.image import Image
from aana.models.core.video import Video
from aana.models.pydantic.asr_output import (
    AsrSegments,
)
from aana.models.pydantic.chat_message import ChatDialog, ChatMessage
from aana.models.pydantic.video_input import VideoInput
from aana.models.pydantic.video_metadata import VideoMetadata
from aana.models.pydantic.video_params import VideoParams


class FramesDict(TypedDict):
    """Represents a set of frames with ids, timestamps and total duration."""

    frames: list[Image]
    timestamps: list[float]
    duration: float
    frame_ids: list[int]


def extract_frames_decord(video: Video, params: VideoParams) -> FramesDict:
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


def generate_frames_decord(
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


def _ignore_invalid_frames(frames: Generator) -> Generator:
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


def _group_frames(frames: Generator, num_samples: int | None = None) -> Generator:
    """Group audio frames and yield groups of frames based on the specified number of samples.

    Args:
        frames (Generator): The input generator of audio frames.
        num_samples (Optional[int]): The target number of samples for each group.

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


def _resample_frames(frames: Generator, resampler) -> Generator:
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
            frames = _ignore_invalid_frames(frames)
            frames = _group_frames(frames, 500000)
            frames = _resample_frames(frames, resampler)

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


def extract_audio(video: Video) -> Audio:
    """Extract the audio file from a Video and return an Audio object.

    Args:
        video (Video): The video file to extract audio.

    Returns:
        Audio: an Audio object containing the extracted audio.

    """
    audio_bytes = load_audio(video.path)

    # Only difference will be in path where WAV file will be stored
    # and in content but has same media_id
    return Audio(
        url=video.url,
        media_id=video.media_id,
        content=audio_bytes,
        title=video.title,
        description=video.description,
    )


def generate_combined_timeline(
    transcription_segments: AsrSegments,
    captions: list[str],
    caption_timestamps: list[float],
    chunk_size: float = 10.0,
):
    """Generates a combined timeline from the ASR segments and the captions.

    Args:
        transcription_segments (AsrSegments): the ASR segments
        captions (list[str]): the captions
        caption_timestamps (list[float]): the timestamps for the captions
        chunk_size (float, optional): the chunk size for the combined timeline in seconds. Defaults to 10.0.

    Returns:
        dict: dictionary containing one key, "timeline", which is a list of dictionaries with the following keys:
            "start_time": the start time of the chunk in seconds
            "end_time": the end time of the chunk in seconds
            "audio_transcript": the audio transcript for the chunk
            "visual_caption": the visual caption for the chunk
    """
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

    return {
        "timeline": timeline,
    }


def generate_dialog(
    metadata: VideoMetadata,
    timeline: list[dict],
    question: str,
    max_timeline_len: int | None = 1024,
) -> ChatDialog:
    """Generates a dialog from the metadata and timeline of a video.

    Args:
        metadata (VideoMetadata): the metadata of the video
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
                video_title=metadata.title,
                video_description=metadata.description,
            ),
            role="user",
        )
    )

    dialog = ChatDialog(messages=messages)
    return dialog
