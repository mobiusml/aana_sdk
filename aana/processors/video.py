from collections import defaultdict
from math import floor

from aana.core.models.asr import AsrSegments
from aana.core.models.audio import Audio
from aana.core.models.video import Video
from aana.integrations.external.av import load_audio


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
        media_id=f"audio_{video.media_id}",
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
