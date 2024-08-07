from aana.core.models.audio import Audio
from aana.core.models.video import Video
from aana.integrations.external.av import load_audio

__all__ = ["extract_audio"]


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
