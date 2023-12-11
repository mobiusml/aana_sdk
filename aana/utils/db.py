# ruff: noqa: A002
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from aana.configs.db import id_type
from aana.models.core.video import Video
from aana.models.db import CaptionEntity, MediaEntity, MediaType, TranscriptEntity
from aana.models.pydantic.asr_output import (
    AsrSegments,
    AsrTranscriptionInfoList,
    AsrTranscriptionList,
)
from aana.models.pydantic.captions import CaptionsList
from aana.models.pydantic.video_params import VideoParams
from aana.repository.datastore.caption_repo import CaptionRepository
from aana.repository.datastore.engine import engine
from aana.repository.datastore.media_repo import MediaRepository
from aana.repository.datastore.transcript_repo import TranscriptRepository


# Just using raw utility functions like this isn't a permanent solution, but
# it's good enough for now to validate what we're working on.
def save_video_batch(
    video_objects: list[Video], video_params: list[VideoParams]
) -> list[id_type]:
    """Saves a batch of videos to datastore."""
    entities = []
    for video_object, _ in zip(video_objects, video_params, strict=True):
        if video_object.url is not None:
            orig_url = video_object.url
            parsed_url = urlparse(orig_url)
            orig_filename = Path(parsed_url.path).name
        elif video_object.path is not None:
            parsed_path = Path(video_object.path)
            orig_filename = parsed_path.name
            orig_url = None
        else:
            orig_url = None
            orig_filename = None
        entity = MediaEntity(
            id=video_object.media_id,
            media_type=MediaType.VIDEO,
            orig_filename=orig_filename,
            orig_url=orig_url,
        )
        entities.append(entity)
    with Session(engine) as session:
        repo = MediaRepository(session)
        results = repo.create_multiple(entities)
    return [result.id for result in results]  # type: ignore


def save_media(media_type: MediaType, duration: float) -> id_type:
    """Creates and saves media to datastore.

    Args:
        media_type (MediaType): type of media
        duration (float): duration of media

    Returns:
        id_type: datastore id of the inserted Media.
    """
    with Session(engine) as session:
        media = MediaEntity(duration=duration, media_type=media_type)
        repo = MediaRepository(session)
        media = repo.create(media)
        return media.id  # type: ignore


def save_captions_batch(
    media_ids: list[id_type],
    model_name: str,
    captions: CaptionsList,
    timestamps: list[float],
    frame_ids: list[int],
) -> list[id_type]:
    """Save captions."""
    with Session(engine) as session:
        entities = [
            CaptionEntity.from_caption_output(model_name, media_id, frame_id, t, c)
            for media_id, c, t, frame_id in zip(
                media_ids, captions, timestamps, frame_ids, strict=True
            )
        ]
        repo = CaptionRepository(session)
        results = repo.create_multiple(entities)
        return [c.id for c in results]  # type: ignore


def save_transcripts_batch(
    model_name: str,
    media_ids: list[id_type],
    transcript_info: AsrTranscriptionInfoList,
    transcripts: AsrTranscriptionList,
    segments: AsrSegments,
) -> list[id_type]:
    """Save transcripts."""
    with Session(engine) as session:
        entities = [
            TranscriptEntity.from_asr_output(model_name, media_id, info, txn, seg)
            for media_id, info, txn, seg in zip(
                media_ids, transcript_info, transcripts, segments, strict=True
            )
        ]

        repo = TranscriptRepository(session)
        entities = repo.create_multiple(entities)
        return [c.id for c in entities]  # type: ignore
