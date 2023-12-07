# ruff: noqa: A002
from sqlalchemy.orm import Session

from aana.configs.db import id_type
from aana.models.db import CaptionEntity, MediaEntity, MediaType, TranscriptEntity
from aana.models.pydantic.asr_output import (
    AsrSegments,
    AsrTranscriptionInfoList,
    AsrTranscriptionList,
)
from aana.models.pydantic.captions import CaptionsList
from aana.repository.datastore.caption_repo import CaptionRepository
from aana.repository.datastore.engine import engine
from aana.repository.datastore.media_repo import MediaRepository
from aana.repository.datastore.transcript_repo import TranscriptRepository


# Just using raw utility functions like this isn't a permanent solution, but
# it's good enough for now to validate what we're working on.
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
) -> list[id_type]:
    """Save captions."""
    with Session(engine) as session:
        entities = [
            CaptionEntity.from_caption_output(model_name, media_id, i, t, c)
            for i, (media_id, c, t) in enumerate(
                zip(media_ids, captions, timestamps, strict=True)
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
