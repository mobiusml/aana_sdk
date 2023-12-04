# ruff: noqa: A002
from sqlalchemy.orm import Session

from aana.configs.db import id_type
from aana.models.db import Caption, Media, MediaType, Transcript
from aana.models.pydantic.asr_output import AsrTranscriptionList
from aana.models.pydantic.captions import VideoCaptionsList
from aana.repository.datastore.caption_repo import CaptionRepository
from aana.repository.datastore.engine import engine
from aana.repository.datastore.media_repo import MediaRepository
from aana.repository.datastore.transcript_repo import TranscriptRepository


# Just using raw utility functions like this isn't a permanent solution, but
# it's good enough for now to validate what we're working on.
def save_media(media_type: MediaType, duration: float) -> id_type:
    """Creates and saves media to datastore.

    Arguments:
        media_type (MediaType): type of media
        duration (float): duration of media

    Returns:
        id_type: datastore id of the inserted Media.
    """
    with Session(engine) as session:
        media = Media(duration=duration, media_type=media_type)
        repo = MediaRepository(session)
        media = repo.create(media)
        return media.id  # type: ignore


def save_captions(media_id: id_type, captions: VideoCaptionsList) -> list[id_type]:
    """Save captions."""
    with Session(engine) as session:
        captions_ = [
            Caption(media_id=media_id, frame_id=i, caption=c)
            for i, c in enumerate(captions)
        ]
        repo = CaptionRepository(session)
        results = repo.create_multiple(captions_)
        return [c.id for c in results]  # type: ignore


def save_transcripts(
    media_id: id_type, transcripts: AsrTranscriptionList
) -> list[id_type]:
    """Save transcripts."""
    with Session(engine) as session:
        entities = [
            Transcript(media_id=media_id, transcript=t.text) for t in transcripts
        ]
        repo = TranscriptRepository(session)
        entities = repo.create_multiple(entities)
        return [c.id for c in entities]  # type: ignore
