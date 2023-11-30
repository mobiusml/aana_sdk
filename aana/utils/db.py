# ruff: noqa: A002
from sqlalchemy.orm import Session

from aana.configs.db import id_type
from aana.models.db.caption import Caption
from aana.models.db.media import Media, MediaType
from aana.models.db.transcript import Transcript
from aana.models.pydantic.asr_output import AsrTranscriptionList
from aana.models.pydantic.captions import VideoCaptionsList
from aana.repository.datastore.caption_repo import CaptionRepository
from aana.repository.datastore.engine import engine
from aana.repository.datastore.media_repo import MediaRepository
from aana.repository.datastore.transcript_repo import TranscriptRepository


# Just using raw utility functions like this isn't a permanent solution, but
# it's good enough for now to validate what we're working on.
def save_media(type: MediaType, duration: float) -> id_type:
    """Creates and saves media to datastore.

    Arguments:
        type (MediaType): type of media
        duration (float): duration of media

    Returns:
        id_type: datastore id of the inserted Media.
    """
    with Session(engine) as session:
        media = Media(duration=duration, type=type)
        repo = MediaRepository(session)
        media = repo.create(media)
        return id_type(media.id)

def save_captions(media_id: id_type, captions: VideoCaptionsList) -> list[id_type]:
    """Save captions."""
    with Session(engine) as session:
        captions_ = [Caption(media_id=media_id, **c.dict()) for c in captions]
        repo = CaptionRepository(session)
        results = repo.create_multiple(captions_)
        return [id_type(c.id) for c in results]
    

def save_transcripts(media_id: id_type, transcripts: AsrTranscriptionList) -> list[id_type]:
    """Save transcripts."""
    with Session(engine) as session:
        entities = [Transcript(media_id=media_id, **t.dict()) for t in transcripts]
        repo = TranscriptRepository(session)
        entities = repo.create_multiple(entities)
        return [id_type(c.id) for c in entities]
