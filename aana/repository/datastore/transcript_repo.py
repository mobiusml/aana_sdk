from sqlalchemy.orm import Session

from aana.models.db import TranscriptEntity
from aana.repository.datastore.base import BaseRepository


class TranscriptRepository(BaseRepository[TranscriptEntity]):
    """Repository for Transcripts."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session=session, model_class=TranscriptEntity)
