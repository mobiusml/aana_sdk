from sqlalchemy.orm import Session

from aana.models.db.transcript import Transcript
from aana.repository.datastore.base import BaseRepository


class TranscriptRepository(BaseRepository[Transcript]):
    """Repository for Transcripts."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session=session, model_class=Transcript)
