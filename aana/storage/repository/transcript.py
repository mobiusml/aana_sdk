from sqlalchemy.orm import Session

from aana.core.models.media import MediaId
from aana.exceptions.db import NotFoundException
from aana.storage.models.transcript import TranscriptEntity
from aana.storage.repository.base import BaseRepository


class TranscriptRepository(BaseRepository[TranscriptEntity]):
    """Repository for Transcripts."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session=session, model_class=TranscriptEntity)

    def get_transcript(self, model_name: str, media_id: MediaId) -> TranscriptEntity:
        """Get the transcript for a video.

        Args:
            model_name (str): The name of the model used to generate the transcript.
            media_id (MediaId): The media ID.

        Returns:
            TranscriptEntity: The transcript entity.

        Raises:
            NotFoundException: The transcript does not exist.
        """
        entity: TranscriptEntity | None = (
            self.session.query(self.model_class)
            .filter_by(model=model_name, media_id=media_id)
            .first()
        )
        if not entity:
            raise NotFoundException(self.table_name, media_id)
        return entity
