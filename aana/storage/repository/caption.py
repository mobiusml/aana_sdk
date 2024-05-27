from sqlalchemy.orm import Session

from aana.core.models.media import MediaId
from aana.storage.models.caption import CaptionEntity
from aana.storage.repository.base import BaseRepository


class CaptionRepository(BaseRepository[CaptionEntity]):
    """Repository for Captions."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, CaptionEntity)

    def get_captions(self, model_name: str, media_id: MediaId) -> list[CaptionEntity]:
        """Get the captions for a video.

        Args:
            model_name (str): The model name.
            media_id (MediaId): The media ID.

        Returns:
            list[CaptionEntity]: The list of caption entities.
        """
        entities: list[CaptionEntity] = (
            self.session.query(self.model_class)
            .filter_by(media_id=media_id, model=model_name)
            .order_by(self.model_class.frame_id)
            .all()
        )
        return entities
