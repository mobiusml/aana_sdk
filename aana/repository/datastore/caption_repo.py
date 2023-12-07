from sqlalchemy.orm import Session

from aana.models.db import CaptionEntity
from aana.repository.datastore.base import BaseRepository


class CaptionRepository(BaseRepository[CaptionEntity]):
    """Repository for Captions."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, CaptionEntity)
