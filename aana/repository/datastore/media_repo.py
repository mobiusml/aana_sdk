from sqlalchemy.orm import Session

from aana.models.db import MediaEntity
from aana.repository.datastore.base import BaseRepository


class MediaRepository(BaseRepository[MediaEntity]):
    """Repository for media files."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, MediaEntity)
