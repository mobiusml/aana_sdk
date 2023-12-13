from sqlalchemy.orm import Session

from aana.models.db import VideoEntity
from aana.repository.datastore.base import BaseRepository


class VideoRepository(BaseRepository[VideoEntity]):
    """Repository for media files."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, VideoEntity)
