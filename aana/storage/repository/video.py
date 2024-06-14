from sqlalchemy.orm import Session

from aana.storage.models import VideoEntity
from aana.storage.repository.base import BaseRepository


class VideoRepository(BaseRepository[VideoEntity]):
    """Repository for media files."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, VideoEntity)
