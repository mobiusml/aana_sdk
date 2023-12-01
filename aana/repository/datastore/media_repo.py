from sqlalchemy.orm import Session

from aana.models.db import Media
from aana.repository.datastore.base import BaseRepository


class MediaRepository(BaseRepository[Media]):
    """Repository for media files."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, Media)
