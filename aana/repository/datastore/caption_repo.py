from sqlalchemy.orm import Session

from aana.models.db.caption import Caption
from aana.repository.datastore.base import BaseRepository


class CaptionRepository(BaseRepository[Caption]):
    """Repository for Captions."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, Caption)
