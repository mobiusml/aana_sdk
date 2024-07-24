from uuid import uuid4

from sqlalchemy.orm import Mapped, mapped_column

from aana.core.models.media import MediaId
from aana.storage.models.base import BaseEntity, TimeStampEntity


class MediaEntity(BaseEntity, TimeStampEntity):
    """Table for media items."""

    __tablename__ = "media"
    id: Mapped[MediaId] = mapped_column(
        primary_key=True,
        default=lambda: str(uuid4()),
        comment="Unique identifier for the media",
    )
    media_type: Mapped[str] = mapped_column(comment="The type of media")

    __mapper_args__ = {  # noqa: RUF012
        "polymorphic_identity": "media",
        "polymorphic_on": "media_type",
    }
