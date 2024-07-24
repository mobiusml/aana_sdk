from uuid import uuid4

from sqlalchemy.orm import Mapped, mapped_column

from aana.core.models.media import MediaId
from aana.storage.models.base import BaseEntity, TimeStampEntity


class MediaEntity(BaseEntity, TimeStampEntity):
    """Base ORM class for media (e.g. videos, images, etc.).

    This class is meant to be subclassed by other media types.

    Attributes:
        id (MediaId): Unique identifier for the media.
        media_type (str): The type of media (populated automatically by ORM based on `polymorphic_identity` of   subclass).
    """

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
