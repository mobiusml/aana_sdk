from enum import Enum
from uuid import uuid4

from sqlalchemy import Column, String
from sqlalchemy.orm import relationship

from aana.storage.models.base import BaseEntity, TimeStampEntity
from aana.storage.types import MediaIdSqlType


class MediaType(str, Enum):
    """Enum for types of media file."""

    VIDEO = "video"


class MediaEntity(BaseEntity, TimeStampEntity):
    """Table for media items."""

    __tablename__ = "media"
    id = Column(
        MediaIdSqlType,
        primary_key=True,
        default=lambda: str(uuid4()),
        comment="Unique identifier for the media",
    )
    media_type = Column(String, comment="The type of media")

    video = relationship(
        "VideoEntity",
        back_populates="media",
        cascade="all, delete",
        uselist=False,
    )
