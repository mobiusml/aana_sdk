from enum import Enum

from sqlalchemy import Column, Float, String
from sqlalchemy.orm import relationship

from aana.configs.db import IdSqlType, id_type
from aana.models.db.base import BaseModel, TimeStampEntity


class MediaType(str, Enum):
    """Enum for types of media file."""

    VIDEO = "video"


class Media(BaseModel, TimeStampEntity):
    """ORM class for media file (video, etc)."""

    __tablename__ = "media"

    id: id_type = Column(IdSqlType, primary_key=True)  # noqa: A003
    duration = Column(Float, comment="Media duration in seconds")
    media_type = Column(String, comment="Media type")

    captions = relationship("Caption", back_populates="media")
    transcripts = relationship("Transcript", back_populates="media")
