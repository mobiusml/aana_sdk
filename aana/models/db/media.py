from enum import Enum

from sqlalchemy import Column, Float, String
from sqlalchemy.orm import relationship

from aana.configs.db import IdSqlType, id_type
from aana.models.db.base import BaseModel, TimeStampEntity


class MediaType(str, Enum):
    """Enum for types of media file."""

    VIDEO = "video"


class MediaEntity(BaseModel, TimeStampEntity):
    """ORM class for media file (video, etc)."""

    __tablename__ = "media"

    id = Column(IdSqlType, primary_key=True)  # noqa: A003
    duration = Column(Float, comment="Media duration in seconds")
    media_type = Column(String, comment="Media type")
    orig_filename = Column(String, comment="Original filename")
    orig_url = Column(String, comment="Original URL")

    captions = relationship("CaptionEntity", back_populates="media")
    transcripts = relationship("TranscriptEntity", back_populates="media")
