from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from aana.configs.db import MediaIdSqlType
from aana.models.db.base import BaseModel, TimeStampEntity


class VideoEntity(BaseModel, TimeStampEntity):
    """ORM class for videp file (video, etc)."""

    __tablename__ = "video"

    id = Column(Integer, primary_key=True)  # noqa: A003
    media_id = Column(
        MediaIdSqlType,
        ForeignKey("media.id"),
        nullable=False,
        comment="Foreign key to media table",
    )
    duration = Column(Float, comment="Media duration in seconds")
    media_type = Column(String, comment="Media type")
    orig_filename = Column(String, comment="Original filename")
    orig_url = Column(String, comment="Original URL")

    media = relationship("MediaEntity", foreign_keys=[media_id])
    captions = relationship("CaptionEntity", back_populates="video")
    transcripts = relationship("TranscriptEntity", back_populates="video")
