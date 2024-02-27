from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from aana.configs.db import MediaIdSqlType
from aana.models.db.base import BaseEntity, TimeStampEntity


class VideoEntity(BaseEntity, TimeStampEntity):
    """ORM class for videp file (video, etc)."""

    __tablename__ = "video"

    id = Column(Integer, primary_key=True)
    media_id = Column(
        MediaIdSqlType,
        ForeignKey("media.id"),
        nullable=False,
        comment="Foreign key to media table",
    )
    duration = Column(Float, comment="Media duration in seconds")
    orig_filename = Column(String, comment="Original filename")
    orig_url = Column(String, comment="Original URL")
    title = Column(String, comment="Title of the video")
    description = Column(String, comment="Description of the video")

    media = relationship("MediaEntity", foreign_keys=[media_id], post_update=True)
    captions = relationship(
        "CaptionEntity", back_populates="video", cascade="all, delete-orphan"
    )
    transcripts = relationship(
        "TranscriptEntity", back_populates="video", cascade="all, delete-orphan"
    )
