from enum import Enum

from sqlalchemy import Column, Float, ForeignKey, String
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import relationship

from aana.storage.models.base import BaseEntity, TimeStampEntity
from aana.storage.types import MediaIdSqlType


class Status(str, Enum):
    """Enum for video status."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoEntity(BaseEntity, TimeStampEntity):
    """ORM class for video file (video, etc)."""

    __tablename__ = "video"

    id = Column(MediaIdSqlType, ForeignKey("media.id"), primary_key=True)
    duration = Column(Float, comment="Media duration in seconds")
    orig_filename = Column(String, comment="Original filename")
    orig_url = Column(String, comment="Original URL")
    title = Column(String, comment="Title of the video")
    description = Column(String, comment="Description of the video")
    status = Column(
        SqlEnum(Status),
        nullable=False,
        default=Status.CREATED,
        comment="Status of the video",
    )
    media = relationship("MediaEntity", back_populates="video", uselist=False)

    captions = relationship(
        "CaptionEntity",
        back_populates="video",
        cascade="all, delete",
        uselist=True,
    )
    transcripts = relationship(
        "TranscriptEntity",
        back_populates="video",
        cascade="all, delete",
        uselist=True,
    )
