from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import backref, relationship

from aana.storage.models.base import BaseEntity, TimeStampEntity
from aana.storage.types import MediaIdSqlType


class VideoEntity(BaseEntity, TimeStampEntity):
    """ORM class for video file (video, etc)."""

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

    captions = relationship(
        "CaptionEntity",
        backref=backref("video", passive_deletes=True),
        cascade="all, delete",
        uselist=True,
    )
    transcripts = relationship(
        "TranscriptEntity",
        backref=backref("video", passive_deletes=True),
        cascade="all, delete",
        uselist=True,
    )
