from enum import Enum

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import backref, relationship

from aana.configs.db import MediaIdSqlType
from aana.models.db.base import BaseEntity, TimeStampEntity


class MediaType(str, Enum):
    """Enum for types of media file."""

    VIDEO = "video"


class MediaEntity(BaseEntity, TimeStampEntity):
    """Table for media items."""

    __tablename__ = "media"
    id = Column(MediaIdSqlType, primary_key=True)
    media_type = Column(String, comment="The type of media")
    video_id = Column(
        Integer,
        ForeignKey("video.id", ondelete="CASCADE"),
        nullable=True,
        comment="If media_type is `video`, the id of the video this entry represents.",
    )

    video = relationship(
        "VideoEntity",
        backref=backref("video", passive_deletes=True),
        cascade="all, delete",
        uselist=False,
        post_update=True,
        foreign_keys=[video_id],
    )
