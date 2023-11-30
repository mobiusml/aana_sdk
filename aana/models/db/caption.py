from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from aana.configs.db import IdSqlType, id_type
from aana.models.db.base import BaseModel, TimeStampEntity


class Caption(BaseModel, TimeStampEntity):
    """ORM model for media captions."""

    __tablename__ = "captions"

    id: id_type = Column(IdSqlType, primary_key=True)  # noqa: A003
    model = Column(String, comment="Name of model used to generate the caption")
    media_id = Column(
        IdSqlType, ForeignKey("media.id"), comment="Foreign key to media table"
    )
    frame_id = Column(Integer, comment="The 0-based frame id of media for caption")
    caption = Column(String, comment="Frame caption")
    timestamp = Column(Float, comment="Frame timestamp in seconds")

    media = relationship("Media", back_populates="captions")
