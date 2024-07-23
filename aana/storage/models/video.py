
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from aana.core.models.media import MediaId
from aana.storage.models.media import MediaEntity


class VideoEntity(MediaEntity):
    """Base ORM class for videos."""

    __tablename__ = "video"

    id: Mapped[MediaId] = mapped_column(ForeignKey("media.id"), primary_key=True)
    path: Mapped[str] = mapped_column(comment="Path", nullable=True)
    url: Mapped[str] = mapped_column(comment="URL", nullable=True)
    title: Mapped[str] = mapped_column(comment="Title", nullable=True)
    description: Mapped[str] = mapped_column(comment="Description", nullable=True)

    __mapper_args__ = {  # noqa: RUF012
        "polymorphic_identity": "video",
    }
