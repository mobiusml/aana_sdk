import uuid
from enum import Enum

from sqlalchemy import (
    JSON,
    UUID,
    PickleType,
)
from sqlalchemy.orm import Mapped, mapped_column

from aana.storage.models.base import BaseEntity, TimeStampEntity, timestamp


class Status(str, Enum):
    """Enum for task status."""

    CREATED = "created"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    RUNNING = "running"
    FAILED = "failed"
    NOT_FINISHED = "not_finished"


class TaskEntity(BaseEntity, TimeStampEntity):
    """Table for task items."""

    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID, primary_key=True, default=uuid.uuid4, comment="Task ID"
    )
    endpoint: Mapped[str] = mapped_column(
        nullable=False, comment="The endpoint to which the task is assigned"
    )
    data = mapped_column(PickleType, nullable=False, comment="Data for the task")
    status: Mapped[Status] = mapped_column(
        default=Status.CREATED,
        comment="Status of the task",
    )
    priority: Mapped[int] = mapped_column(
        nullable=False, default=0, comment="Priority of the task (0 is the lowest)"
    )
    assigned_at: Mapped[timestamp | None] = mapped_column(
        comment="Timestamp when the task was assigned",
    )
    completed_at: Mapped[timestamp | None] = mapped_column(
        server_default=None,
        comment="Timestamp when the task was completed",
    )
    progress: Mapped[float] = mapped_column(
        nullable=False, default=0.0, comment="Progress of the task in percentage"
    )
    result: Mapped[dict | None] = mapped_column(
        JSON, comment="Result of the task in JSON format"
    )
    num_retries: Mapped[int] = mapped_column(
        nullable=False, default=0, comment="Number of retries"
    )

    def __repr__(self) -> str:
        """String representation of the task."""
        return (
            f"<TaskEntity(id={self.id}, "
            f"endpoint={self.endpoint}, status={self.status}, "
            f"priority={self.priority}, progress={self.progress}, "
            f"num_retries={self.num_retries}, "
            f"updated_at={self.updated_at})>"
        )
