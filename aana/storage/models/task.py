import uuid
from enum import Enum

from sqlalchemy import (
    JSON,
    UUID,
    Column,
    DateTime,
    Float,
    Integer,
    PickleType,
    String,
)
from sqlalchemy import Enum as SqlEnum

from aana.storage.models.base import BaseEntity, TimeStampEntity


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
    id = Column(UUID, primary_key=True, default=uuid.uuid4, comment="Task ID")
    endpoint = Column(
        String, nullable=False, comment="The endpoint to which the task is assigned"
    )
    data = Column(PickleType, nullable=False, comment="Data for the task")
    status = Column(
        SqlEnum(Status),
        nullable=False,
        default=Status.CREATED,
        comment="Status of the task",
    )
    priority = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Priority of the task (0 is the lowest)",
    )
    assigned_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the task was assigned",
    )
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the task was completed",
    )
    progress = Column(
        Float, nullable=False, default=0.0, comment="Progress of the task in percentage"
    )
    result = Column(JSON, nullable=True, comment="Result of the task in JSON format")

    def __repr__(self):
        """String representation of the task."""
        return f"<TaskEntity(id={self.id}, endpoint={self.endpoint}, status={self.status}, priority={self.priority}, progress={self.progress})>"
