import uuid
from enum import Enum

from sqlalchemy import JSON, UUID
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from aana.storage.models.base import BaseEntity, TimeStampEntity


class WebhookEventType(str, Enum):
    """Enum for webhook event types."""

    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_STARTED = "task.started"


class WebhookEntity(BaseEntity, TimeStampEntity):
    """Table for webhook items."""

    __tablename__ = "webhooks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID, primary_key=True, default=uuid.uuid4, comment="Webhook ID"
    )
    user_id: Mapped[str | None] = mapped_column(
        nullable=True, index=True, comment="The user ID associated with the webhook"
    )
    url: Mapped[str] = mapped_column(
        nullable=False, comment="The URL to which the webhook will send requests"
    )
    events: Mapped[list[str]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        comment="List of events the webhook is subscribed to. If the list is empty, the webhook is subscribed to all events.",
    )

    def __repr__(self) -> str:
        """String representation of the webhook."""
        return (
            f"<WebhookEntity(id={self.id}, user_id={self.user_id}, "
            f"webhook_url={self.url}, events={self.events}, "
            f"created_at={self.created_at}, updated_at={self.updated_at})>"
        )
