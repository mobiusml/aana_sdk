from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from aana.storage.models.webhook import WebhookEntity, WebhookEventType
from aana.storage.repository.base import BaseRepository


class WebhookRepository(BaseRepository[WebhookEntity]):
    """Repository for webhooks."""

    def __init__(self, session: Session):
        """Constructor."""
        super().__init__(session, WebhookEntity)

    def save(self, webhook: WebhookEntity) -> WebhookEntity:
        """Save a webhook to the database.

        Args:
            webhook (WebhookEntity): The webhook to save.

        Returns:
            WebhookEntity: The saved webhook.
        """
        if webhook.events is None:
            webhook.events = []

        # Check if events are in WebhookEventType enum
        if webhook.events:
            webhook.events = [WebhookEventType(event) for event in webhook.events]

        self.session.add(webhook)
        self.session.commit()
        return webhook

    def get_webhooks(
        self, user_id: str | None, event_type: str | None = None
    ) -> list[WebhookEntity]:
        """Get webhooks for a user.

        Args:
            user_id (str | None): The user ID. If None, get system-wide webhooks.
            event_type (str | None): Filter webhooks by event type. If None, return all webhooks.

        Returns:
            List[WebhookEntity]: The list of webhooks.
        """
        query = self.session.query(WebhookEntity).filter_by(user_id=user_id)
        if event_type:
            if self.session.bind.dialect.name == "postgresql":
                query = query.filter(
                    or_(
                        WebhookEntity.events.op("@>")([event_type]),
                        WebhookEntity.events == [],
                    )
                )
            elif self.session.bind.dialect.name == "sqlite":
                events_func = func.json_each(WebhookEntity.events).table_valued(
                    "value", joins_implicitly=True
                )
                query = query.filter(
                    or_(
                        self.session.query(events_func)
                        .filter(events_func.c.value == event_type)
                        .exists(),
                        WebhookEntity.events == "[]",
                    )
                )
            else:
                raise NotImplementedError(
                    f"Filtering by event type is not supported for {self.session.bind.dialect.name}"
                )

        return query.all()
