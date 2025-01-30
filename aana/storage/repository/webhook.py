from sqlalchemy.orm import Session

from aana.storage.models.webhook import WebhookEntity
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
            query = query.filter(WebhookEntity.events.contains([event_type]))
        return query.all()
