from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from aana.exceptions.runtime import InvalidWebhookEventType
from aana.storage.models.webhook import WebhookEntity, WebhookEventType
from aana.storage.repository.base import BaseRepository


class WebhookRepository(BaseRepository[WebhookEntity]):
    """Repository for webhooks."""

    def __init__(self, session: AsyncSession):
        """Constructor."""
        super().__init__(session, WebhookEntity)

    async def read(
        self, item_id: str | UUID, check: bool = True
    ) -> WebhookEntity | None:
        """Reads a single webhook from the database.

        Args:
            item_id (str | UUID): ID of the webhook to retrieve.
            check (bool): whether to raise if the entity is not found (defaults to True).

        Returns:
            The corresponding entity from the database if found.

        Raises:
            NotFoundException if the entity is not found and `check` is True.
        """
        item_id = self._convert_to_uuid(item_id)
        if item_id is None:
            return None
        return await super().read(item_id, check)

    async def delete(
        self, item_id: str | UUID, check: bool = True
    ) -> WebhookEntity | None:
        """Delete a webhook from the database.

        Args:
            item_id (str | UUID): The ID of the webhook to delete.
            check (bool): whether to raise if the entity is not found (defaults to True).

        Returns:
            WebhookEntity: The deleted webhook.

        Raises:
            NotFoundException: The id does not correspond to a record in the database.
        """
        item_id = self._convert_to_uuid(item_id)
        if item_id is None:
            return None
        return await super().delete(item_id, check)

    async def save(self, webhook: WebhookEntity) -> WebhookEntity:
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
            try:
                webhook.events = [WebhookEventType(event) for event in webhook.events]
            except ValueError as e:
                raise InvalidWebhookEventType(event_type=e.args[0]) from e

        self.session.add(webhook)
        await self.session.commit()
        return webhook

    async def get_webhooks(
        self, user_id: str | None, event_type: str | None = None
    ) -> list[WebhookEntity]:
        """Get webhooks for a user.

        Args:
            user_id (str | None): The user ID. If None, get system-wide webhooks.
            event_type (str | None): Filter webhooks by event type. If None, return all webhooks.

        Returns:
            List[WebhookEntity]: The list of webhooks.
        """
        query = select(WebhookEntity).filter_by(user_id=user_id)
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
                        select(events_func)
                        .filter(events_func.c.value == event_type)
                        .exists(),
                        WebhookEntity.events == "[]",
                    )
                )
            else:
                raise NotImplementedError(
                    f"Filtering by event type is not supported for {self.session.bind.dialect.name}"
                )

        result = await self.session.execute(query)
        return list(result.scalars().all())
