# ruff: noqa: S101

import pytest

from aana.exceptions.runtime import InvalidWebhookEventType
from aana.storage.models.webhook import WebhookEntity
from aana.storage.repository.webhook import WebhookRepository


@pytest.fixture
def webhook_entities():
    """Create webhook entities for testing."""
    # fmt: off
    webhooks = [
        WebhookEntity(user_id="user1", url="https://example1.com", events=["task.completed", "task.failed"]),
        WebhookEntity(user_id="user1", url="https://example2.com", events=["task.failed"]),
        WebhookEntity(user_id="user2", url="https://example3.com", events=["task.completed", "task.failed"]),
        WebhookEntity(user_id=None, url="https://example4.com", events=["task.failed"]),  # System webhook for task.failed
        WebhookEntity(user_id=None, url="https://example5.com", events=None),  # System webhook for all events
    ]
    return webhooks
    # fmt: on


@pytest.mark.asyncio
async def test_save_webhook(db_session_manager, webhook_entities):
    """Test saving a webhook."""
    async with db_session_manager.session() as session:
        webhook_repo = WebhookRepository(session)
        for webhook in webhook_entities:
            saved_webhook = await webhook_repo.save(webhook)
            assert saved_webhook.id is not None
            assert saved_webhook.user_id == webhook.user_id
            assert saved_webhook.url == webhook.url
            assert saved_webhook.events == webhook.events

            retrieved_webhook = await webhook_repo.read(saved_webhook.id)
            assert retrieved_webhook.id == saved_webhook.id
            assert retrieved_webhook.user_id == retrieved_webhook.user_id
            assert retrieved_webhook.url == retrieved_webhook.url

        # Test saving a webhook with invalid event type
        with pytest.raises(InvalidWebhookEventType):
            webhook = WebhookEntity(
                user_id="user1", url="https://example7.com", events=["invalid.event"]
            )
            await webhook_repo.save(webhook)


@pytest.mark.asyncio
async def test_get_webhooks(db_session_manager, webhook_entities):
    """Test fetching webhooks."""
    async with db_session_manager.session() as session:
        webhook_repo = WebhookRepository(session)

        # Save webhooks
        for webhook in webhook_entities:
            await webhook_repo.save(webhook)

        # Test webhooks with user ID set to None
        webhooks = await webhook_repo.get_webhooks(user_id=None)
        assert len(webhooks) == 2
        assert {webhook.url for webhook in webhooks} == {
            "https://example4.com",
            "https://example5.com",
        }

        # Test webhooks with user ID set to "user1"
        webhooks = await webhook_repo.get_webhooks(user_id="user1")
        assert len(webhooks) == 2
        assert {webhook.url for webhook in webhooks} == {
            "https://example1.com",
            "https://example2.com",
        }

        # Test webhooks with user ID set to "user2"
        webhooks = await webhook_repo.get_webhooks(user_id="user2")
        assert len(webhooks) == 1

        # Test webhooks with user ID set to "user3"
        webhooks = await webhook_repo.get_webhooks(user_id="user3")
        assert len(webhooks) == 0

        # Test webhooks with user ID set to None and event type set to "task.failed"
        webhooks = await webhook_repo.get_webhooks(
            user_id=None, event_type="task.failed"
        )
        assert len(webhooks) == 2
        assert {webhook.url for webhook in webhooks} == {
            "https://example4.com",
            "https://example5.com",
        }

        # Test webhooks with user ID set to "user1" and event type set to "task.failed"
        webhooks = await webhook_repo.get_webhooks(
            user_id="user1", event_type="task.failed"
        )
        assert len(webhooks) == 2
        assert {webhook.url for webhook in webhooks} == {
            "https://example1.com",
            "https://example2.com",
        }

        # Test webhooks with user ID set to "user1" and event type set to "task.completed"
        webhooks = await webhook_repo.get_webhooks(
            user_id="user1", event_type="task.completed"
        )
        assert len(webhooks) == 1
        assert {webhook.url for webhook in webhooks} == {"https://example1.com"}
