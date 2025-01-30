import pytest
from fastapi.testclient import TestClient
from aana.api.app import app
from aana.storage.session import get_session
from aana.storage.models.task import TaskEntity, Status as TaskStatus
from aana.storage.repository.task import TaskRepository
from aana.storage.models.webhook import WebhookEntity
from aana.storage.repository.webhook import WebhookRepository

client = TestClient(app)


@pytest.fixture
def db_session():
    session = get_session()
    yield session
    session.close()


def test_trigger_webhook_on_task_status_change(db_session, mocker):
    # Mock the send_webhook_request method
    mock_send_webhook_request = mocker.patch(
        "aana.api.request_handler.RequestHandler.send_webhook_request"
    )

    # Create a webhook
    webhook = WebhookEntity(
        user_id="user_123",
        webhook_url="https://example.com/task-updates",
        events=["task.completed", "task.failed"],
        secret="test_secret",
    )
    webhook_repo = WebhookRepository(db_session)
    webhook_repo.save(webhook)

    # Create a task
    task = TaskEntity(
        endpoint="/some-endpoint",
        data={"key": "value"},
        status=TaskStatus.CREATED,
        user_id="user_123",
    )
    task_repo = TaskRepository(db_session)
    task_repo.save(task)

    # Trigger the webhook
    payload = {
        "event": "task.completed",
        "task_id": str(task.id),
        "status": "completed",
        "timestamp": "2025-01-29T12:00:00Z",
    }
    signature = "test_signature"
    headers = {"X-Signature": signature}

    mock_send_webhook_request.assert_called_once_with(
        "https://example.com/task-updates", payload, headers
    )
