import asyncio
import hashlib
import hmac
import json
import logging

import httpx
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from aana.configs.settings import settings as aana_settings
from aana.storage.models.api_key import ApiKeyEntity
from aana.storage.models.task import TaskEntity
from aana.storage.models.webhook import WebhookEventType
from aana.storage.repository.webhook import WebhookRepository
from aana.storage.session import get_session

logger = logging.getLogger(__name__)


class WebhookRegistrationRequest(BaseModel):
    """Request to register a webhook."""

    user_id: str | None = Field(
        None, description="The user ID. If None, the webhook is a system-wide webhook."
    )
    url: str = Field(
        ..., description="The URL to which the webhook will send requests."
    )
    events: list[WebhookEventType] = Field(
        None,
        description="The events to subscribe to. If None, the webhook is subscribed to all events.",
    )

    model_config = ConfigDict(extra="forbid")


class WebhookRegistrationResponse(BaseModel):
    """Response for a webhook registration."""

    id: str
    message: str


class TaskStatusChangeWebhookPayload(BaseModel):
    """Payload for a task status change webhook."""

    task_id: str
    status: str
    result: dict | None
    num_retries: int


class WebhookBody(BaseModel):
    """Body for a task status change webhook."""

    event: WebhookEventType
    payload: TaskStatusChangeWebhookPayload


def generate_hmac_signature(body: dict, user_id: str | None) -> str:
    """Generate HMAC signature for a payload for a given user.

    Args:
        body (dict): The webhook body.
        user_id (str | None): The user ID associated with the payload.

    Returns:
        str: The generated HMAC signature.
    """
    # Use the default secret if no user ID is provided
    secret = aana_settings.webhook.hmac_secret
    # Get the user-specific secret if a user ID is provided
    if user_id:
        with get_session() as session:
            api_key_info = (
                session.query(ApiKeyEntity).filter_by(user_id=user_id).first()
            )
            if api_key_info and api_key_info.hmac_secret:
                secret = api_key_info.hmac_secret

    payload_str = json.dumps(body, separators=(",", ":"))
    return hmac.new(secret.encode(), payload_str.encode(), hashlib.sha256).hexdigest()


@retry(
    stop=stop_after_attempt(aana_settings.webhook.retry_attempts),
    wait=wait_exponential(),
    reraise=True,
)
async def send_webhook_request(url: str, body: dict, headers: dict):
    """Send a webhook request with retries.

    Args:
        url (str): The webhook URL.
        body (dict): The body of the request.
        headers (dict): The headers to include in the request.
    """
    async with httpx.AsyncClient(timeout=1.0) as client:
        response = await client.post(url, json=body, headers=headers)
        response.raise_for_status()


async def trigger_webhooks(
    event: WebhookEventType, body: WebhookBody, user_id: str | None
):
    """Trigger webhooks for an event.

    Args:
        event (WebhookEventType): The event type.
        body (WebhookBody): The body of the webhook request.
        user_id (str | None): The user ID associated with the event.
    """
    with get_session() as session:
        webhook_repo = WebhookRepository(session)
        webhooks = webhook_repo.get_webhooks(user_id, event)
        body_dict = body.model_dump()

        for webhook in webhooks:
            signature = generate_hmac_signature(body_dict, user_id)
            headers = {"X-Signature": signature}
            try:
                asyncio.create_task(  # noqa: RUF006
                    send_webhook_request(webhook.url, body_dict, headers)
                )
            except Exception:
                logger.exception(f"Failed to send webhook request to {webhook.url}.")


async def trigger_task_webhooks(event: WebhookEventType, task: TaskEntity):
    """Trigger webhooks for a task event.

    Args:
        event (WebhookEventType): The event type.
        task (TaskEntity): The task entity.
    """
    payload = TaskStatusChangeWebhookPayload(
        task_id=str(task.id),
        status=task.status,
        result=task.result,
        num_retries=task.num_retries,
    )
    body = WebhookBody(event=event, payload=payload)
    await trigger_webhooks(event, body, task.user_id)
