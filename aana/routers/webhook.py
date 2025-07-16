import asyncio
import hashlib
import hmac
import json
import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from sqlalchemy import select
from tenacity import retry, stop_after_attempt, wait_exponential

from aana.api.security import UserIdDependency
from aana.configs.settings import settings as aana_settings
from aana.storage.models.api_key import ApiKeyEntity
from aana.storage.models.task import TaskEntity
from aana.storage.models.webhook import WebhookEntity, WebhookEventType
from aana.storage.repository.webhook import WebhookRepository
from aana.storage.session import GetDbDependency, get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["webhooks"])


# Request models


class WebhookRegistrationRequest(BaseModel):
    """Request to register a webhook."""

    url: HttpUrl = Field(
        ..., description="The URL to which the webhook will send requests."
    )
    events: list[WebhookEventType] = Field(
        None,
        description="The events to subscribe to. If None, the webhook is subscribed to all events.",
    )

    model_config = ConfigDict(extra="forbid")


class WebhookUpdateRequest(BaseModel):
    """Request to update a webhook."""

    url: HttpUrl | None = Field(None, description="New URL for the webhook.")
    events: list[WebhookEventType] | None = Field(
        None, description="New list of events to subscribe to."
    )

    model_config = ConfigDict(extra="forbid")


# Response models


class WebhookResponse(BaseModel):
    """Response for a webhook registration."""

    id: str = Field(
        ...,
        description="The webhook ID.",
        example="00000000-0000-0000-0000-000000000000",
    )
    url: HttpUrl = Field(
        ..., description="The URL to which the webhook will send requests."
    )
    events: list[WebhookEventType] = Field(
        ..., description="The events that the webhook is subscribed to."
    )

    @classmethod
    def from_entity(cls, webhook: WebhookEntity) -> "WebhookResponse":
        """Create a WebhookResponse from a WebhookEntity."""
        return WebhookResponse(
            id=str(webhook.id),
            url=str(webhook.url),
            events=webhook.events,
        )


class WebhookListResponse(BaseModel):
    """Response for a list of webhooks."""

    webhooks: list[WebhookResponse] = Field(..., description="The list of webhooks.")


# Webhook Models


class TaskStatusChangeWebhookPayload(BaseModel):
    """Payload for a task status change webhook."""

    task_id: str
    task_type: str
    status: str
    result: Any | None
    num_retries: int


class WebhookBody(BaseModel):
    """Body for a task status change webhook."""

    event: WebhookEventType
    payload: TaskStatusChangeWebhookPayload


# Webhook functions


async def generate_hmac_signature(body: dict, user_id: str | None) -> str:
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
        async with get_session() as session:
            result = await session.execute(
                select(ApiKeyEntity).where(ApiKeyEntity.user_id == user_id)
            )
            api_key_info = result.scalars().first()

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


async def send_webhook_request_with_retry(url: str, body: dict, headers: dict):
    """Send a webhook request with retries.

    Args:
        url (str): The webhook URL.
        body (dict): The body of the request.
        headers (dict): The headers to include in the request.
    """
    try:
        await send_webhook_request(url, body, headers)
    except Exception:
        logger.exception(f"Failed to send webhook request to {url}.")


async def trigger_webhooks(
    event: WebhookEventType, body: WebhookBody, user_id: str | None
):
    """Trigger webhooks for an event.

    Args:
        event (WebhookEventType): The event type.
        body (WebhookBody): The body of the webhook request.
        user_id (str | None): The user ID associated with the event.
    """
    async with get_session() as session:
        webhook_repo = WebhookRepository(session)
        webhooks = await webhook_repo.get_webhooks(user_id, event)
        body_dict = body.model_dump()

        for webhook in webhooks:
            signature = await generate_hmac_signature(body_dict, user_id)
            headers = {"X-Signature": signature}
            asyncio.create_task(  # noqa: RUF006
                send_webhook_request_with_retry(webhook.url, body_dict, headers)
            )


async def trigger_task_webhooks(event: WebhookEventType, task: TaskEntity):
    """Trigger webhooks for a task event.

    Args:
        event (WebhookEventType): The event type.
        task (TaskEntity): The task entity.
    """
    payload = TaskStatusChangeWebhookPayload(
        task_id=str(task.id),
        task_type=task.endpoint,
        status=task.status,
        result=task.result,
        num_retries=task.num_retries,
    )
    body = WebhookBody(event=event, payload=payload)
    await trigger_webhooks(event, body, task.user_id)


# Webhook endpoints


@router.post("/webhooks", status_code=201)
async def create_webhook(
    request: WebhookRegistrationRequest,
    db: GetDbDependency,
    user_id: UserIdDependency,
) -> WebhookResponse:
    """This endpoint is used to register a webhook."""
    webhook_repo = WebhookRepository(db)
    webhook = WebhookEntity(
        user_id=user_id,
        url=str(request.url),
        events=request.events,
    )
    webhook = await webhook_repo.save(webhook)
    return WebhookResponse.from_entity(webhook)


@router.get("/webhooks")
async def list_webhooks(
    db: GetDbDependency, user_id: UserIdDependency
) -> WebhookListResponse:
    """This endpoint is used to list all registered webhooks."""
    webhook_repo = WebhookRepository(db)
    webhooks = await webhook_repo.get_webhooks(user_id, None)
    return WebhookListResponse(
        webhooks=[WebhookResponse.from_entity(webhook) for webhook in webhooks]
    )


@router.get("/webhooks/{webhook_id}")
async def get_webhook(
    webhook_id: str, db: GetDbDependency, user_id: UserIdDependency
) -> WebhookResponse:
    """This endpoint is used to fetch a webhook by ID."""
    webhook_repo = WebhookRepository(db)
    webhook = await webhook_repo.read(webhook_id, check=False)
    if not webhook or webhook.user_id != user_id:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return WebhookResponse.from_entity(webhook)


@router.put("/webhooks/{webhook_id}")
async def update_webhook(
    webhook_id: str,
    request: WebhookUpdateRequest,
    db: GetDbDependency,
    user_id: UserIdDependency,
) -> WebhookResponse:
    """This endpoint is used to update a webhook."""
    webhook_repo = WebhookRepository(db)
    webhook = await webhook_repo.read(webhook_id, check=False)
    if not webhook or webhook.user_id != user_id:
        raise HTTPException(status_code=404, detail="Webhook not found")
    if request.url is not None:
        webhook.url = str(request.url)
    if request.events is not None:
        webhook.events = request.events
    webhook = await webhook_repo.save(webhook)
    return WebhookResponse.from_entity(webhook)


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: str, db: GetDbDependency, user_id: UserIdDependency
) -> WebhookResponse:
    """This endpoint is used to delete a webhook."""
    webhook_repo = WebhookRepository(db)
    webhook = await webhook_repo.read(webhook_id, check=False)
    if not webhook or webhook.user_id != user_id:
        raise HTTPException(status_code=404, detail="Webhook not found")
    webhook = await webhook_repo.delete(webhook.id)
    return WebhookResponse.from_entity(webhook)
