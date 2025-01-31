# ruff: noqa: S101, S113
import hashlib
import hmac
import json
from typing import Annotated, TypedDict

import requests
from pydantic import Field

from aana.api.api_generation import Endpoint
from aana.configs.settings import settings as aana_settings

TextList = Annotated[list[str], Field(description="List of text to lowercase.")]


class LowercaseEndpointOutput(TypedDict):
    """The output of the lowercase endpoint."""

    text: list[str]


class LowercaseEndpoint(Endpoint):
    """Lowercase endpoint."""

    async def run(self, text: TextList) -> LowercaseEndpointOutput:
        """Lowercase the text.

        Args:
            text (TextList): The list of text to lowercase

        Returns:
            LowercaseEndpointOutput: The lowercase texts
        """
        return {"text": [t.lower() for t in text]}


deployments = []

endpoints = [
    {
        "name": "lowercase",
        "path": "/lowercase",
        "summary": "Lowercase text",
        "endpoint_cls": LowercaseEndpoint,
    }
]


def test_webhooks(create_app, httpserver):
    """Test webhooks."""
    aana_app = create_app(deployments, endpoints)

    port = aana_app.port

    # Check that the server is ready
    response = requests.get(f"http://localhost:{port}/api/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}

    # Setup the webhook listener
    def webhook_listener(request):
        payload = request.json

        # Validate the HMAC signature
        secret_key = aana_settings.webhook.hmac_secret
        signature = request.headers.get("X-Signature")
        payload_str = json.dumps(payload, separators=(",", ":"))
        actual_signature = hmac.new(
            secret_key.encode(), payload_str.encode(), hashlib.sha256
        ).hexdigest()
        assert signature == actual_signature

        print("Received Webhook:", json.dumps(payload, indent=2))
        assert payload["event"] == "task.completed"

    httpserver.expect_request("/webhooks").respond_with_handler(webhook_listener)

    # Register the webhook
    data = {
        "url": f"http://localhost:{httpserver.port}/webhooks",
        "events": ["task.completed"],
    }
    response = requests.post(
        f"http://localhost:{port}/webhooks",
        json=data,
    )
    assert response.status_code == 201, response.text

    # Test lowercase endpoint
    data = {"text": ["Hello World!", "This is a test."]}
    response = requests.post(
        f"http://localhost:{port}/lowercase",
        data={"body": json.dumps(data)},
    )
    assert response.status_code == 200
    lowercase_text = response.json().get("text")
    assert lowercase_text == ["hello world!", "this is a test."]

    # Defer endpoint execution
    response = requests.post(
        f"http://localhost:{port}/lowercase?defer=True",
        data={"body": json.dumps(data)},
    )
    assert response.status_code == 200

    httpserver.check_assertions()
    httpserver.check_handler_errors()
