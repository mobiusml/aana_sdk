# ruff: noqa: S101, S113
import json
import os
from typing import TypedDict

import pytest
import requests
from sqlalchemy.orm import Session

from aana.api.api_generation import Endpoint
from aana.core.models.api_service import ApiKeyType
from aana.storage.models.api_key import ApiKeyEntity

ACTIVE_API_KEY = "1234567890"
INACTIVE_API_KEY = "0000000000"
NON_EXISTENT_API_KEY = "1111111111"


@pytest.fixture(scope="module")
def enable_api_service():
    """Enable the API service."""
    os.environ["API_SERVICE__ENABLED"] = "True"
    os.environ["API_SERVICE__LAGO_URL"] = "http://localhost:3000"
    os.environ["API_SERVICE__LAGO_API_KEY"] = "test"
    yield
    # Cleanup after test
    os.environ.pop("API_SERVICE__ENABLED", None)
    os.environ.pop("API_SERVICE__LAGO_URL", None)
    os.environ.pop("API_SERVICE__LAGO_API_KEY", None)


@pytest.fixture(scope="module")
def add_test_api_keys():
    """Add test API keys to the database."""
    from aana.configs.settings import settings

    engine = settings.api_service_db_config.get_engine()
    with Session(engine) as session:
        # fmt: off
        session.add_all(
            [
                ApiKeyEntity(user_id="1", is_subscription_active=True, api_key=ACTIVE_API_KEY, subscription_id="sub1"),
                ApiKeyEntity(user_id="2", is_subscription_active=False, api_key=INACTIVE_API_KEY, subscription_id="sub2"),
            ]
        )
        # fmt: on
        session.commit()


class LowercaseEndpointOutput(TypedDict):
    """The output of the lowercase endpoint."""

    text: str
    api_key: ApiKeyType


class LowercaseEndpoint(Endpoint):
    """Lowercase endpoint."""

    async def run(self, text: str, api_key: ApiKeyType) -> LowercaseEndpointOutput:
        """Lowercase the text.

        Args:
            text (str): The list of text to lowercase
            api_key (ApiKey): The API key info

        Returns:
            LowercaseEndpointOutput: The lowercase texts
        """
        lowercase_output = text.lower()
        return {"text": lowercase_output, "api_key": api_key}


class UppercaseEndpointOutput(TypedDict):
    """The output of the uppercase endpoint."""

    text: str


class UppercaseEndpoint(Endpoint):
    """Uppercase endpoint (withouth the API key)."""

    async def run(self, text: str) -> UppercaseEndpointOutput:
        """Uppercase the text.

        Args:
            text (str): The list of text to uppercase

        Returns:
            UppercaseEndpointOutput: The uppercase texts
        """
        uppercase_output = text.upper()
        return {"text": uppercase_output}


deployments = []

endpoints = [
    {
        "name": "lowercase",
        "path": "/lowercase",
        "summary": "Lowercase text",
        "endpoint_cls": LowercaseEndpoint,
    },
    {
        "name": "uppercase",
        "path": "/uppercase",
        "summary": "Uppercase text",
        "endpoint_cls": UppercaseEndpoint,
    },
]


def test_app(enable_api_service, create_app, add_test_api_keys):
    """Test the Ray Serve app."""
    aana_app = create_app(deployments, endpoints)

    port = aana_app.port
    route_prefix = ""

    headers = {
        "x-api-key": ACTIVE_API_KEY,
    }

    # Check that the server is ready
    response = requests.get(
        f"http://localhost:{port}{route_prefix}/api/ready", headers=headers
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"ready": True}, response.text

    # Test lowercase endpoint (endpoint with API key)
    data = {"text": "Hello World! This is a test."}
    response = requests.post(
        f"http://localhost:{port}{route_prefix}/lowercase",
        data={"body": json.dumps(data)},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    lowercase_text = response.json().get("text")
    assert lowercase_text == "hello world! this is a test."

    api_key_in_response = response.json().get("api_key")
    assert api_key_in_response["api_key"] == ACTIVE_API_KEY, api_key_in_response

    # Test uppercase endpoint (endpoint without API key)
    data = {"text": "Hello World! This is a test."}
    response = requests.post(
        f"http://localhost:{port}{route_prefix}/uppercase",
        data={"body": json.dumps(data)},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    uppercase_text = response.json().get("text")
    assert uppercase_text == "HELLO WORLD! THIS IS A TEST."

    # Test with inactive API key
    headers = {
        "x-api-key": INACTIVE_API_KEY,
    }
    response = requests.post(
        f"http://localhost:{port}{route_prefix}/lowercase",
        data={"body": json.dumps(data)},
        headers=headers,
    )
    assert response.status_code == 400, response.text
    assert response.json().get("error") == "InactiveSubscription", response.json()

    # Test with non-existent API key
    headers = {
        "x-api-key": NON_EXISTENT_API_KEY,
    }
    response = requests.post(
        f"http://localhost:{port}{route_prefix}/lowercase",
        data={"body": json.dumps(data)},
        headers=headers,
    )
    assert response.status_code == 400, response.text
    assert response.json().get("error") == "ApiKeyNotFound", response.json()

    # Test with missing API key
    headers = {}
    response = requests.post(
        f"http://localhost:{port}{route_prefix}/lowercase",
        data={"body": json.dumps(data)},
        headers=headers,
    )
    assert response.status_code == 400, response.text
    assert response.json().get("error") == "ApiKeyNotProvided", response.json()
