# ruff: noqa: S101, S113
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import TypedDict

import pytest
import pytest_asyncio
import requests

from aana.api.api_generation import Endpoint
from aana.core.models.api_service import ApiKeyType
from aana.storage.models.api_key import ApiKeyEntity, ApiServiceBase

ACTIVE_API_KEY = "1234567890"
INACTIVE_API_KEY = "0000000000"
EXPIRED_API_KEY = "9999999999"
NON_EXISTENT_API_KEY = "1111111111"
ADMIN_API_KEY = "7777777777"


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


@pytest_asyncio.fixture(scope="module")
async def add_test_api_keys():
    """Add test API keys to the database."""
    from aana.configs.settings import settings
    from aana.storage.op import DatabaseSessionManager

    session_manager = DatabaseSessionManager(settings)

    async with session_manager._api_service_engine.begin() as conn:
        await conn.run_sync(ApiServiceBase.metadata.drop_all)
        await conn.run_sync(ApiServiceBase.metadata.create_all)

    async with session_manager.session() as session:
        # fmt: off
        session.add_all(
            [
                ApiKeyEntity(user_id="1", is_subscription_active=True, api_key=ACTIVE_API_KEY, key_id=ACTIVE_API_KEY, subscription_id="sub1", expired_at=datetime.now(tz=timezone.utc) + timedelta(days=180)),
                ApiKeyEntity(user_id="2", is_subscription_active=False, api_key=INACTIVE_API_KEY, key_id=INACTIVE_API_KEY, subscription_id="sub2", expired_at=datetime.now(tz=timezone.utc) + timedelta(days=180)),
                ApiKeyEntity(user_id="3", is_subscription_active=True, api_key=EXPIRED_API_KEY, key_id=EXPIRED_API_KEY, subscription_id="sub3", expired_at=datetime.now(tz=timezone.utc) - timedelta(days=1)),
                ApiKeyEntity(user_id="4", is_subscription_active=True, api_key=ADMIN_API_KEY, key_id=ADMIN_API_KEY, is_admin=True, subscription_id="sub4", expired_at=datetime.now(tz=timezone.utc) + timedelta(days=180)),
            ]
        )
        # fmt: on
        await session.commit()

    await session_manager.close()


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
        if lowercase_output == "error":
            raise ValueError("This is a test error for the lowercase endpoint.")  # noqa: TRY003
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
        "active_subscription_required": True,
        "endpoint_cls": LowercaseEndpoint,
    },
    {
        "name": "uppercase",
        "path": "/uppercase",
        "summary": "Uppercase text",
        "endpoint_cls": UppercaseEndpoint,
    },
]


def test_app(enable_api_service, create_app_with_api_service, add_test_api_keys):
    """Test the Ray Serve app."""
    aana_app = create_app_with_api_service(deployments, endpoints)

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

    # Test lowercase endpoint with error handling
    data = {"text": "Error"}
    response = requests.post(
        f"http://localhost:{port}{route_prefix}/lowercase",
        data={"body": json.dumps(data)},
        headers=headers,
    )
    assert response.status_code == 400, response.text
    error_response = response.json()
    assert error_response.get("error") == "ValueError", error_response
    # For non-admin users, stacktrace is not included
    assert error_response.get("stacktrace") is None, error_response

    # Test lowercase endpoint with error handling when task is deferred
    data = {"text": "Error"}
    response = requests.post(
        f"http://localhost:{port}{route_prefix}/lowercase?defer=True",
        data={"body": json.dumps(data)},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    task_id = response.json().get("task_id")
    assert task_id is not None, "Task ID should not be None"

    # Check the task status with timeout of 10 seconds
    start_time = time.time()
    while time.time() - start_time < 10:
        response = requests.get(
            f"http://localhost:{port}/tasks/{task_id}", headers=headers
        )
        task_status = response.json().get("status")
        result = response.json().get("result")
        if task_status == "completed":
            break
        time.sleep(0.1)

    assert task_status == "failed", response.text
    assert result.get("error") == "ValueError", response.text
    # For non-admin users, stacktrace is not included
    assert result.get("stacktrace") is None, response.text

    # Test lowercase endpoint with admin API key
    headers = {
        "x-api-key": ADMIN_API_KEY,
    }
    response = requests.post(
        f"http://localhost:{port}{route_prefix}/lowercase",
        data={"body": json.dumps(data)},
        headers=headers,
    )
    assert response.status_code == 400, response.text
    error_response = response.json()
    assert error_response.get("error") == "ValueError", error_response
    # For admin users, stacktrace is included
    assert error_response.get("stacktrace") is not None, error_response

    # Test lowercase endpoint with admin API key and error handling when task is deferred
    data = {"text": "Error"}
    response = requests.post(
        f"http://localhost:{port}{route_prefix}/lowercase?defer=True",
        data={"body": json.dumps(data)},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    task_id = response.json().get("task_id")
    assert task_id is not None, "Task ID should not be None"

    # Check the task status with timeout of 10 seconds
    start_time = time.time()
    while time.time() - start_time < 10:
        response = requests.get(
            f"http://localhost:{port}/tasks/{task_id}", headers=headers
        )
        task_status = response.json().get("status")
        result = response.json().get("result")
        if task_status == "completed":
            break
        time.sleep(0.1)
    assert task_status == "failed", response.text
    assert result.get("error") == "ValueError", response.text
    # For admin users, stacktrace is included
    assert result.get("stacktrace") is not None, response.text

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

    # Test with expired API key
    headers = {
        "x-api-key": EXPIRED_API_KEY,
    }
    response = requests.post(
        f"http://localhost:{port}{route_prefix}/lowercase",
        data={"body": json.dumps(data)},
        headers=headers,
    )
    assert response.status_code == 400, response.text
    assert response.json().get("error") == "ApiKeyExpired", response.json()
