# ruff: noqa: S101, S113
import asyncio
import json
import os
from typing import TypedDict

import pytest
import requests
from ray import serve

from aana.api.api_generation import Endpoint
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.base_deployment import BaseDeployment


@serve.deployment
class Lowercase(BaseDeployment):
    """Ray deployment that returns the lowercase version of a text."""

    async def lower(self, text: str) -> dict:
        """Lowercase the text.

        Args:
            text (str): The text to lowercase

        Returns:
            dict: The lowercase text
        """
        # kill the deployment if the text is "kill"
        if text == "kill":
            os._exit(0)
        return {"text": text.lower()}


class LowercaseEndpointOutput(TypedDict):
    """The output of the lowercase endpoint."""

    text: str


class LowercaseEndpoint(Endpoint):
    """Lowercase endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.lowercase_handle = await AanaDeploymentHandle.create(
            "lowercase_deployment"
        )
        await super().initialize()

    async def run(self, text: str) -> LowercaseEndpointOutput:
        """Lowercase the text.

        Args:
            text (TextList): The list of text to lowercase

        Returns:
            LowercaseEndpointOutput: The lowercase texts
        """
        lowercase_output = await self.lowercase_handle.lower(text=text)
        return {"text": lowercase_output["text"]}


deployments = [
    {
        "name": "lowercase_deployment",
        "instance": Lowercase,
    }
]

endpoints = [
    {
        "name": "lowercase",
        "path": "/lowercase",
        "summary": "Lowercase text",
        "endpoint_cls": LowercaseEndpoint,
    }
]


def get_status(app):
    """Get the status of the app."""
    response = requests.get(f"http://localhost:{app.port}/api/status")
    return response.json()


@pytest.mark.asyncio
async def test_status_endpoint(create_app):
    """Test SDK status endpoint."""
    aana_app = create_app(deployments, endpoints)

    status = get_status(aana_app)
    assert status["status"] == "RUNNING"

    # kill the deployment
    response = requests.post(  # noqa: ASYNC100
        f"http://localhost:{aana_app.port}/lowercase",
        data={"body": json.dumps({"text": "kill"})},
    ).json()
    assert "error" in response

    # Wait for the UNHEALTHY status for 30 seconds
    for _ in range(30):
        status = get_status(aana_app)
        if status["status"] == "UNHEALTHY":
            break
        await asyncio.sleep(1)
    assert status["status"] == "UNHEALTHY"
    assert len(status["message"]) > 0
    assert status["deployments"]["lowercase_deployment"]["status"] == "UNHEALTHY"
    assert len(status["deployments"]["lowercase_deployment"]["message"]) > 0

    # Wait for the RUNNING status for 30 seconds (deployment should restart and be healthy again)
    for _ in range(30):
        status = get_status(aana_app)
        if status["status"] == "RUNNING":
            break
        await asyncio.sleep(1)
    assert status["status"] == "RUNNING"
    assert len(status["message"]) == 0
    assert status["deployments"]["lowercase_deployment"]["status"] == "RUNNING"
    assert len(status["deployments"]["lowercase_deployment"]["message"]) == 0
