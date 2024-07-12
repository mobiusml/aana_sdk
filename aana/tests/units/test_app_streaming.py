# ruff: noqa: S101, S113
import json
from collections.abc import AsyncGenerator
from typing import TypedDict

import requests
from ray import serve

from aana.api.api_generation import Endpoint
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.base_deployment import BaseDeployment


@serve.deployment
class Lowercase(BaseDeployment):
    """Ray Serve deployment that returns the lowercase version of a text."""

    async def lower_stream(self, text: str) -> AsyncGenerator[dict, None]:
        """Lowercase the text and yield the lowercase text in chunks of 4 characters.

        Args:
            text (str): The text to lowercase

        Yields:
            dict: The lowercase text in chunks of 4 characters
        """
        chunk_size = 4
        for i in range(0, len(text), chunk_size):
            t = text[i : i + chunk_size]
            yield {"text": t.lower()}


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

    async def run(self, text: str) -> AsyncGenerator[LowercaseEndpointOutput, None]:
        """Lowercase the text.

        Args:
            text (TextList): The list of text to lowercase

        Returns:
            LowercaseEndpointOutput: The lowercase texts
        """
        async for chunk in self.lowercase_handle.lower_stream(text=text):
            yield {"text": chunk["text"]}


deployments = [
    {
        "name": "lowercase_deployment",
        "instance": Lowercase,
    },
]

endpoints = [
    {
        "name": "lowercase",
        "path": "/lowercase",
        "summary": "Lowercase text",
        "endpoint_cls": LowercaseEndpoint,
    }
]


def test_app_streaming(app_setup):
    """Test the Ray Serve app with streaming enabled."""
    aana_app = app_setup(deployments=deployments, endpoints=endpoints)

    port = aana_app.port
    route_prefix = ""

    # Check that the server is ready
    response = requests.get(f"http://localhost:{port}{route_prefix}/api/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}

    # Test lowercase endpoint
    text = "Hello World, this is a test."
    data = {"text": text}
    response = requests.post(
        f"http://localhost:{port}{route_prefix}/lowercase",
        data={"body": json.dumps(data)},
        stream=True,
    )
    assert response.status_code == 200

    lowercase_text = ""
    offset = 0
    for chunk in response.iter_content(chunk_size=None):
        json_data = json.loads(chunk)
        lowercase_text_chunk = json_data["text"]
        lowercase_text += lowercase_text_chunk

        chunk_size = len(lowercase_text_chunk)
        expected_chunk = text[offset : offset + chunk_size]

        assert lowercase_text_chunk == expected_chunk.lower()

        offset += chunk_size

    assert lowercase_text == text.lower()
