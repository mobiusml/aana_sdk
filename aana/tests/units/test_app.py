# ruff: noqa: S101, S113
import json
from typing import Annotated, TypedDict

import requests
from pydantic import Field
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
        return {"text": [t.lower() for t in text]}


TextList = Annotated[list[str], Field(description="List of text to lowercase.")]


class LowercaseEndpointOutput(TypedDict):
    """The output of the lowercase endpoint."""

    text: list[str]


class LowercaseEndpoint(Endpoint):
    """Lowercase endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.lowercase_handle = await AanaDeploymentHandle.create(
            "lowercase_deployment"
        )
        await super().initialize()

    async def run(self, text: TextList) -> LowercaseEndpointOutput:
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


def test_app(app_setup):
    """Test the Ray Serve app."""
    aana_app = app_setup(deployments, endpoints)

    port = aana_app.port
    route_prefix = ""

    # Check that the server is ready
    response = requests.get(f"http://localhost:{port}{route_prefix}/api/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}

    # Test lowercase endpoint
    data = {"text": ["Hello World!", "This is a test."]}
    response = requests.post(
        f"http://localhost:{port}{route_prefix}/lowercase",
        data={"body": json.dumps(data)},
    )
    assert response.status_code == 200
    lowercase_text = response.json().get("text")
    print(lowercase_text)
    assert lowercase_text == ["hello world!", "this is a test."]
