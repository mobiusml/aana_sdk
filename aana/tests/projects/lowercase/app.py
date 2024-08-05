# ruff: noqa: S101, S113
from typing import Annotated, TypedDict

from pydantic import Field
from ray import serve

from aana.api.api_generation import Endpoint
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle
from aana.deployments.base_deployment import BaseDeployment
from aana.sdk import AanaSDK


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

aana_app = AanaSDK(name="lowercase_app")

for deployment in deployments:
    aana_app.register_deployment(
        name=deployment["name"],
        instance=deployment["instance"],
    )

for endpoint in endpoints:
    aana_app.register_endpoint(
        name=endpoint["name"],
        path=endpoint["path"],
        summary=endpoint["summary"],
        endpoint_cls=endpoint["endpoint_cls"],
    )
