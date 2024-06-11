from typing import Annotated, TypedDict

import numpy as np
from pydantic import Field

from aana.api.api_generation import Endpoint
from aana.core.models.chat import Prompt
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle


class ImageGenerationEndpointOutput(TypedDict):
    """Image generation endpoint output."""

    image: Annotated[
        list, Field(description="The generated image as a array of pixels.")
    ]


class ImageGenerationEndpoint(Endpoint):
    """Image generation endpoint."""

    async def initialize(self):
        """Initialize the endpoint."""
        self.image_generation_handle = await AanaDeploymentHandle.create(
            "image_generation_deployment"
        )
        await super().initialize()

    async def run(self, prompt: Prompt) -> ImageGenerationEndpointOutput:
        """Run the image generation endpoint."""
        image_generation_output = await self.image_generation_handle.generate(
            prompt=prompt,
        )
        image = np.array(image_generation_output["image"]).tolist()
        return ImageGenerationEndpointOutput(image=image)
