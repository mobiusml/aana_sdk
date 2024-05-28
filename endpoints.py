from typing import Annotated, TypedDict

import numpy as np
from pydantic import Field

from aana.api.api_generation import Endpoint
from aana.deployments.aana_deployment_handle import AanaDeploymentHandle

IMAGEGEN_DEPLOYMENT_NAME = "image_generation_deployment"

class ImageGenerationEndpointOutput(TypedDict):
    """Output model for the image generation endpoint"""

    image: Annotated[
        list, Field(description="The generated image as a array of pixel values.")
    ]


class ImageGenerationEndpoint(Endpoint):
    """Endpoint for image generation."""

    async def initialize(self):
        """Initialize the endpoint.
        
        Here we load a handle to the remote Ray deployment for image generation. The handle allows us to seamlessly make (`async`) calls to functions on the Deployment class istance, even if it's running in another process, or on another machine altogether."""
        self.image_generation_handle = await AanaDeploymentHandle.create(
            IMAGEGEN_DEPLOYMENT_NAME
        )

    async def run(self, prompt: Prompt) -> ImageGenerationEndpointOutput:
        """Run the image generation endpoint.
        
        This calls our remote endpoint and formats the results.
        
        Because most of our production code works focuses on handling structured response data, we haven't added support for returning binary files. Maybe that could be your first contribution?
        """
        image_generation_output = await self.image_generation_handle.generate_single(
            prompt=prompt,
        )
        image = np.array(image_generation_output["image"]).tolist()
        return ImageGenerationEndpointOutput(image=image)
