from typing import Any, TypedDict

import PIL
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from pydantic import BaseModel, Field
from ray import serve

from aana.deployments.base_deployment import BaseDeployment
from aana.models.core.dtype import Dtype

class StableDiffusion2Deployment(BaseDeployment):
    """Stable Diffusion 2 deployment."""

    async def apply_config(self, _config: dict[str, Any]):
        """Apply configuration.

        The method is called when the deployment is created or updated.

        Normally we'd have a Config object, a TypedDict, to represent configurable parameters. In this case, hardcoded values are used and we load the model and scheduler from HuggingFace. You could also use the HuggingFace pipeline deployment class in `aana.deployments.hf_pipeline_deployment.py`.
        """

        # Load the model and processor from HuggingFace
        model_id = "stabilityai/stable-diffusion-2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=Dtype.FLOAT16.to_torch(),
            scheduler=EulerDiscreteScheduler.from_pretrained(
                model_id, subfolder="scheduler"
            ),
            device_map="auto",
        )

        # Move the model to the GPU.
        self.model.to(device)

    async def generate_single(self, prompt: str) -> StableDiffusion2Output:
        """Runs the model on a given prompt and returns the first output.

        Arguments:
            prompt (str): the prompt to the model.

        Returns:
            StableDiffusion2Output: a dictionary with one key containing the result
        """
        image = self.model(prompt).images[0]
        return {"image": image}

class StableDiffusion2Output(TypedDict):
    """Output class for StableDiffusion2Deployment."""

    image: PIL.Image.Image

stablediffusion2_deployment = StableDiffusion2Deployment.options(
    num_replicas=1,
    max_concurrent_queries=1000,
    ray_actor_options={"num_gpus": 1},
    user_config={}, # This is what gets passed to apply_config()
)

