from typing import Any, TypedDict

import PIL
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from pydantic import BaseModel, Field
from ray import serve

from aana.core.models.chat import Prompt
from aana.core.models.types import Dtype
from aana.deployments.base_deployment import BaseDeployment, test_cache


class StableDiffusion2Output(TypedDict):
    """Output class for the StableDiffusion2 deployment."""

    image: PIL.Image.Image


class StableDiffusion2Config(BaseModel):
    """The configuration for the Stable Diffusion 2 deployment.

    Attributes:
        model (str): the model ID on HuggingFace
        dtype (str): the data type (optional, default: "auto"), one of "auto", "float32", "float16"
    """

    model: str
    dtype: Dtype = Field(default=Dtype.AUTO)


@serve.deployment
class StableDiffusion2Deployment(BaseDeployment):
    """Stable Diffusion 2 deployment."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and scheduler from HuggingFace.

        The configuration should conform to the StableDiffusion2Config schema.
        """
        config_obj = StableDiffusion2Config(**config)

        # Load the model and processor from HuggingFace
        self.model_id = config_obj.model
        self.dtype = config_obj.dtype
        if self.dtype == Dtype.INT8:
            self.torch_dtype = Dtype.FLOAT16.to_torch()
        else:
            self.torch_dtype = self.dtype.to_torch()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            scheduler=EulerDiscreteScheduler.from_pretrained(
                self.model_id, subfolder="scheduler"
            ),
            device_map="auto",
        )

        self.model.to(self.device)

    @test_cache
    async def generate(self, prompt: Prompt) -> StableDiffusion2Output:
        """Runs the model on a given prompt and returns the first output.

        Arguments:
            prompt (Prompt): the prompt to the model.

        Returns:
            StableDiffusion2Output: a dictionary with one key containing the result
        """
        image = self.model(str(prompt)).images[0]
        return {"image": image}
