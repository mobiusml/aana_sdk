from typing import Any

import torch
from pydantic import BaseModel
from ray import serve
from transformers import pipeline

from aana.deployments.base_deployment import BaseDeployment


class HfPipelineConfig(BaseModel):
    """The configuration for the Hugging Face pipeline deployment.

    Attributes:
        model_id (str): the model ID on Hugging Face
        task (str): the task name (optional, by default the task is inferred from the model ID)
        framework (str): the framework (optional, default: "pt"), one of "pt", "tf"
    """

    model_id: str
    task: str | None = None
    framework: str = "pt"


@serve.deployment
class HfPipelineDeployment(BaseDeployment):
    """Deployment to serve Hugging Face pipelines."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and processor from HuggingFace.

        The configuration should conform to the HFBlip2Config schema.
        """
        config_obj = HfPipelineConfig(**config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = pipeline(
            task=config_obj.task,
            model=config_obj.model_id,
            framework=config_obj.framework,
            device=device,
        )

    async def call(self, *args, **kwargs):
        """Call the pipeline.

        Args:
            args: the input arguments
            kwargs: the input keyword arguments

        Returns:
            the output of the pipeline
        """
        return self.pipeline(*args, **kwargs)
