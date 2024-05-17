import pickle
from copy import copy, deepcopy
from typing import Annotated, Any

import torch
from pydantic import BaseModel, BeforeValidator, PlainSerializer
from ray import serve
from transformers import pipeline

from aana.deployments.base_deployment import BaseDeployment
from aana.models.core.image import Image
from aana.utils.test import test_cache

CustomConfig = Annotated[
    dict,
    PlainSerializer(lambda x: pickle.dumps(x).decode("latin1"), return_type=str),
    BeforeValidator(
        lambda x: x if isinstance(x, dict) else pickle.loads(x.encode("latin1"))  # noqa: S301
    ),
]


class HfPipelineConfig(BaseModel):
    """The configuration for the Hugging Face pipeline deployment.

    Attributes:
        model_id (str): the model ID on Hugging Face
        task (str): the task name (optional, by default the task is inferred from the model ID)
        model_kwargs (dict): the model keyword arguments
        pipeline_kwargs (dict): the pipeline keyword arguments
        generation_kwargs (dict): the generation keyword arguments
    """

    model_id: str
    task: str | None = None
    model_kwargs: CustomConfig = {}
    pipeline_kwargs: CustomConfig = {}
    generation_kwargs: CustomConfig = {}


@serve.deployment
class HfPipelineDeployment(BaseDeployment):
    """Deployment to serve Hugging Face pipelines."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the pipeline from HuggingFace.

        The configuration should conform to the HfPipelineConfig schema.
        """
        config_obj = HfPipelineConfig(**config)
        self.model_kwargs = config_obj.model_kwargs
        self.pipeline_kwargs = config_obj.pipeline_kwargs
        self.generation_kwargs = config_obj.generation_kwargs

        try:
            # Try to load pipeline with device_map=auto to use accelerate
            self.pipeline = pipeline(
                task=config_obj.task,
                model=config_obj.model_id,
                device_map="auto",
                model_kwargs=copy(self.model_kwargs),
                **self.pipeline_kwargs,
            )
        except ValueError as e:
            if "does not support `device_map='auto'`" in str(e):
                # If model doesn't support device_map=auto, use torch to figure out where to load the model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.pipeline = pipeline(
                    task=config_obj.task,
                    model=config_obj.model_id,
                    device=device,
                    model_kwargs=copy(self.model_kwargs),
                    **self.pipeline_kwargs,
                )
            else:
                raise e

    @test_cache
    async def call(self, *args, **kwargs):
        """Call the pipeline.

        Args:
            args: the input arguments
            kwargs: the input keyword arguments

        Returns:
            the output of the pipeline
        """

        def convert_images(obj):
            """Convert Aana Image objects to PIL images."""
            if isinstance(obj, Image):
                return obj.get_pil_image()
            elif isinstance(obj, list):
                return [convert_images(item) for item in obj]
            return obj

        # Apply the conversion to args
        args = [convert_images(arg) for arg in args]

        # Apply the conversion to kwargs
        kwargs = {k: convert_images(v) for k, v in kwargs.items()}

        # Update default generation kwargs with the provided ones
        _generation_kwargs = deepcopy(self.generation_kwargs)
        _generation_kwargs.update(kwargs)
        return self.pipeline(*args, **_generation_kwargs)
