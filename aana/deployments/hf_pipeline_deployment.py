from copy import copy, deepcopy
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict
from ray import serve
from transformers import pipeline

from aana.core.models.base import pydantic_protected_fields
from aana.core.models.custom_config import CustomConfig
from aana.core.models.image import Image
from aana.deployments.base_deployment import BaseDeployment


class HfPipelineConfig(BaseModel):
    """The configuration for the Hugging Face Pipeline deployment.

    Attributes:
        model_id (str): The model ID on Hugging Face.
        task (str | None): The task name. If not provided, the task is inferred from the model ID. Defaults to None.
        model_kwargs (CustomConfig): The model keyword arguments. Defaults to {}.
        pipeline_kwargs (CustomConfig): The pipeline keyword arguments. Defaults to {}.
        generation_kwargs (CustomConfig): The generation keyword arguments. Defaults to {}.
    """

    model_id: str
    task: str | None = None
    model_kwargs: CustomConfig = {}
    pipeline_kwargs: CustomConfig = {}
    generation_kwargs: CustomConfig = {}

    model_config = ConfigDict(protected_namespaces=(*pydantic_protected_fields,))


@serve.deployment
class HfPipelineDeployment(BaseDeployment):
    """Deployment to serve Hugging Face pipelines."""

    def load_pipeline(self):
        """Load the pipeline from Hugging Face."""
        self.pipeline = pipeline(
            task=self.task,
            model=self.model_id,
            model_kwargs=copy(self.model_kwargs),
            **self.pipeline_kwargs,
        )

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the pipeline from HuggingFace.

        The configuration should conform to the HfPipelineConfig schema.
        """
        config_obj = HfPipelineConfig(**config)
        self.model_id = config_obj.model_id
        self.task = config_obj.task
        self.model_kwargs = config_obj.model_kwargs
        self.pipeline_kwargs = config_obj.pipeline_kwargs
        self.generation_kwargs = config_obj.generation_kwargs

        if "device" in self.pipeline_kwargs or "device_map" in self.pipeline_kwargs:
            self.load_pipeline()  # Load pipeline with the provided device or device_map
        else:
            try:
                # If no device or device_map is provided, try to load pipeline with device_map=auto to use accelerate
                self.pipeline_kwargs["device_map"] = "auto"
                self.load_pipeline()
            except ValueError as e:
                if "does not support `device_map='auto'`" in str(e):
                    # If model doesn't support device_map=auto, use torch to figure out where to load the model
                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    del self.pipeline_kwargs["device_map"]
                    self.pipeline_kwargs["device"] = device
                    self.load_pipeline()
                else:
                    raise

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
