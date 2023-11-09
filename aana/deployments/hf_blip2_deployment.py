from typing import Any, Dict, List, TypedDict
from pydantic import BaseModel, Field, validator
from ray import serve
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from aana.deployments.base_deployment import BaseDeployment

from aana.exceptions.general import InferenceException
from aana.models.core.dtype import Dtype
from aana.models.core.image import Image
from aana.utils.batch_processor import BatchProcessor


class HFBlip2Config(BaseModel):
    """
    The configuration for the BLIP2 deployment with HuggingFace models.

    Attributes:
        model (str): the model ID on HuggingFace
        dtype (str): the data type (optional, default: "auto"), one of "auto", "float32", "float16"
        batch_size (int): the batch size (optional, default: 1)
        num_processing_threads (int): the number of processing threads (optional, default: 1)
    """

    model: str
    dtype: Dtype = Field(default=Dtype.AUTO)
    batch_size: int = Field(default=1)
    num_processing_threads: int = Field(default=1)

    @validator("dtype", pre=True, always=True)
    def validate_dtype(cls, value: Dtype) -> Dtype:
        """
        Validate the data type. For BLIP2 only "float32" and "float16" are supported.

        Args:
            value (Dtype): the data type

        Returns:
            Dtype: the validated data type

        Raises:
            ValueError: if the data type is not supported
        """
        if value not in {Dtype.AUTO, Dtype.FLOAT32, Dtype.FLOAT16}:
            raise ValueError(
                f"Invalid dtype: {value}. BLIP2 only supports 'auto', 'float32', and 'float16'."
            )
        return value


class CaptioningOutput(TypedDict):
    """
    The output of the captioning model.

    Attributes:
        captions (List[str]): the list of captions
    """

    captions: List[str]


@serve.deployment
class HFBlip2Deployment(BaseDeployment):
    """
    Deployment to serve BLIP2 models using HuggingFace.
    """

    async def apply_config(self, config: Dict[str, Any]):
        """
        Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and processor from HuggingFace.

        The configuration should conform to the HFBlip2Config schema.
        """

        config_obj = HFBlip2Config(**config)

        # Create the batch processor to split the requests into batches
        # and process them in parallel
        self.batch_size = config_obj.batch_size
        self.num_processing_threads = config_obj.num_processing_threads
        # The actual inference is done in _generate_captions()
        # We use lambda because BatchProcessor expects dict as input
        # and we use **kwargs to unpack the dict into named arguments for _generate_captions()
        self.batch_processor = BatchProcessor(
            process_batch=lambda request: self._generate_captions(**request),
            batch_size=self.batch_size,
            num_threads=self.num_processing_threads,
        )

        # Load the model and processor for BLIP2 from HuggingFace
        self.model_id = config_obj.model
        self.dtype = config_obj.dtype
        self.torch_dtype = self.dtype.to_torch()
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype
        )
        self.processor = Blip2Processor.from_pretrained(self.model_id)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    async def generate_captions(self, **kwargs) -> CaptioningOutput:
        """
        Generate captions for the given images.

        Args:
            images (List[Image]): the images

        Returns:
            CaptioningOutput: the dictionary with one key "captions"
                            and the list of captions for the images as value

        Raises:
            InferenceException: if the inference fails
        """
        # Call the batch processor to process the requests
        # The actual inference is done in _generate_captions()
        return await self.batch_processor.process(kwargs)

    def _generate_captions(self, images: List[Image]) -> CaptioningOutput:
        """
        Generate captions for the given images.

        This method is called by the batch processor.

        Args:
            images (List[Image]): the images

        Returns:
            CaptioningOutput: the dictionary with one key "captions"
                            and the list of captions for the images as value

        Raises:
            InferenceException: if the inference fails
        """
        numpy_images = []
        for image in images:
            numpy_images.append(image.get_numpy())  # loading images
        inputs = self.processor(numpy_images, return_tensors="pt").to(
            self.device, self.torch_dtype
        )

        try:
            generated_ids = self.model.generate(**inputs)
            generated_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            generated_texts = [
                generated_text.strip() for generated_text in generated_texts
            ]
            return CaptioningOutput(captions=generated_texts)
        except Exception as e:
            raise InferenceException(self.model_id) from e
