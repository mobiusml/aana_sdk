from typing import Any

import torch
import transformers
from pydantic import BaseModel, Field
from ray import serve
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from typing_extensions import TypedDict

from aana.core.models.captions import Caption, CaptionsList
from aana.core.models.image import Image
from aana.core.models.types import Dtype
from aana.deployments.base_deployment import BaseDeployment, test_cache
from aana.exceptions.runtime import InferenceException
from aana.processors.batch import BatchProcessor


class HFBlip2Config(BaseModel):
    """The configuration for the BLIP2 deployment with HuggingFace models.

    Attributes:
        model (str): the model ID on HuggingFace
        dtype (str): the data type (optional, default: "auto"), one of "auto", "float32", "float16"
        batch_size (int): the batch size (optional, default: 1)
        num_processing_threads (int): the number of processing threads (optional, default: 1)
        max_new_tokens (int): The maximum numbers of tokens to generate. (optional, default: 64)
    """

    model: str
    dtype: Dtype = Field(default=Dtype.AUTO)
    batch_size: int = Field(default=1)
    num_processing_threads: int = Field(default=1)
    max_new_tokens: int = Field(default=64)


class CaptioningOutput(TypedDict):
    """The output of the captioning model.

    Attributes:
        caption (str): the caption
    """

    caption: Caption


class CaptioningBatchOutput(TypedDict):
    """The output of the captioning model.

    Attributes:
        captions (list[str]): the list of captions
    """

    captions: CaptionsList


@serve.deployment
class HFBlip2Deployment(BaseDeployment):
    """Deployment to serve BLIP2 models using HuggingFace."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and processor from HuggingFace.

        The configuration should conform to the HFBlip2Config schema.
        """
        config_obj = HFBlip2Config(**config)

        # Create the batch processor to split the requests into batches
        # and process them in parallel
        self.batch_size = config_obj.batch_size
        self.num_processing_threads = config_obj.num_processing_threads
        # The actual inference is done in _generate()
        # We use lambda because BatchProcessor expects dict as input
        # and we use **kwargs to unpack the dict into named arguments for _generate()
        self.batch_processor = BatchProcessor(
            process_batch=lambda request: self._generate(**request),
            batch_size=self.batch_size,
            num_threads=self.num_processing_threads,
        )

        # Load the model and processor for BLIP2 from HuggingFace
        self.model_id = config_obj.model
        self.dtype = config_obj.dtype
        if self.dtype == Dtype.INT8:
            load_in_8bit = True
            self.torch_dtype = Dtype.FLOAT16.to_torch()
        else:
            load_in_8bit = False
            self.torch_dtype = self.dtype.to_torch()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = config_obj.max_new_tokens
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            load_in_8bit=load_in_8bit,
            device_map=self.device,
        )
        self.model = torch.compile(self.model)
        self.model.eval()
        self.processor = Blip2Processor.from_pretrained(self.model_id)
        self.model.to(self.device)

    @test_cache
    async def generate(self, image: Image) -> CaptioningOutput:
        """Generate captions for the given image.

        Args:
            image (Image): the image

        Returns:
            CaptioningOutput: the dictionary with one key "captions"
                            and the list of captions for the image as value

        Raises:
            InferenceException: if the inference fails
        """
        captions: CaptioningBatchOutput = await self.batch_processor.process(
            {"images": [image]}
        )
        return CaptioningOutput(caption=captions["captions"][0])

    @test_cache
    async def generate_batch(self, **kwargs) -> CaptioningBatchOutput:
        """Generate captions for the given images.

        Args:
            images (List[Image]): the images
            **kwargs (dict[str, Any]): keywordarguments to pass to the
                batch processor.

        Returns:
            CaptioningBatchOutput: the dictionary with one key "captions"
                            and the list of captions for the images as value

        Raises:
            InferenceException: if the inference fails
        """
        # Call the batch processor to process the requests
        # The actual inference is done in _generate()
        return await self.batch_processor.process(kwargs)

    def _generate(self, images: list[Image]) -> CaptioningBatchOutput:
        """Generate captions for the given images.

        This method is called by the batch processor.

        Args:
            images (List[Image]): the images

        Returns:
            CaptioningBatchOutput: the dictionary with one key "captions"
                            and the list of captions for the images as value

        Raises:
            InferenceException: if the inference fails
        """
        # Set the seed to make the results reproducible
        transformers.set_seed(42)
        # Loading images
        numpy_images = [im.get_numpy() for im in images]
        inputs = self.processor(numpy_images, return_tensors="pt").to(
            self.device, self.torch_dtype
        )

        try:
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )
            generated_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            generated_texts = [
                generated_text.strip() for generated_text in generated_texts
            ]
            return CaptioningBatchOutput(captions=generated_texts)
        except Exception as e:
            raise InferenceException(self.model_id) from e
