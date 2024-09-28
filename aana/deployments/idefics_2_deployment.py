from collections.abc import AsyncGenerator
from threading import Thread
from typing import Any

import torch
import transformers
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from ray import serve
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TextIteratorStreamer,
)
from transformers.utils.import_utils import is_flash_attn_2_available

from aana.core.models.base import merged_options, pydantic_protected_fields
from aana.core.models.chat import ChatMessage
from aana.core.models.custom_config import CustomConfig
from aana.core.models.image_chat import ImageChatDialog
from aana.core.models.sampling import SamplingParams
from aana.core.models.types import Dtype
from aana.deployments.base_deployment import BaseDeployment
from aana.deployments.base_text_generation_deployment import ChatOutput, LLMOutput
from aana.exceptions.runtime import InferenceException
from aana.utils.streamer import async_streamer_adapter


class Idefics2Config(BaseModel):
    """The configuration for the Idefics 2 model.

    Attributes:
        model_id (str): The model ID on HuggingFace.
        dtype (Dtype): The data type. Defaults to Dtype.AUTO.
        enable_flash_attention_2 (bool | None): Use Flash Attention 2. If None, Flash Attention 2 wii be enabled if available. Defaults to None.
        model_kwargs (CustomConfig): The extra model keyword arguments. Defaults to {}.
        processor_kwargs (CustomConfig): The extra processor keyword arguments. Defaults to {}.
        default_sampling_params (SamplingParams): The default sampling parameters. Defaults to SamplingParams(temperature=1.0, max_tokens=256).
    """

    model_id: str = Field(validation_alias=AliasChoices("model_id", "model"))
    model_kwargs: CustomConfig = {}
    processor_kwargs: CustomConfig = {}
    enable_flash_attention_2: bool | None = Field(default=None)
    dtype: Dtype = Field(default=Dtype.AUTO)
    default_sampling_params: SamplingParams = SamplingParams(
        temperature=1.0, max_tokens=256
    )
    model_config = ConfigDict(protected_namespaces=(*pydantic_protected_fields,))


@serve.deployment
class Idefics2Deployment(BaseDeployment):
    """Deployment to serve the Idefics 2 model."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and processor from HuggingFace.

        The configuration should conform to the Idefics2Config schema.
        """
        config_obj = Idefics2Config(**config)

        self.model_id = config_obj.model_id
        self.model_kwargs = config_obj.model_kwargs
        self.dtype = config_obj.dtype
        self.default_sampling_params = config_obj.default_sampling_params
        self.torch_dtype = self.dtype.to_torch()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, **config_obj.processor_kwargs
        )
        self.model_kwargs.update(
            dict(
                torch_dtype=self.torch_dtype,
                device_map=self.device,
            )
        )

        if config_obj.enable_flash_attention_2 is None:
            config_obj.enable_flash_attention_2 = is_flash_attn_2_available()
        if config_obj.enable_flash_attention_2:
            self.model_kwargs["_attn_implementation"] = "flash_attention_2"
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id, **self.model_kwargs
        )

    async def chat_stream(
        self, dialog: ImageChatDialog, sampling_params: SamplingParams | None = None
    ) -> AsyncGenerator[LLMOutput, None]:
        """Chat with the vision model and stream the results.

        Args:
            dialog (ImageChatDialog): the dialog with images
            sampling_params (SamplingParams | None): the sampling parameters

        Yields:
            LLMOutput: the dictionary with the key "text" and the chunk of generated text as the value
        """
        # Set the seed to make the results reproducible
        transformers.set_seed(42)

        if sampling_params is None:
            sampling_params = SamplingParams()
        sampling_params = merged_options(self.default_sampling_params, sampling_params)

        messages, images = dialog.to_objects()
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            images=[img.get_pil_image() for img in images],
            text=text,
            return_tensors="pt",
        ).to(self.device)

        try:
            with torch.no_grad():
                streamer = TextIteratorStreamer(
                    self.processor.tokenizer,
                    timeout=0,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                generation_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=sampling_params.max_tokens,
                    top_k=sampling_params.top_k,
                    top_p=sampling_params.top_p,
                    temperature=sampling_params.temperature,
                    num_return_sequences=1,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    **sampling_params.kwargs,
                )
                if sampling_params.temperature == 0:
                    generation_kwargs["do_sample"] = False
                    generation_kwargs["temperature"] = None
                generation_thread = Thread(
                    target=self.model.generate, kwargs=generation_kwargs
                )
                generation_thread.start()

                async_streamer = async_streamer_adapter(streamer)
                async for new_text in async_streamer:
                    yield LLMOutput(text=new_text)

                # clean up
                inputs.to("cpu")
                del streamer
                del inputs
        except Exception as e:
            raise InferenceException(model_name=self.model_id) from e

    async def chat(
        self, dialog: ImageChatDialog, sampling_params: SamplingParams | None = None
    ) -> ChatOutput:
        """Chat with the vision model.

        Args:
            dialog (ImageChatDialog): the dialog with images
            sampling_params (SamplingParams | None): the sampling parameters

        Returns:
            ChatOutput: the chat output with the message
        """
        text = ""
        async for chunk in self.chat_stream(dialog, sampling_params):
            text += chunk["text"]

        return ChatOutput(message=ChatMessage(content=text, role="assistant"))

    async def chat_batch(
        self,
        dialogs: list[ImageChatDialog],
        sampling_params: SamplingParams | None = None,
    ) -> list[ChatOutput]:
        """Chat with the vision model in batch mode.

        Args:
            dialogs (list[ImageChatDialog]): the list of dialogs with images
            sampling_params (SamplingParams | None): the sampling parameters

        Returns:
            list[ChatOutput]: the list of chat outputs with the messages
        """
        # Set the seed to make the results reproducible
        transformers.set_seed(42)

        if sampling_params is None:
            sampling_params = SamplingParams()
        sampling_params = merged_options(self.default_sampling_params, sampling_params)

        text_batch = []
        image_batch = []
        for dialog in dialogs:
            messages, images = dialog.to_objects()
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            text_batch.append(text)
            image_batch.append([img.get_pil_image() for img in images])

        inputs = self.processor(
            images=image_batch,
            text=text_batch,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        try:
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=sampling_params.max_tokens,
                top_k=sampling_params.top_k,
                top_p=sampling_params.top_p,
                temperature=sampling_params.temperature,
                num_return_sequences=1,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            if sampling_params.temperature == 0:
                generation_kwargs["do_sample"] = False
                generation_kwargs["temperature"] = None

            generated_texts = self.model.generate(**generation_kwargs)
            generated_texts = self.processor.batch_decode(
                generated_texts, skip_special_tokens=True
            )

            chat_outputs = []
            for text in generated_texts:
                # Remove the prompt from the generated text
                # TODO: find a better way to remove the prompt, it will not work for other prompt formats
                text = text.split("Assistant:")[1].strip()
                chat_outputs.append(
                    ChatOutput(message=ChatMessage(content=text, role="assistant"))
                )
        except Exception as e:
            raise InferenceException(model_name=self.model_id) from e
        else:
            return chat_outputs
