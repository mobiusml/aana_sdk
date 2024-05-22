import asyncio
import pickle
from collections.abc import AsyncGenerator
from queue import Empty
from threading import Thread
from typing import Annotated, Any

import torch
import transformers
from pydantic import BaseModel, BeforeValidator, PlainSerializer
from ray import serve
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from aana.core.models.base import merged_options
from aana.core.models.sampling import SamplingParams
from aana.deployments.base_deployment import test_cache
from aana.deployments.base_text_generation_deployment import (
    BaseTextGenerationDeployment,
    LLMOutput,
)
from aana.exceptions.runtime import InferenceException, PromptTooLongException

CustomConfig = Annotated[
    dict,
    PlainSerializer(lambda x: pickle.dumps(x).decode("latin1"), return_type=str),
    BeforeValidator(
        lambda x: x if isinstance(x, dict) else pickle.loads(x.encode("latin1"))  # noqa: S301
    ),
]


class HfTextGenerationConfig(BaseModel):
    """The configuration for the Hugging Face text generation deployment.

    Attributes:
        model_id (str): the model ID on Hugging Face
        model_kwargs (dict): the model keyword arguments
        default_sampling_params (SamplingParams): the default sampling parameters
        chat_template (str): the name of the chat template (optional, default: None)
    """

    model_id: str
    model_kwargs: CustomConfig = {}
    default_sampling_params: SamplingParams = SamplingParams(
        temperature=0, max_tokens=256
    )
    chat_template: str | None = None


async def async_streamer_adapter(streamer):
    """Adapt the TextIteratorStreamer to an async generator."""
    while True:
        try:
            for item in streamer:
                yield item
            break
        except Empty:
            # wait for the next item
            await asyncio.sleep(0.01)


@serve.deployment
class HfTextGenerationDeployment(BaseTextGenerationDeployment):
    """Deployment to serve Hugging Face text generation models."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and tokenizer from HuggingFace.

        The configuration should conform to the HfTextGenerationConfig schema.
        """
        config_obj = HfTextGenerationConfig(**config)
        self.model_id = config_obj.model_id
        self.model_kwargs = config_obj.model_kwargs
        self.default_sampling_params = config_obj.default_sampling_params

        self.model_kwargs["device_map"] = self.model_kwargs.get("device_map", "auto")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **self.model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.chat_template_name = config_obj.chat_template

    @test_cache
    async def generate_stream(
        self, prompt: str, sampling_params: SamplingParams | None = None
    ) -> AsyncGenerator[LLMOutput, None]:
        """Generate text stream.

        Args:
            prompt (str): the prompt
            sampling_params (SamplingParams | None): the sampling parameters

        Yields:
            LLMOutput: the generated text
        """
        # Set the seed to make the results reproducible
        transformers.set_seed(42)

        prompt = str(prompt)

        if sampling_params is None:
            sampling_params = SamplingParams()
        sampling_params = merged_options(self.default_sampling_params, sampling_params)

        prompt_input = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )

        if prompt_input["input_ids"].shape[-1] > self.tokenizer.model_max_length:
            raise PromptTooLongException(
                prompt_len=prompt_input["input_ids"].shape[-1],
                max_len=self.tokenizer.model_max_length,
            )

        try:
            with torch.no_grad():
                prompt_input.to(self.model.device)
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    timeout=0,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                generation_kwargs = dict(
                    prompt_input,
                    streamer=streamer,
                    max_new_tokens=sampling_params.max_tokens,
                    top_k=sampling_params.top_k,
                    top_p=sampling_params.top_p,
                    temperature=sampling_params.temperature,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
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
                prompt_input.to("cpu")
                del streamer
        except Exception as e:
            raise InferenceException(model_name=self.model_id) from e
