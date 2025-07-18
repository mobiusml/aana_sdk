from collections.abc import AsyncGenerator
from threading import Thread
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict
from ray import serve

from aana.core.models.base import merged_options, pydantic_protected_fields
from aana.core.models.sampling import SamplingParams
from aana.deployments.base_deployment import exception_handler
from aana.deployments.base_text_generation_deployment import (
    BaseTextGenerationDeployment,
    LLMOutput,
)
from aana.deployments.hf_pipeline_deployment import CustomConfig
from aana.exceptions.runtime import InferenceException, PromptTooLongException
from aana.utils.lazy_import import LazyImport

with LazyImport(
    "Run 'pip install transformers' or 'pip install aana[transformers]'"
) as transformers_imports:
    import transformers
    from transformers import (
        AsyncTextIteratorStreamer,
        AutoModelForCausalLM,
        AutoTokenizer,
    )


class HfTextGenerationConfig(BaseModel):
    """The configuration for the Hugging Face text generation deployment.

    Attributes:
        model_id (str): The model ID on Hugging Face.
        model_kwargs (CustomConfig): The extra model keyword arguments. Defaults to {}.
        default_sampling_params (SamplingParams): The default sampling parameters.
            Defaults to SamplingParams(temperature=0, max_tokens=256).
        chat_template (str | None): The name of the chat template. If not provided, the chat template
            from the model will be used. Some models may not have a chat template. Defaults to None.
    """

    model_id: str
    model_kwargs: CustomConfig = {}
    default_sampling_params: SamplingParams = SamplingParams(
        temperature=0, max_tokens=256
    )
    chat_template: str | None = None

    model_config = ConfigDict(protected_namespaces=(*pydantic_protected_fields,))


class BaseHfTextGenerationDeployment(BaseTextGenerationDeployment):
    """Base class for Hugging Face text generation deployments."""

    @exception_handler
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
            sampling_params = self.default_sampling_params
        else:
            sampling_params = merged_options(
                self.default_sampling_params, sampling_params
            )

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
                streamer = AsyncTextIteratorStreamer(
                    self.tokenizer,
                    timeout=None,
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
                    repetition_penalty=sampling_params.repetition_penalty,
                    **sampling_params.kwargs,
                )
                if sampling_params.temperature == 0:
                    generation_kwargs["do_sample"] = False
                    generation_kwargs["temperature"] = None
                generation_thread = Thread(
                    target=self.model.generate, kwargs=generation_kwargs
                )
                generation_thread.start()

                async for new_text in streamer:
                    yield LLMOutput(text=new_text)

                # clean up
                prompt_input.to("cpu")
                del streamer
        except Exception as e:
            raise InferenceException(self.model_id, str(e)) from e


@serve.deployment
class HfTextGenerationDeployment(BaseHfTextGenerationDeployment):
    """Deployment to serve Hugging Face text generation models."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and tokenizer from HuggingFace.

        The configuration should conform to the HfTextGenerationConfig schema.
        """
        transformers_imports.check()
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
