import asyncio
import pickle
from collections.abc import AsyncGenerator
from queue import Empty
from threading import Thread
from typing import Annotated, Any, TypedDict

import torch
import transformers
from pydantic import BaseModel, BeforeValidator, PlainSerializer
from ray import serve
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from aana.deployments.base_deployment import BaseDeployment
from aana.exceptions.general import InferenceException
from aana.models.pydantic.chat_message import ChatDialog, ChatMessage
from aana.models.pydantic.sampling_params import SamplingParams
from aana.utils.chat_template import apply_chat_template
from aana.utils.general import merged_options

CustomConfig = Annotated[
    dict,
    PlainSerializer(lambda x: pickle.dumps(x).decode("latin1"), return_type=str),
    BeforeValidator(
        lambda x: x if isinstance(x, dict) else pickle.loads(x.encode("latin1"))  # noqa: S301
    ),
]


class LLMOutput(TypedDict):
    """The output of the LLM model.

    Attributes:
        text (str): the generated text
    """

    text: str


class LLMBatchOutput(TypedDict):
    """The output of the LLM model for a batch of inputs.

    Attributes:
        texts (List[str]): the list of generated texts
    """

    texts: list[str]


class ChatOutput(TypedDict):
    """The output of the chat model.

    Attributes:
        dialog (ChatDialog): the dialog with the responses from the model
    """

    message: ChatMessage


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
            await asyncio.sleep(0.001)


@serve.deployment
class HfTextGenerationDeployment(BaseDeployment):
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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map="auto", **self.model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.chat_template_name = config_obj.chat_template

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

        try:
            with torch.no_grad():
                input_ids = self.tokenizer(prompt, return_tensors="pt")
                input_ids.to(self.model.device)
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    timeout=0,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                generation_kwargs = dict(
                    input_ids,
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
                input_ids.to("cpu")
                del streamer
        except Exception as e:
            raise InferenceException(model_name=self.model_id) from e

    async def generate(
        self, prompt: str, sampling_params: SamplingParams | None = None
    ) -> LLMOutput:
        """Generate completion for the given prompt.

        Args:
            prompt (str): the prompt
            sampling_params (SamplingParams | None): the sampling parameters

        Returns:
            LLMOutput: the dictionary with the key "text" and the generated text as the value
        """
        generated_text = ""
        async for chunk in self.generate_stream(prompt, sampling_params):
            generated_text += chunk["text"]
        return LLMOutput(text=generated_text)

    async def generate_batch(
        self, prompts: list[str], sampling_params: SamplingParams
    ) -> LLMBatchOutput:
        """Generate completion for the batch of prompts.

        Args:
            prompts (List[str]): the prompts
            sampling_params (SamplingParams | None): the sampling parameters

        Returns:
            LLMBatchOutput: the dictionary with the key "texts"
                            and the list of generated texts as the value
        """
        texts = []
        for prompt in prompts:
            text = await self.generate(prompt, sampling_params)
            texts.append(text["text"])

        return LLMBatchOutput(texts=texts)

    async def chat(
        self, dialog: ChatDialog, sampling_params: SamplingParams | None = None
    ) -> ChatOutput:
        """Chat with the model.

        Args:
            dialog (ChatDialog): the dialog
            sampling_params (SamplingParams | None): the sampling parameters

        Returns:
            ChatOutput: the dictionary with the key "message"
                        and the response message with a role "assistant"
                        and the generated text as the content
        """
        prompt = apply_chat_template(self.tokenizer, dialog, self.chat_template_name)
        response = await self.generate(prompt, sampling_params)
        response_message = ChatMessage(content=response["text"], role="assistant")
        return ChatOutput(message=response_message)

    async def chat_stream(
        self, dialog: ChatDialog, sampling_params: SamplingParams | None = None
    ) -> AsyncGenerator[LLMOutput, None]:
        """Chat with the model and stream the responses.

        Args:
            dialog (ChatDialog): the dialog
            sampling_params (SamplingParams | None): the sampling parameters

        Yields:
            LLMOutput: the dictionary with the key "text" and the generated text as the value
        """
        prompt = apply_chat_template(self.tokenizer, dialog, self.chat_template_name)
        async for chunk in self.generate_stream(prompt, sampling_params):
            yield chunk
