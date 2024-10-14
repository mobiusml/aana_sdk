import base64
from collections.abc import AsyncGenerator
from typing import Any

from outlines.integrations.vllm import JSONLogitsProcessor, RegexLogitsProcessor
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.chat_utils import (
    apply_hf_chat_template,
    apply_mistral_chat_template,
    parse_chat_messages,
)
from vllm.inputs import TokensPrompt
from vllm.model_executor.utils import set_random_seed
from vllm.sampling_params import SamplingParams as VLLMSamplingParams
from vllm.transformers_utils.tokenizer import MistralTokenizer
from vllm.utils import random_uuid

from aana.core.models.base import merged_options, pydantic_protected_fields
from aana.core.models.chat import ChatDialog, ChatMessage
from aana.core.models.custom_config import CustomConfig
from aana.core.models.image import Image
from aana.core.models.image_chat import ImageChatDialog
from aana.core.models.sampling import SamplingParams
from aana.core.models.types import Dtype
from aana.deployments.base_deployment import BaseDeployment
from aana.deployments.base_text_generation_deployment import (
    ChatOutput,
    LLMBatchOutput,
    LLMOutput,
)
from aana.exceptions.runtime import InferenceException, PromptTooLongException
from aana.utils.gpu import get_gpu_memory


class VLLMConfig(BaseModel):
    """The configuration of the vLLM deployment.

    Attributes:
        model_id (str): The model name.
        dtype (Dtype): The data type. Defaults to Dtype.AUTO.
        quantization (str | None): The quantization method. Defaults to None.
        gpu_memory_reserved (float): The GPU memory reserved for the model in MB.
        default_sampling_params (SamplingParams): The default sampling parameters.
            Defaults to SamplingParams(temperature=0, max_tokens=256).
        max_model_len (int | None): The maximum generated text length in tokens. Defaults to None.
        chat_template (str | None): The name of the chat template. If not provided, the chat template
            from the model will be used. Some models may not have a chat template.
            Defaults to None.
        enforce_eager (bool): Whether to enforce eager execution. Defaults to False.
        engine_args (CustomConfig): Extra engine arguments. Defaults to {}.
    """

    model_id: str = Field(validation_alias=AliasChoices("model_id", "model"))
    dtype: Dtype = Field(default=Dtype.AUTO)
    quantization: str | None = Field(default=None)
    gpu_memory_reserved: float
    default_sampling_params: SamplingParams = SamplingParams(
        temperature=0, max_tokens=256
    )
    max_model_len: int | None = Field(default=None)
    chat_template: str | None = Field(default=None)
    enforce_eager: bool = Field(default=False)
    engine_args: CustomConfig = {}

    model_config = ConfigDict(protected_namespaces=(*pydantic_protected_fields,))


@serve.deployment
class VLLMDeployment(BaseDeployment):
    """Deployment to serve large language models using vLLM."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and creates the engine.

        The configuration should contain the following keys:
        - model_id: the model name
        - dtype: the data type (optional, default: "auto")
        - quantization: the quantization method (optional, default: None)
        - gpu_memory_reserved: the GPU memory reserved for the model in mb
        - default_sampling_params: the default sampling parameters.
        - max_model_len: the maximum generated text length in tokens (optional, default: None)
        - chat_template: the name of the chat template (optional, default: None)
        - enforce_eager: whether to enforce eager execution (optional, default: False)
        - engine_args: extra engine arguments (optional, default: {})

        Args:
            config (dict): the configuration of the deployment
        """
        config_obj = VLLMConfig(**config)
        self.model_id = config_obj.model_id
        total_gpu_memory_bytes = get_gpu_memory()
        total_gpu_memory_mb = total_gpu_memory_bytes / 1024**2
        self.gpu_memory_utilization = (
            config_obj.gpu_memory_reserved / total_gpu_memory_mb
        )
        self.default_sampling_params: SamplingParams = (
            config_obj.default_sampling_params
        )
        self.chat_template_name = config_obj.chat_template

        args = AsyncEngineArgs(
            model=config_obj.model_id,
            dtype=config_obj.dtype,
            quantization=config_obj.quantization,
            enforce_eager=config_obj.enforce_eager,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=config_obj.max_model_len,
            **config_obj.engine_args,
        )

        # TODO: check if the model is already loaded.
        # If it is and none of the model parameters changed, we don't need to reload the model.

        # create the engine
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = self.engine.engine.tokenizer.tokenizer
        self.model_config = await self.engine.get_model_config()

    def apply_chat_template(
        self, dialog: ChatDialog | ImageChatDialog
    ) -> tuple[str | list[int], dict | None]:
        """Apply the chat template to the dialog.

        Args:
            dialog (ChatDialog | ImageChatDialog): the dialog (optionally with images)

        Returns:
            tuple[str | list[int], dict | None]: the prompt and the multimodal data
        """

        def image_to_base64(image: Image) -> str:
            """Convert an image to a base64 string."""
            image_data = image.get_content()
            base64_encoded_image = base64.b64encode(image_data)
            base64_string = "data:image;base64," + base64_encoded_image.decode("utf-8")
            return base64_string

        def replace_image_type(messages: list[dict], images: list[Image]) -> list[dict]:
            """Replace the image type with image_url for compatibility with vLLM chat utils.

            vLLM chat utils (parse_chat_messages) only support image_url type for images.
            We need to replace the image type with image_url and provide the actual image content as base64 string.
            """
            i = 0
            for message in messages:
                for item in message["content"]:
                    if item["type"] == "image":
                        item["type"] = "image_url"
                        if i >= len(images):
                            raise ValueError(  # noqa: TRY003
                                "Number of images does not match the number of image items in the message."
                            )
                        item["image_url"] = {"url": image_to_base64(images[i])}
                        i += 1
            if i != len(images):
                raise ValueError(  # noqa: TRY003
                    "Number of images does not match the number of image items in the message."
                )
            return messages

        if isinstance(dialog, ImageChatDialog):
            messages, images = dialog.to_objects()
            messages = replace_image_type(messages, images)
        else:
            messages = dialog.model_dump()["messages"]
            images = None

        conversation, mm_data = parse_chat_messages(
            messages, self.model_config, self.tokenizer
        )

        if isinstance(self.tokenizer, MistralTokenizer):
            prompt = apply_mistral_chat_template(
                self.tokenizer,
                messages=messages,
                add_generation_prompt=True,
            )
        else:
            prompt = apply_hf_chat_template(
                self.tokenizer,
                conversation=conversation,
                chat_template=self.tokenizer.chat_template,
                add_generation_prompt=True,
                # tokenize=True
            )
        return prompt, mm_data

    async def generate_stream(  # noqa: C901
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        mm_data: dict | None = None,
    ) -> AsyncGenerator[LLMOutput, None]:
        """Generate completion for the given prompt and stream the results.

        Args:
            prompt (str | list[int]): the prompt or the tokenized prompt
            mm_data (dict | None): the multimodal data
            sampling_params (SamplingParams | None): the sampling parameters

        Yields:
            LLMOutput: the dictionary with the key "text" and the generated text as the value
        """
        if isinstance(prompt, str):
            prompt_token_ids = self.tokenizer.encode(prompt)
        else:
            prompt_token_ids = prompt

        if sampling_params is None:
            sampling_params = SamplingParams()
        sampling_params = merged_options(self.default_sampling_params, sampling_params)

        json_schema = sampling_params.json_schema
        regex_string = sampling_params.regex_string
        if json_schema is not None:
            logits_processors = [JSONLogitsProcessor(json_schema, self.engine.engine)]
        elif regex_string is not None:
            logits_processors = [RegexLogitsProcessor(regex_string, self.engine.engine)]
        else:
            logits_processors = []

        request_id = None

        if len(prompt_token_ids) > self.model_config.max_model_len:
            raise PromptTooLongException(
                prompt_len=len(prompt_token_ids),
                max_len=self.model_config.max_model_len,
            )

        try:
            # convert SamplingParams to VLLMSamplingParams
            sampling_params_vllm = VLLMSamplingParams(
                **sampling_params.model_dump(
                    exclude_unset=True,
                    exclude=["kwargs", "json_schema", "regex_string"],
                ),
                **sampling_params.kwargs,
                logits_processors=logits_processors,
            )
            # start the request
            request_id = random_uuid()
            # set the random seed for reproducibility
            set_random_seed(42)
            if mm_data is not None:
                inputs = TokensPrompt(
                    prompt_token_ids=prompt_token_ids,
                    multi_modal_data=mm_data,
                )
            else:
                inputs = TokensPrompt(prompt_token_ids=prompt_token_ids)
            results_generator = self.engine.generate(
                sampling_params=sampling_params_vllm,
                request_id=request_id,
                inputs=inputs,
            )

            num_returned = 0
            async for request_output in results_generator:
                text_output = request_output.outputs[0].text[num_returned:]
                yield LLMOutput(text=text_output)
                num_returned += len(text_output)
        except GeneratorExit:
            # If the generator is cancelled, we need to cancel the request
            if request_id is not None:
                await self.engine.abort(request_id)
            raise
        except Exception as e:
            raise InferenceException(model_name=self.model_id) from e

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        mm_data: dict | None = None,
    ) -> LLMOutput:
        """Generate completion for the given prompt.

        Args:
            prompt (str | list[int]): the prompt or the tokenized prompt
            mm_data (dict | None): the multimodal data
            sampling_params (SamplingParams | None): the sampling parameters

        Returns:
            LLMOutput: the dictionary with the key "text" and the generated text as the value
        """
        generated_text = ""
        async for chunk in self.generate_stream(
            prompt, sampling_params=sampling_params, mm_data=mm_data
        ):
            generated_text += chunk["text"]
        return LLMOutput(text=generated_text)

    async def generate_batch(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | None = None,
        mm_data_list: list[dict] | None = None,
    ) -> LLMBatchOutput:
        """Generate completion for the batch of prompts.

        Args:
            prompts (List[str] | List[List[int]]): the list of prompts or the tokenized prompts
            mm_data_list (List[dict] | None): the list of multimodal data
            sampling_params (SamplingParams | None): the sampling parameters

        Returns:
            LLMBatchOutput: the dictionary with the key "texts"
                            and the list of generated texts as the value
        """
        texts = []
        for i, prompt in enumerate(prompts):
            if mm_data_list is not None:
                text = await self.generate(
                    prompt, sampling_params=sampling_params, mm_data=mm_data_list[i]
                )
            else:
                text = await self.generate(prompt, sampling_params=sampling_params)
            texts.append(text["text"])

        return LLMBatchOutput(texts=texts)

    async def chat(
        self,
        dialog: ChatDialog | ImageChatDialog,
        sampling_params: SamplingParams | None = None,
    ) -> ChatOutput:
        """Chat with the model.

        Args:
            dialog (ChatDialog | ImageChatDialog): the dialog (optionally with images)
            sampling_params (SamplingParams | None): the sampling parameters

        Returns:
            ChatOutput: the dictionary with the key "message"
                        and the response message with a role "assistant"
                        and the generated text as the content
        """
        prompt_ids, mm_data = self.apply_chat_template(dialog)
        response = await self.generate(
            prompt_ids, sampling_params=sampling_params, mm_data=mm_data
        )
        response_message = ChatMessage(content=response["text"], role="assistant")
        return ChatOutput(message=response_message)

    async def chat_stream(
        self,
        dialog: ChatDialog | ImageChatDialog,
        sampling_params: SamplingParams | None = None,
    ) -> AsyncGenerator[LLMOutput, None]:
        """Chat with the model and stream the responses.

        Args:
            dialog (ChatDialog | ImageChatDialog): the dialog (optionally with images)
            sampling_params (SamplingParams | None): the sampling parameters

        Yields:
            LLMOutput: the dictionary with the key "text" and the generated text as the value
        """
        prompt_ids, mm_data = self.apply_chat_template(dialog)
        async for chunk in self.generate_stream(
            prompt_ids, sampling_params=sampling_params, mm_data=mm_data
        ):
            yield chunk
