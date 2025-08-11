import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from enum import Enum
from pathlib import Path
from typing import Any

import anyio
import torch
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from ray import serve

from aana.core.models.base import merged_options, pydantic_protected_fields
from aana.core.models.chat import ChatDialog, ChatMessage
from aana.core.models.custom_config import CustomConfig
from aana.core.models.image import Image
from aana.core.models.image_chat import ImageChatDialog
from aana.core.models.multimodal_chat import FrameVideo, MultimodalChatDialog
from aana.core.models.sampling import SamplingParams
from aana.core.models.types import Dtype
from aana.deployments.base_deployment import BaseDeployment, exception_handler
from aana.deployments.base_text_generation_deployment import (
    ChatOutput,
    LLMBatchOutput,
    LLMOutput,
)
from aana.exceptions.runtime import InferenceException, PromptTooLongException
from aana.utils.gpu import get_gpu_memory
from aana.utils.lazy_import import LazyImport

with LazyImport("Run 'pip install vllm' or 'pip install aana[vllm]'") as vllm_imports:
    from huggingface_hub import snapshot_download
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.chat_utils import (
        apply_hf_chat_template,
        apply_mistral_chat_template,
        parse_chat_messages,
        resolve_chat_template_content_format,
    )
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.model_executor.utils import set_random_seed
    from vllm.sampling_params import (
        GuidedDecodingParams,
    )
    from vllm.sampling_params import (
        SamplingParams as VLLMSamplingParams,
    )
    from vllm.transformers_utils.tokenizer import MistralTokenizer
    from vllm.utils import random_uuid
    from vllm.v1.engine.async_llm import AsyncLLM as AsyncLLMV1


logger = logging.getLogger(__name__)


class GemliteQuantizationConfig(BaseModel):
    """The configuration of the gemlite quantization.

    Attributes:
        weight_bits (int): The number of bits to use for weights. Defaults to 4.
        group_size (int | None): The group size for quantization. Defaults to 64.
        quant_mode (str): The quantization mode. Defaults to "static".
        skip_modules (list[str]): The list of modules to skip for quantization. Defaults to ["lm_head", "vision", "visual"].
    """

    weight_bits: int = Field(default=4)
    group_size: int | None = Field(default=64)
    quant_mode: str = Field(default="static")
    skip_modules: list[str] = Field(
        default_factory=lambda: ["lm_head", "vision", "visual"]
    )
    model_config = ConfigDict(
        protected_namespaces=(*pydantic_protected_fields,), extra="forbid"
    )


class GemliteMode(str, Enum):
    """The mode of the gemlite quantization.

    Attributes:
        OFF (str): The gemlite quantization is off.
        PREQUANTIZED (str): The gemlite quantization is prequantized.
        ONTHEFLY (str): The gemlite quantization is on the fly.
    """

    OFF = "off"
    PREQUANTIZED = "prequantized"
    ONTHEFLY = "onthefly"


class LoRAConfig(BaseModel):
    """The configuration of the LoRA request.

    Attributes:
        name (str): The name of the LoRA adapter.
        model_id (str): The model ID on Hugging Face Hub.
    """

    name: str
    model_id: str

    model_config = ConfigDict(
        protected_namespaces=(*pydantic_protected_fields,), extra="forbid"
    )


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
        gemlite_mode (GemliteMode): The mode of the gemlite quantization. Defaults to GemliteMode.OFF.
        gemlite_config (GemliteQuantizationConfig | None): The configuration of the gemlite quantization.
        engine_args (CustomConfig): Extra engine arguments. Defaults to {}.
        enable_non_cjk_filter (bool): Whether to filter out CJK characters from the generated text.
        mm_data_concurrency_limit (int): The limit for concurrent requests with multimodal data.
            Defaults to 100.
        loras (list[LoRAConfig]): The list of LoRA configurations. Defaults to an empty list.
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

    gemlite_mode: GemliteMode = Field(default=GemliteMode.OFF)
    gemlite_config: GemliteQuantizationConfig | None = Field(default=None)

    engine_args: CustomConfig = {}
    mm_data_concurrency_limit: int = Field(default=100, ge=1)

    enable_non_cjk_filter: bool = Field(
        default=False, description="Filter out CJK characters from the generated text."
    )
    loras: list[LoRAConfig] = Field(default_factory=list)

    model_config = ConfigDict(
        protected_namespaces=(*pydantic_protected_fields,), extra="forbid"
    )


@serve.deployment
class VLLMDeployment(BaseDeployment):
    """Deployment to serve large language models using vLLM."""

    def __init__(self):
        """Initialize the deployment."""
        super().__init__()
        self.engine = None
        self.mm_data_semaphore = None

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
        - gemlite_mode: the mode of the gemlite quantization (optional, default: "off")
        - gemlite_config: the configuration of the gemlite quantization (optional, default: None)
        - mm_data_concurrency_limit: the limit for concurrent requests with multimodal data
            (optional, default: 100)
        - enable_non_cjk_filter: whether to filter out CJK characters from the generated text (optional, default: False)

        Args:
            config (dict): the configuration of the deployment
        """
        vllm_imports.check()
        config_obj = VLLMConfig(**config)

        if config_obj.gemlite_mode != GemliteMode.OFF:
            with LazyImport(
                "Run 'pip install gemlite hqq' or 'pip install aana[gemlite]'"
            ) as gemlite_imports:
                from hqq.utils.vllm import (
                    VLLM_HQQ_BACKEND,
                    set_vllm_hqq_backend,
                    set_vllm_onthefly_hqq_quant,
                )

            gemlite_imports.check()

            # Use float16 for gemlite by default
            if config_obj.dtype == Dtype.AUTO:
                logger.warning(
                    "Using float16 for gemlite by default. Can be overridden by setting dtype."
                )
                config_obj.dtype = Dtype.FLOAT16

            # For ONTHEFLY mode, we need to set the gemlite config
            if config_obj.gemlite_mode == GemliteMode.ONTHEFLY:
                if config_obj.gemlite_config is None:
                    gemlite_config = GemliteQuantizationConfig()
                else:
                    gemlite_config = config_obj.gemlite_config

                set_vllm_onthefly_hqq_quant(
                    weight_bits=gemlite_config.weight_bits,
                    group_size=gemlite_config.group_size,
                    quant_mode=gemlite_config.quant_mode,
                    skip_modules=gemlite_config.skip_modules,
                )
            elif config_obj.gemlite_mode == GemliteMode.PREQUANTIZED:
                set_vllm_hqq_backend(backend=VLLM_HQQ_BACKEND.GEMLITE)

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

        self.loras: dict[str, LoRARequest] = {}
        for lora_id, lora_config in enumerate(config_obj.loras):
            lora_path = snapshot_download(repo_id=lora_config.model_id)
            # Check if the LoRA adapter is compatible with the base model
            adapter_config_path = Path(lora_path, "adapter_config.json")
            if not adapter_config_path.exists():
                raise ValueError(  # noqa: TRY003
                    f"LoRA adapter '{lora_config.name}' is invalid: adapter_config.json not found."
                )
            async with await anyio.open_file(str(adapter_config_path), "r") as f:
                adapter_cfg = json.loads(await f.read())
                if adapter_cfg["base_model_name_or_path"] != self.model_id:
                    raise ValueError(  # noqa: TRY003
                        f"LoRA adapter '{lora_config.name}' targets "
                        f"{adapter_cfg['model_type']} but base model is {self.model_id}."
                    )

            self.loras[lora_config.name] = LoRARequest(
                lora_name=lora_config.name,
                lora_int_id=lora_id + 1,  # LoRA IDs start from 1
                lora_path=lora_path,
            )

        args = AsyncEngineArgs(
            model=config_obj.model_id,
            dtype=config_obj.dtype,
            quantization=config_obj.quantization,
            enforce_eager=config_obj.enforce_eager,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=config_obj.max_model_len,
            enable_lora=bool(config_obj.loras),
            **config_obj.engine_args,
        )

        # TODO: check if the model is already loaded.
        # If it is and none of the model parameters changed, we don't need to reload the model.

        # create the engine
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = await self.engine.get_tokenizer()
        self.model_config = await self.engine.get_model_config()

        # Optionally filter out non-CJK characters
        self.enable_non_cjk_filter = config_obj.enable_non_cjk_filter
        if self.enable_non_cjk_filter:
            self.non_cjk_token_ids = self.get_non_cjk_token_ids()

        self.mm_data_semaphore = asyncio.Semaphore(config_obj.mm_data_concurrency_limit)

    def get_non_cjk_token_ids(self) -> list[int]:
        """Return all token IDs whose decoded tokens do NOT include CJK characters."""
        vocab: dict[str, int] = self.tokenizer.get_vocab()
        non_cjk_ids = []
        for token_id in vocab.values():
            decoded = self.tokenizer.decode([token_id])
            if all(ord(ch) < 0x3000 for ch in decoded):
                non_cjk_ids.append(token_id)
        return non_cjk_ids

    async def check_health(self):
        """Check the health of the deployment and clear torch cache to prevent memory leaks."""
        if self.engine:
            await self.engine.check_health()
            torch.cuda.empty_cache()

            if isinstance(self.engine, AsyncLLMV1):
                # vLLM v1 engine does not have proper health check,
                # so we manually check if each engine core process is alive.
                try:
                    engine_core_processes = (
                        self.engine.engine_core.resources.engine_manager.processes
                    )
                    for process in engine_core_processes:
                        if not process.is_alive():
                            raise InferenceException(
                                self.model_id,
                                f"Engine core process {process.pid} died unexpectedly.",
                            )
                except AttributeError as e:
                    logger.warning(
                        f"Unable to access engine core processes for health check: {e}"
                    )

        await super().check_health()

    def apply_chat_template(  # noqa: C901
        self, dialog: ChatDialog | ImageChatDialog | MultimodalChatDialog
    ) -> tuple[str | list[int], dict | None]:
        """Apply the chat template to the dialog.

        Args:
            dialog (ChatDialog | ImageChatDialog): the dialog (optionally with images)

        Returns:
            tuple[str | list[int], dict | None]: the prompt and the multimodal data
        """

        def replace_multimodal_types(
            messages: list[dict],
            images: list[Image] | None = None,
            videos: list[FrameVideo] | None = None,
        ) -> list[dict]:
            images = images or []
            videos = videos or []
            image_index = 0
            video_index = 0

            for message in messages:
                for item in message["content"]:
                    if item["type"] == "image":
                        item["type"] = "image_url"
                        if image_index >= len(images):
                            raise ValueError(  # noqa: TRY003
                                "Number of images does not match the number of image items in the message."
                            )
                        item["image_url"] = {
                            "url": images[image_index].get_base64_url()
                        }
                        image_index += 1
                    elif item["type"] == "video":
                        item["type"] = "video_url"
                        if video_index >= len(videos):
                            raise ValueError(  # noqa: TRY003
                                "Number of videos does not match the number of video items in the message."
                            )
                        item["video_url"] = {
                            "url": videos[video_index].get_base64_url()
                        }
                        video_index += 1

            if image_index != len(images):
                raise ValueError(  # noqa: TRY003
                    "Number of images does not match the number of image items in the message."
                )
            if video_index != len(videos):
                raise ValueError(  # noqa: TRY003
                    "Number of videos does not match the number of video items in the message."
                )
            return messages

        if isinstance(dialog, ImageChatDialog):
            messages, images = dialog.to_objects()
            messages = replace_multimodal_types(messages, images=images)
        elif isinstance(dialog, MultimodalChatDialog):
            messages, images, videos = dialog.to_objects()
            messages = replace_multimodal_types(messages, images=images, videos=videos)
        else:
            messages = dialog.model_dump()["messages"]
            images = None

        content_format = resolve_chat_template_content_format(
            chat_template=None,  # Use default chat template from tokenizer
            given_format="auto",  # Use auto as the default content format ( ChatTemplateContentFormatOption = Literal["auto", "string", "openai"])
            tokenizer=self.tokenizer,
            tools=None,
            model_config=self.model_config,
        )
        conversation, mm_data = parse_chat_messages(
            messages, self.model_config, self.tokenizer, content_format=content_format
        )

        if isinstance(self.tokenizer, MistralTokenizer):
            prompt = apply_mistral_chat_template(
                self.tokenizer,
                messages=messages,
                chat_template=None,
                tools=None,
                add_generation_prompt=True,
            )
        else:
            prompt = apply_hf_chat_template(
                tokenizer=self.tokenizer,
                conversation=conversation,
                chat_template=None,
                add_generation_prompt=True,
                tools=None,
                model_config=self.model_config,
                trust_remote_code=self.model_config.trust_remote_code,
            )
        return prompt, mm_data

    @exception_handler
    async def generate_stream(  # noqa: C901
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        lora: str | None = None,
        mm_data: dict | None = None,
    ) -> AsyncGenerator[LLMOutput, None]:
        """Generate completion for the given prompt and stream the results.

        Args:
            prompt (str | list[int]): the prompt or the tokenized prompt
            mm_data (dict | None): the multimodal data
            sampling_params (SamplingParams | None): the sampling parameters
            lora (str | None): the name of the LoRA adapter to apply

        Yields:
            LLMOutput: the dictionary with the key "text" and the generated text as the value
        """
        if isinstance(prompt, str):
            prompt_token_ids = self.tokenizer.encode(prompt)
        else:
            prompt_token_ids = prompt

        if lora is not None:
            if lora not in self.loras:
                raise ValueError(f"LoRA adapter '{lora}' is not available.")  # noqa: TRY003
            lora_request = self.loras[lora]
        else:
            lora_request = None

        if sampling_params is None:
            sampling_params = self.default_sampling_params
        else:
            sampling_params = merged_options(
                self.default_sampling_params, sampling_params
            )

        json_schema = sampling_params.json_schema
        regex_string = sampling_params.regex_string
        if json_schema is not None:
            guided_decoding_params = GuidedDecodingParams(json=json_schema)
        elif regex_string is not None:
            guided_decoding_params = GuidedDecodingParams(regex=regex_string)
        else:
            guided_decoding_params = None

        request_id = None

        if len(prompt_token_ids) > self.model_config.max_model_len:
            raise PromptTooLongException(
                prompt_len=len(prompt_token_ids),
                max_len=self.model_config.max_model_len,
            )

        semaphore_acquired = False
        if mm_data is not None:
            await self.mm_data_semaphore.acquire()
            semaphore_acquired = True

        try:
            # convert SamplingParams to VLLMSamplingParams
            sampling_params_vllm = VLLMSamplingParams(
                **sampling_params.model_dump(
                    exclude_unset=True,
                    exclude=[
                        "kwargs",
                        "json_schema",
                        "regex_string",
                    ],
                ),
                **sampling_params.kwargs,
                guided_decoding=guided_decoding_params,
                allowed_token_ids=self.non_cjk_token_ids
                if self.enable_non_cjk_filter
                else None,
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
                inputs,
                sampling_params=sampling_params_vllm,
                request_id=request_id,
                lora_request=lora_request,
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
            raise InferenceException(self.model_id, str(e)) from e
        finally:
            if semaphore_acquired:
                self.mm_data_semaphore.release()

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        lora: str | None = None,
        mm_data: dict | None = None,
    ) -> LLMOutput:
        """Generate completion for the given prompt.

        Args:
            prompt (str | list[int]): the prompt or the tokenized prompt
            mm_data (dict | None): the multimodal data
            sampling_params (SamplingParams | None): the sampling parameters
            lora (str | None): the name of the LoRA adapter to apply

        Returns:
            LLMOutput: the dictionary with the key "text" and the generated text as the value
        """
        generated_text = ""
        async for chunk in self.generate_stream(
            prompt, sampling_params=sampling_params, mm_data=mm_data, lora=lora
        ):
            generated_text += chunk["text"]
        return LLMOutput(text=generated_text)

    async def generate_batch(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | None = None,
        mm_data_list: list[dict] | None = None,
        lora: str | None = None,
    ) -> LLMBatchOutput:
        """Generate completion for the batch of prompts.

        Args:
            prompts (List[str] | List[List[int]]): the list of prompts or the tokenized prompts
            mm_data_list (List[dict] | None): the list of multimodal data
            sampling_params (SamplingParams | None): the sampling parameters
            lora (str | None): the name of the LoRA adapter to apply

        Returns:
            LLMBatchOutput: the dictionary with the key "texts"
                            and the list of generated texts as the value
        """
        texts = []
        for i, prompt in enumerate(prompts):
            if mm_data_list is not None:
                text = await self.generate(
                    prompt,
                    sampling_params=sampling_params,
                    mm_data=mm_data_list[i],
                    lora=lora,
                )
            else:
                text = await self.generate(
                    prompt, sampling_params=sampling_params, lora=lora
                )
            texts.append(text["text"])

        return LLMBatchOutput(texts=texts)

    async def chat(
        self,
        dialog: ChatDialog | ImageChatDialog,
        sampling_params: SamplingParams | None = None,
        lora: str | None = None,
    ) -> ChatOutput:
        """Chat with the model.

        Args:
            dialog (ChatDialog | ImageChatDialog): the dialog (optionally with images)
            sampling_params (SamplingParams | None): the sampling parameters
            lora (str | None): the name of the LoRA adapter to apply

        Returns:
            ChatOutput: the dictionary with the key "message"
                        and the response message with a role "assistant"
                        and the generated text as the content
        """
        prompt_ids, mm_data = self.apply_chat_template(dialog)
        response = await self.generate(
            prompt_ids, sampling_params=sampling_params, mm_data=mm_data, lora=lora
        )
        response_message = ChatMessage(content=response["text"], role="assistant")
        return ChatOutput(message=response_message)

    async def chat_stream(
        self,
        dialog: ChatDialog | ImageChatDialog,
        sampling_params: SamplingParams | None = None,
        lora: str | None = None,
    ) -> AsyncGenerator[LLMOutput, None]:
        """Chat with the model and stream the responses.

        Args:
            dialog (ChatDialog | ImageChatDialog): the dialog (optionally with images)
            sampling_params (SamplingParams | None): the sampling parameters
            lora (str | None): the name of the LoRA adapter to apply

        Yields:
            LLMOutput: the dictionary with the key "text" and the generated text as the value
        """
        prompt_ids, mm_data = self.apply_chat_template(dialog)
        async for chunk in self.generate_stream(
            prompt_ids, sampling_params=sampling_params, mm_data=mm_data, lora=lora
        ):
            yield chunk
