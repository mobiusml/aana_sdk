from collections.abc import AsyncGenerator
from enum import Enum
from threading import Thread
from typing import Any

import torch
import transformers
from hqq.core.quantize import BaseQuantizeConfig
from hqq.core.quantize import HQQBackend as HQQBackendKernel
from hqq.models.hf.base import AutoHQQHFModel
from hqq.utils.patching import (
    HQQLinear,
    patch_add_quant_config,
    patch_linearlayers,
    prepare_for_inference,
)
from pydantic import BaseModel, ConfigDict, Field
from ray import serve
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from aana.core.models.base import merged_options, pydantic_protected_fields
from aana.core.models.sampling import SamplingParams
from aana.core.models.types import Dtype
from aana.deployments.base_text_generation_deployment import (
    BaseTextGenerationDeployment,
    LLMOutput,
)
from aana.deployments.hf_pipeline_deployment import CustomConfig
from aana.exceptions.runtime import InferenceException, PromptTooLongException
from aana.utils.streamer import async_streamer_adapter


class HQQBackend(str, Enum):
    """HQQ Backend types.

    Possible values are "default", "torchao_int4", "bitblas" and "marlin".

    Attributes:
        DEFAULT (str): default
        TORCHAO_INT4 (str): torchao_int4
        BITBLAS (str): bitblas
        MARLIN (str): marlin
    """

    DEFAULT = "default"
    TORCHAO_INT4 = "torchao_int4"
    BITBLAS = "bitblas"
    MARLIN = "marlin"

class HQQConfig(BaseModel):
    """The configuration for the HQQ deployment.

    Attributes:
        model_id (str): The model ID on Hugging Face.
        quantize_on_fly: Whether to quantize the model or it is already pre-quantized
        backend (HQQBackend): The backend lib for quantization the model
        dtype (Dtype): The data type. Defaults to Dtype.BFLOAT16.
        quantization_config (dict): The quantization params
        model_kwargs (CustomConfig): The extra model keyword arguments. Defaults to {}.
        default_sampling_params (SamplingParams): The default sampling parameters.
            Defaults to SamplingParams(temperature=0, max_tokens=256).
        chat_template (str | None): The name of the chat template. If not provided, the chat template
            from the model will be used. Some models may not have a chat template. Defaults to None.
    """

    model_id: str
    quantize_on_fly: bool = False
    backend: HQQBackend
    dtype: Dtype = Field(default=Dtype.BFLOAT16)
    quantization_config: CustomConfig = BaseQuantizeConfig()
    model_kwargs: CustomConfig = {}

    default_sampling_params: SamplingParams = SamplingParams(
        temperature=0, max_tokens=512
    )
    chat_template: str | None = None

    model_config = ConfigDict(protected_namespaces=(*pydantic_protected_fields,))

@serve.deployment
class HQQDeployment(BaseTextGenerationDeployment):
    """Deployment to serve Hugging Face text generation models."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and tokenizer from HuggingFace and if needed quantize it on the fly.

        The configuration should conform to the HfTextGenerationConfig schema.
        """
        config_obj = HQQConfig(**config)
        self.model_id = config_obj.model_id
        self.backend = config_obj.backend
        self.quantization_config = config_obj.quantization_config
        self.dtype = config_obj.dtype
        self.model_kwargs = config_obj.model_kwargs
        self.default_sampling_params = config_obj.default_sampling_params

        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.model_kwargs.get("device_map", default_device)

        if config_obj.quantize_on_fly:
            self.model_kwargs["device_map"] = self.device
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype= self.dtype.to_torch(), **self.model_kwargs)
            AutoHQQHFModel.quantize_model(self.model, quant_config=self.quantization_config , device=self.device, compute_dtype=self.dtype.to_torch())
        else:
            self.model_kwargs["device"] = self.device
            self.model = AutoHQQHFModel.from_quantized(self.model_id, compute_dtype=self.dtype.to_torch(), **self.model_kwargs)
            patch_linearlayers(self.model, patch_add_quant_config, self.quantization_config)

        HQQLinear.set_backend(HQQBackendKernel.PYTORCH)
        prepare_for_inference(self.model, backend=self.backend)

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
