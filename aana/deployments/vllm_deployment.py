import contextlib
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from aana.core.models.base import merged_options
from aana.deployments.base_text_generation_deployment import (
    BaseTextGenerationDeployment,
    LLMOutput,
)
from aana.utils.gpu import get_gpu_memory

with contextlib.suppress(ImportError):
    from vllm.model_executor.utils import (
        set_random_seed,  # Ignore if we don't have GPU and only run on CPU with test cache
    )
from vllm.sampling_params import SamplingParams as VLLMSamplingParams
from vllm.utils import random_uuid

from aana.core.models.sampling import SamplingParams
from aana.deployments.base_deployment import test_cache
from aana.exceptions.runtime import InferenceException, PromptTooLongException


class VLLMConfig(BaseModel):
    """The configuration of the vLLM deployment.

    Attributes:
        model (str): the model name
        dtype (str): the data type (optional, default: "auto")
        quantization (str): the quantization method (optional, default: None)
        gpu_memory_reserved (float): the GPU memory reserved for the model in mb
        default_sampling_params (SamplingParams): the default sampling parameters.
        max_model_len (int): the maximum generated text length in tokens (optional, default: None)
        chat_template (str): the name of the chat template, if not provided, the chat template from the model will be used
                             but some models may not have a chat template (optional, default: None)
        enforce_eager (bool): whether to enforce eager execution (optional, default: False)
    """

    model: str
    dtype: str | None = Field(default="auto")
    quantization: str | None = Field(default=None)
    gpu_memory_reserved: float
    default_sampling_params: SamplingParams = SamplingParams(
        temperature=0, max_tokens=256
    )
    max_model_len: int | None = Field(default=None)
    chat_template: str | None = Field(default=None)
    enforce_eager: bool | None = Field(default=False)


@serve.deployment
class VLLMDeployment(BaseTextGenerationDeployment):
    """Deployment to serve large language models using vLLM."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and creates the engine.

        The configuration should contain the following keys:
        - model: the model name
        - dtype: the data type (optional, default: "auto")
        - quantization: the quantization method (optional, default: None)
        - gpu_memory_reserved: the GPU memory reserved for the model in mb
        - default_sampling_params: the default sampling parameters.
        - max_model_len: the maximum generated text length in tokens (optional, default: None)
        - chat_template: the name of the chat template (optional, default: None)

        Args:
            config (dict): the configuration of the deployment
        """
        config_obj = VLLMConfig(**config)
        self.model = config_obj.model
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
            model=config_obj.model,
            dtype=config_obj.dtype,
            quantization=config_obj.quantization,
            enforce_eager=config_obj.enforce_eager,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=config_obj.max_model_len,
        )

        # TODO: check if the model is already loaded.
        # If it is and none of the model parameters changed, we don't need to reload the model.

        # create the engine
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = self.engine.engine.tokenizer.tokenizer
        self.model_config = await self.engine.get_model_config()

    @test_cache
    async def generate_stream(
        self, prompt: str, sampling_params: SamplingParams | None = None
    ) -> AsyncGenerator[LLMOutput, None]:
        """Generate completion for the given prompt and stream the results.

        Args:
            prompt (str): the prompt
            sampling_params (SamplingParams | None): the sampling parameters

        Yields:
            LLMOutput: the dictionary with the key "text" and the generated text as the value
        """
        prompt = str(prompt)

        if sampling_params is None:
            sampling_params = SamplingParams()
        sampling_params = merged_options(self.default_sampling_params, sampling_params)

        request_id = None

        # tokenize the prompt
        prompt_token_ids = self.tokenizer.encode(prompt)

        if len(prompt_token_ids) > self.model_config.max_model_len:
            raise PromptTooLongException(
                prompt_len=len(prompt_token_ids),
                max_len=self.model_config.max_model_len,
            )

        try:
            # convert SamplingParams to VLLMSamplingParams
            sampling_params_vllm = VLLMSamplingParams(
                **sampling_params.model_dump(exclude_unset=True)
            )
            # start the request
            request_id = random_uuid()
            # set the random seed for reproducibility
            set_random_seed(42)
            results_generator = self.engine.generate(
                prompt=None,
                sampling_params=sampling_params_vllm,
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
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
            raise InferenceException(model_name=self.model) from e
