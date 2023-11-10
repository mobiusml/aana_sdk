from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams as VLLMSamplingParams
from vllm.utils import random_uuid
from vllm.model_executor.utils import set_random_seed

from aana.deployments.base_deployment import BaseDeployment
from aana.exceptions.general import InferenceException
from aana.models.pydantic.sampling_params import SamplingParams
from aana.utils.general import merged_options


class VLLMConfig(BaseModel):
    """
    The configuration of the vLLM deployment.

    Attributes:
        model (str): the model name
        dtype (str): the data type (optional, default: "auto")
        quantization (str): the quantization method (optional, default: None)
        gpu_memory_utilization (float): the GPU memory utilization.
        default_sampling_params (SamplingParams): the default sampling parameters.
        max_model_len (int): the maximum generated text length in tokens (optional, default: None)
    """

    model: str
    dtype: Optional[str] = Field(default="auto")
    quantization: Optional[str] = Field(default=None)
    gpu_memory_utilization: float
    default_sampling_params: SamplingParams
    max_model_len: Optional[int] = Field(default=None)


class LLMOutput(TypedDict):
    """
    The output of the LLM model.

    Attributes:
        text (str): the generated text
    """

    text: str


class LLMBatchOutput(TypedDict):
    """
    The output of the LLM model for a batch of inputs.

    Attributes:
        texts (List[str]): the list of generated texts
    """

    texts: List[str]


@serve.deployment
class VLLMDeployment(BaseDeployment):
    """
    Deployment to serve large language models using vLLM.
    """

    async def apply_config(self, config: Dict[str, Any]):
        """
        Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and creates the engine.

        The configuration should contain the following keys:
        - model: the model name
        - dtype: the data type (optional, default: "auto")
        - quantization: the quantization method (optional, default: None)
        - gpu_memory_utilization: the GPU memory utilization.
        - default_sampling_params: the default sampling parameters.
        - max_model_len: the maximum generated text length in tokens (optional, default: None)

        Args:
            config (dict): the configuration of the deployment
        """
        config_obj = VLLMConfig(**config)
        self.model = config_obj.model
        self.default_sampling_params: SamplingParams = (
            config_obj.default_sampling_params
        )
        args = AsyncEngineArgs(
            model=config_obj.model,
            dtype=config_obj.dtype,
            quantization=config_obj.quantization,
            gpu_memory_utilization=config_obj.gpu_memory_utilization,
            max_model_len=config_obj.max_model_len,
        )

        # TODO: check if the model is already loaded.
        # If it is and none of the model parameters changed, we don't need to reload the model.

        # create the engine
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def generate_stream(
        self, prompt: str, sampling_params: SamplingParams
    ) -> AsyncGenerator[LLMOutput, None]:
        """
        Generate completion for the given prompt and stream the results.

        Args:
            prompt (str): the prompt
            sampling_params (SamplingParams): the sampling parameters

        Yields:
            LLMOutput: the dictionary with the key "text" and the generated text as the value
        """
        prompt = str(prompt)
        sampling_params = merged_options(self.default_sampling_params, sampling_params)
        request_id = None
        try:
            # convert SamplingParams to VLLMSamplingParams
            sampling_params_vllm = VLLMSamplingParams(
                **sampling_params.dict(exclude_unset=True)
            )
            # start the request
            request_id = random_uuid()
            # set the random seed for reproducibility
            set_random_seed(42)
            results_generator = self.engine.generate(
                prompt, sampling_params_vllm, request_id
            )

            num_returned = 0
            async for request_output in results_generator:
                text_output = request_output.outputs[0].text[num_returned:]
                yield LLMOutput(text=text_output)
                num_returned += len(text_output)
        except GeneratorExit as e:
            # If the generator is cancelled, we need to cancel the request
            if request_id is not None:
                await self.engine.abort(request_id)
            raise e
        except Exception as e:
            raise InferenceException(model_name=self.model) from e

    async def generate(self, prompt: str, sampling_params: SamplingParams) -> LLMOutput:
        """
        Generate completion for the given prompt.

        Args:
            prompt (str): the prompt
            sampling_params (SamplingParams): the sampling parameters

        Returns:
            LLMOutput: the dictionary with the key "text" and the generated text as the value
        """
        generated_text = ""
        async for chunk in self.generate_stream(prompt, sampling_params):
            generated_text += chunk["text"]
        return LLMOutput(text=generated_text)

    async def generate_batch(
        self, prompts: List[str], sampling_params: SamplingParams
    ) -> LLMBatchOutput:
        """
        Generate completion for the batch of prompts.

        Args:
            prompts (List[str]): the prompts
            sampling_params (SamplingParams): the sampling parameters

        Returns:
            LLMBatchOutput: the dictionary with the key "texts"
                            and the list of generated texts as the value
        """
        texts = []
        for prompt in prompts:
            text = await self.generate(prompt, sampling_params)
            texts.append(text["text"])

        return LLMBatchOutput(texts=texts)
