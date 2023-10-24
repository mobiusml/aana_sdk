from typing import Any, Dict, List
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

        Args:
            config (dict): the configuration of the deployment
        """
        await super().apply_config(config)

        # parse the config
        model: str = config["model"]
        dtype: str = config.get("dtype", "auto")
        quantization: str = config.get("quantization", None)
        gpu_memory_utilization: float = config["gpu_memory_utilization"]
        self.default_sampling_params: SamplingParams = config["default_sampling_params"]
        args = AsyncEngineArgs(
            model=model,
            dtype=dtype,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        # TODO: check if the model is already loaded.
        # If it is and none of the model parameters changed, we don't need to reload the model.

        # create the engine
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def generate_stream(self, prompt: str, sampling_params: SamplingParams):
        """
        Generate completion for the given prompt and stream the results.

        Args:
            prompt (str): the prompt
            sampling_params (SamplingParams): the sampling parameters

        Yields:
            dict: the generated text
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
                yield {"text": text_output}
                num_returned += len(text_output)
        except GeneratorExit as e:
            # If the generator is cancelled, we need to cancel the request
            if request_id is not None:
                await self.engine.abort(request_id)
            raise e
        except Exception as e:
            raise InferenceException() from e

    async def generate(self, prompt: str, sampling_params: SamplingParams):
        """
        Generate completion for the given prompt.

        Args:
            prompt (str): the prompt
            sampling_params (SamplingParams): the sampling parameters

        Returns:
            dict: the generated text
        """
        generated_text = ""
        async for chunk in self.generate_stream(prompt, sampling_params):
            generated_text += chunk["text"]
        return {"text": generated_text}

    async def generate_batch(self, prompts: List[str], sampling_params: SamplingParams):
        """
        Generate completion for the batch of prompts.

        Args:
            prompts (List[str]): the prompts
            sampling_params (SamplingParams): the sampling parameters

        Returns:
            dict: the generated texts
        """
        texts = []
        for prompt in prompts:
            text = await self.generate(prompt, sampling_params)
            texts.append(text["text"])

        return {"texts": texts}
