from aana.deployments.hf_blip2_deployment import HFBlip2Config, HFBlip2Deployment
from aana.deployments.vllm_deployment import VLLMConfig, VLLMDeployment
from aana.deployments.whisper_deployment import (
    WhisperComputeType,
    WhisperConfig,
    WhisperDeployment,
    WhisperModelSize,
)
from aana.models.core.dtype import Dtype
from aana.models.pydantic.sampling_params import SamplingParams

deployments = {
    "vllm_deployment_llama2_7b_chat": VLLMDeployment.options(
        num_replicas=1,
        max_concurrent_queries=1000,
        ray_actor_options={"num_gpus": 0.25},
        user_config=VLLMConfig(
            model="TheBloke/Llama-2-7b-Chat-AWQ",
            dtype="auto",
            quantization="awq",
            gpu_memory_reserved=10000,
            default_sampling_params=SamplingParams(
                temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
            ),
            chat_template="llama2",
        ).dict(),
    ),
    "hf_blip2_deployment_opt_2_7b": HFBlip2Deployment.options(
        num_replicas=1,
        max_concurrent_queries=1000,
        ray_actor_options={"num_gpus": 0.25},
        user_config=HFBlip2Config(
            model="Salesforce/blip2-opt-2.7b",
            dtype=Dtype.FLOAT16,
            batch_size=2,
            num_processing_threads=2,
        ).dict(),
    ),
    "whisper_deployment_medium": WhisperDeployment.options(
        num_replicas=1,
        max_concurrent_queries=1000,
        ray_actor_options={"num_gpus": 0.25},
        user_config=WhisperConfig(
            model_size=WhisperModelSize.MEDIUM,
            compute_type=WhisperComputeType.FLOAT16,
        ).dict(),
    ),
}
