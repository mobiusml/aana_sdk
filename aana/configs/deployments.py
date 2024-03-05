from aana.deployments.hf_blip2_deployment import HFBlip2Config, HFBlip2Deployment
from aana.deployments.stablediffusion2_deployment import (
    StableDiffusion2Config,
    StableDiffusion2Deployment,
)
from aana.deployments.standard_concepts2_deployment import (
    StandardConceptsV2Config,
    StandardConceptsV2Deployment,
)
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
            gpu_memory_reserved=13000,
            enforce_eager=True,
            default_sampling_params=SamplingParams(
                temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
            ),
            chat_template="llama2",
        ).model_dump(),
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
        ).model_dump(),
    ),
    "whisper_deployment_medium": WhisperDeployment.options(
        num_replicas=1,
        max_concurrent_queries=1000,
        ray_actor_options={"num_gpus": 0.25},
        user_config=WhisperConfig(
            model_size=WhisperModelSize.MEDIUM,
            compute_type=WhisperComputeType.FLOAT16,
        ).model_dump(),
    ),
    "stablediffusion2_deployment": StableDiffusion2Deployment.options(
        num_replicas=1,
        max_concurrent_queries=1000,
        ray_actor_options={"num_gpus": 1},
        user_config=StableDiffusion2Config(
            model="stabilityai/stable-diffusion-2",
            dtype=Dtype.FLOAT16,
        ).dict(),
    ),
    "standard_concepts_v2_deployment": StandardConceptsV2Deployment.options(
        num_replicas=1,
        max_concurrent_queries=1000,
        ray_actor_options={"num_gpus": 1},
        user_config=StandardConceptsV2Config(
            config_path="",
            model_path="",
            keywords_path="",
            keywords_encoding="utf-8",
            image_size=[224, 224, 3],
            confidence_threshold=0.55,
            top_n=20,
        ).dict(),
    ),
}
