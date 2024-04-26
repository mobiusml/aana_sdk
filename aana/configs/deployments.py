from aana.deployments.hf_blip2_deployment import HFBlip2Config, HFBlip2Deployment
from aana.deployments.stablediffusion2_deployment import (
    StableDiffusion2Config,
    StableDiffusion2Deployment,
)
from aana.deployments.vad_deployment import VadConfig, VadDeployment
from aana.deployments.vllm_deployment import VLLMConfig, VLLMDeployment
from aana.deployments.whisper_deployment import (
    WhisperComputeType,
    WhisperConfig,
    WhisperDeployment,
    WhisperModelSize,
)
from aana.models.core.dtype import Dtype
from aana.models.pydantic.sampling_params import SamplingParams

available_deployments = {}

vllm_llama2_7b_chat_deployment = VLLMDeployment.options(
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
)
available_deployments["vllm_llama2_7b_chat_deployment"] = vllm_llama2_7b_chat_deployment

hf_blip2_opt_2_7b_deployment = HFBlip2Deployment.options(
    num_replicas=1,
    max_concurrent_queries=1000,
    ray_actor_options={"num_gpus": 0.25},
    user_config=HFBlip2Config(
        model="Salesforce/blip2-opt-2.7b",
        dtype=Dtype.FLOAT16,
        batch_size=2,
        num_processing_threads=2,
    ).model_dump(),
)
available_deployments["hf_blip2_opt_2_7b_deployment"] = hf_blip2_opt_2_7b_deployment

whisper_medium_deployment = WhisperDeployment.options(
    num_replicas=1,
    max_concurrent_queries=1000,
    ray_actor_options={"num_gpus": 0.25},
    user_config=WhisperConfig(
        model_size=WhisperModelSize.MEDIUM,
        compute_type=WhisperComputeType.FLOAT16,
    ).model_dump(),
)
available_deployments["whisper_medium_deployment"] = whisper_medium_deployment

stablediffusion2_deployment = StableDiffusion2Deployment.options(
    num_replicas=1,
    max_concurrent_queries=1000,
    ray_actor_options={"num_gpus": 1},
    user_config=StableDiffusion2Config(
        model="stabilityai/stable-diffusion-2",
        dtype=Dtype.FLOAT16,
    ).model_dump(),
)
available_deployments["stablediffusion2_deployment"] = stablediffusion2_deployment

vad_deployment = VadDeployment.options(
    num_replicas=1,
    max_concurrent_queries=1000,
    ray_actor_options={"num_gpus": 0.05},
    user_config=VadConfig(
        model=(
            "https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/"
            "0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin"
        ),
        onset=0.5,
        sample_rate=16000,
    ).model_dump(),
)
available_deployments["vad_deployment"] = vad_deployment
