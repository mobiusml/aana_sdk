from transformers import BitsAndBytesConfig

from aana.core.models.sampling import SamplingParams
from aana.core.models.types import Dtype
from aana.deployments.hf_blip2_deployment import HFBlip2Config, HFBlip2Deployment
from aana.deployments.hf_pipeline_deployment import (
    HfPipelineConfig,
    HfPipelineDeployment,
)
from aana.deployments.hf_text_generation_deployment import (
    HfTextGenerationConfig,
    HfTextGenerationDeployment,
)
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

available_deployments = {}

vllm_llama2_7b_chat_deployment = VLLMDeployment.options(
    num_replicas=1,
    max_ongoing_requests=1000,
    ray_actor_options={"num_gpus": 0.25},
    user_config=VLLMConfig(
        model="TheBloke/Llama-2-7b-Chat-AWQ",
        dtype=Dtype.AUTO,
        quantization="awq",
        gpu_memory_reserved=13000,
        enforce_eager=True,
        default_sampling_params=SamplingParams(
            temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
        ),
        chat_template="llama2",
    ).model_dump(mode="json"),
)

available_deployments["vllm_llama2_7b_chat_deployment"] = vllm_llama2_7b_chat_deployment

meta_llama3_8b_instruct_deployment = VLLMDeployment.options(
    num_replicas=1,
    max_ongoing_requests=1000,
    ray_actor_options={"num_gpus": 0.45},
    user_config=VLLMConfig(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        dtype=Dtype.AUTO,
        gpu_memory_reserved=30000,
        enforce_eager=True,
        default_sampling_params=SamplingParams(
            temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
        ),
    ).model_dump(mode="json"),
)

available_deployments[
    "meta_llama3_8b_instruct_deployment"
] = meta_llama3_8b_instruct_deployment

hf_blip2_opt_2_7b_deployment = HFBlip2Deployment.options(
    num_replicas=1,
    max_ongoing_requests=1000,
    ray_actor_options={"num_gpus": 0.25},
    user_config=HFBlip2Config(
        model="Salesforce/blip2-opt-2.7b",
        dtype=Dtype.FLOAT16,
        batch_size=2,
        num_processing_threads=2,
    ).model_dump(mode="json"),
)
available_deployments["hf_blip2_opt_2_7b_deployment"] = hf_blip2_opt_2_7b_deployment

whisper_medium_deployment = WhisperDeployment.options(
    num_replicas=1,
    max_ongoing_requests=1000,
    ray_actor_options={"num_gpus": 0.25},
    user_config=WhisperConfig(
        model_size=WhisperModelSize.MEDIUM,
        compute_type=WhisperComputeType.FLOAT16,
    ).model_dump(mode="json"),
)
available_deployments["whisper_medium_deployment"] = whisper_medium_deployment

stablediffusion2_deployment = StableDiffusion2Deployment.options(
    num_replicas=1,
    max_ongoing_requests=1000,
    ray_actor_options={"num_gpus": 1},
    user_config=StableDiffusion2Config(
        model="stabilityai/stable-diffusion-2",
        dtype=Dtype.FLOAT16,
    ).model_dump(mode="json"),
)
available_deployments["stablediffusion2_deployment"] = stablediffusion2_deployment

vad_deployment = VadDeployment.options(
    num_replicas=1,
    max_ongoing_requests=1000,
    ray_actor_options={"num_gpus": 0.05},
    user_config=VadConfig(
        model=(
            "https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/"
            "0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin"
        ),
        onset=0.5,
        sample_rate=16000,
    ).model_dump(mode="json"),
)
available_deployments["vad_deployment"] = vad_deployment

hf_blip2_opt_2_7b_pipeline_deployment = HfPipelineDeployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
    user_config=HfPipelineConfig(
        model_id="Salesforce/blip2-opt-2.7b",
        model_kwargs={
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=False, load_in_4bit=True
            ),
        },
    ).model_dump(mode="json"),
)
available_deployments[
    "hf_blip2_opt_2_7b_pipeline_deployment"
] = hf_blip2_opt_2_7b_pipeline_deployment


hf_phi3_mini_4k_instruct_text_gen_deployment = HfTextGenerationDeployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.25},
    user_config=HfTextGenerationConfig(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        model_kwargs={
            "trust_remote_code": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=False, load_in_4bit=True
            ),
        },
    ).model_dump(mode="json"),
)

available_deployments[
    "hf_phi3_mini_4k_instruct_text_gen_deployment"
] = hf_phi3_mini_4k_instruct_text_gen_deployment
