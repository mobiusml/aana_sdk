from aana.deployments.vllm_deployment import VLLMConfig, VLLMDeployment
from aana.models.pydantic.sampling_params import SamplingParams


deployments = {
    "vllm_deployment_llama2_7b_chat": VLLMDeployment.options(
        num_replicas=1,
        max_concurrent_queries=1000,
        ray_actor_options={"num_gpus": 0.5},
        user_config=VLLMConfig(
            model="TheBloke/Llama-2-7b-Chat-AWQ",
            dtype="auto",
            quantization="awq",
            gpu_memory_utilization=0.7,
            default_sampling_params=SamplingParams(
                temperature=1.0, top_p=1.0, top_k=-1, max_tokens=256
            ),
        ).dict(),
    ),
}
