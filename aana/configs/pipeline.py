"""
This file contains the pipeline configuration for the aana application.
It is used to generate the pipeline and the API endpoints.
"""

from aana.models.pydantic.prompt import Prompt
from aana.models.pydantic.sampling_params import SamplingParams

# container data model
# we don't enforce this data model for now but it's a good reference for writing paths and flatten_by
# class Container:
#     prompt: Prompt
#     sampling_params: SamplingParams
#     vllm_llama2_7b_chat_output_stream: str
#     vllm_llama2_7b_chat_output: str


nodes = [
    {
        "name": "prompt",
        "type": "input",
        "inputs": [],
        "outputs": [
            {"name": "prompt", "key": "prompt", "path": "prompt", "data_model": Prompt}
        ],
    },
    {
        "name": "sampling_params",
        "type": "input",
        "inputs": [],
        "outputs": [
            {
                "name": "sampling_params",
                "key": "sampling_params",
                "path": "sampling_params",
                "data_model": SamplingParams,
            }
        ],
    },
    {
        "name": "vllm_stream_llama2_7b_chat",
        "type": "ray_deployment",
        "deployment_name": "vllm_deployment_llama2_7b_chat",
        "data_type": "generator",
        "generator_path": "prompt",
        "method": "generate_stream",
        "inputs": [
            {"name": "prompt", "key": "prompt", "path": "prompt"},
            {
                "name": "sampling_params",
                "key": "sampling_params",
                "path": "sampling_params",
            },
        ],
        "outputs": [
            {
                "name": "vllm_llama2_7b_chat_output_stream",
                "key": "text",
                "path": "vllm_llama2_7b_chat_output_stream",
            }
        ],
    },
    {
        "name": "vllm_llama2_7b_chat",
        "type": "ray_deployment",
        "deployment_name": "vllm_deployment_llama2_7b_chat",
        "method": "generate",
        "inputs": [
            {"name": "prompt", "key": "prompt", "path": "prompt"},
            {
                "name": "sampling_params",
                "key": "sampling_params",
                "path": "sampling_params",
            },
        ],
        "outputs": [
            {
                "name": "vllm_llama2_7b_chat_output",
                "key": "text",
                "path": "vllm_llama2_7b_chat_output",
            }
        ],
    },
]
