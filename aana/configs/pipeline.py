"""
This file contains the pipeline configuration for the aana application.
It is used to generate the pipeline and the API endpoints.
"""

from aana.models.pydantic.captions import CaptionsList, VideoCaptionsList
from aana.models.pydantic.image_input import ImageInputList
from aana.models.pydantic.prompt import Prompt
from aana.models.pydantic.sampling_params import SamplingParams
from aana.models.pydantic.video_input import VideoInputList
from aana.models.pydantic.video_params import VideoParams

# container data model
# we don't enforce this data model for now but it's a good reference for writing paths and flatten_by
# class Container:
#     prompt: Prompt
#     sampling_params: SamplingParams
#     vllm_llama2_7b_chat_output_stream: str
#     vllm_llama2_7b_chat_output: str
#     vllm_zephyr_7b_beta_output_stream: str
#     vllm_zephyr_7b_beta_output: str
#     image_batch: ImageBatch
#     video_batch: VideoBatch
#
# class ImageBatch:
#     images: list[Image]
#
# class Image:
#     image: ImageInput
#     caption_hf_blip2_opt_2_7b: str
#
# class VideoBatch:
#     videos: list[Video]
#     params: VideoParams
# class Video:
#     video_input: VideoInput
#     video: VideoObject
#     frames: Frame
#     timestamps: Timestamps
#     duration: float
# class Frame:
#     image: Image
#     caption_hf_blip2_opt_2_7b: str


# pipeline configuration


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
    {
        "name": "vllm_stream_zephyr_7b_beta",
        "type": "ray_deployment",
        "deployment_name": "vllm_deployment_zephyr_7b_beta",
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
                "name": "vllm_zephyr_7b_beta_output_stream",
                "key": "text",
                "path": "vllm_zephyr_7b_beta_output_stream",
            }
        ],
    },
    {
        "name": "vllm_zephyr_7b_beta",
        "type": "ray_deployment",
        "deployment_name": "vllm_deployment_zephyr_7b_beta",
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
                "name": "vllm_zephyr_7b_beta_output",
                "key": "text",
                "path": "vllm_zephyr_7b_beta_output",
            }
        ],
    },
    {
        "name": "images",
        "type": "input",
        "inputs": [],
        "outputs": [
            {
                "name": "images",
                "key": "images",
                "path": "image_batch.images.[*].image",
                "data_model": ImageInputList,
            }
        ],
    },
    {
        "name": "hf_blip2_opt_2_7b",
        "type": "ray_deployment",
        "deployment_name": "hf_blip2_deployment_opt_2_7b",
        "method": "generate_batch",
        "inputs": [
            {
                "name": "images",
                "key": "images",
                "path": "image_batch.images.[*].image",
                "data_model": ImageInputList,
            }
        ],
        "outputs": [
            {
                "name": "captions_hf_blip2_opt_2_7b",
                "key": "captions",
                "path": "image_batch.images.[*].caption_hf_blip2_opt_2_7b",
                "data_model": CaptionsList,
            }
        ],
    },
    {
        "name": "videos",
        "type": "input",
        "inputs": [],
        "outputs": [
            {
                "name": "videos",
                "key": "videos",
                "path": "video_batch.videos.[*].video_input",
                "data_model": VideoInputList,
            }
        ],
    },
    {
        "name": "download_video",
        "type": "ray_task",
        "function": "aana.utils.video.download_video",
        "batched": True,
        "flatten_by": "video_batch.videos.[*]",
        "dict_output": False,
        "inputs": [
            {
                "name": "videos",
                "key": "video_input",
                "path": "video_batch.videos.[*].video_input",
            },
        ],
        "outputs": [
            {
                "name": "video_objects",
                "key": "output",
                "path": "video_batch.videos.[*].video",
            },
        ],
    },
    {
        "name": "video_params",
        "type": "input",
        "inputs": [],
        "outputs": [
            {
                "name": "video_params",
                "key": "video_params",
                "path": "video_batch.params",
                "data_model": VideoParams,
            }
        ],
    },
    {
        "name": "frame_extraction",
        "type": "ray_task",
        "function": "aana.utils.video.extract_frames_decord",
        "batched": True,
        "flatten_by": "video_batch.videos.[*]",
        "inputs": [
            {
                "name": "video_objects",
                "key": "video",
                "path": "video_batch.videos.[*].video",
            },
            {"name": "video_params", "key": "params", "path": "video_batch.params"},
        ],
        "outputs": [
            {
                "name": "frames",
                "key": "frames",
                "path": "video_batch.videos.[*].frames.[*].image",
            },
            {
                "name": "timestamps",
                "key": "timestamps",
                "path": "video_batch.videos.[*].timestamp",
            },
            {
                "name": "duration",
                "key": "duration",
                "path": "video_batch.videos.[*].duration",
            },
        ],
    },
    {
        "name": "hf_blip2_opt_2_7b_video",
        "type": "ray_deployment",
        "deployment_name": "hf_blip2_deployment_opt_2_7b",
        "method": "generate_batch",
        "flatten_by": "video_batch.videos.[*].frames.[*]",
        "inputs": [
            {
                "name": "frames",
                "key": "images",
                "path": "video_batch.videos.[*].frames.[*].image",
            }
        ],
        "outputs": [
            {
                "name": "video_captions_hf_blip2_opt_2_7b",
                "key": "captions",
                "path": "video_batch.videos.[*].frames.[*].caption_hf_blip2_opt_2_7b",
                "data_model": VideoCaptionsList,
            }
        ],
    },
]
