"""Pipeline configuration for the aana application.

It is used to generate the pipeline and the API endpoints.
"""

from aana.models.pydantic.asr_output import (
    AsrSegments,
    AsrSegmentsList,
    AsrTranscription,
    AsrTranscriptionInfo,
    AsrTranscriptionInfoList,
    AsrTranscriptionList,
)
from aana.models.pydantic.vad_output import VadSegments
from aana.models.pydantic.captions import CaptionsList, VideoCaptionsList
from aana.models.pydantic.chat_message import ChatDialog
from aana.models.pydantic.image_input import ImageInputList
from aana.models.pydantic.prompt import Prompt
from aana.models.pydantic.sampling_params import SamplingParams
from aana.models.pydantic.video_input import VideoInput, VideoInputList
from aana.models.pydantic.video_metadata import VideoMetadata
from aana.models.pydantic.video_params import VideoParams
from aana.models.pydantic.whisper_params import WhisperParams
from aana.models.pydantic.vad_params import VadParams

# container data model
# we don't enforce this data model for now but it's a good reference for writing paths and flatten_by
# class Container:
#     prompt: Prompt
#     sampling_params: SamplingParams
#     vllm_llama2_7b_chat_output_stream: str
#     vllm_llama2_7b_chat_output: str
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
        "name": "dialog",
        "type": "input",
        "inputs": [],
        "outputs": [
            {
                "name": "dialog",
                "key": "dialog",
                "path": "dialog",
                "data_model": ChatDialog,
            }
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
        "name": "vllm_llama2_7b_chat_dialog",
        "type": "ray_deployment",
        "deployment_name": "vllm_deployment_llama2_7b_chat",
        "method": "chat",
        "inputs": [
            {
                "name": "dialog",
                "key": "dialog",
                "path": "dialog",
                "data_model": ChatDialog,
            },
            {
                "name": "sampling_params",
                "key": "sampling_params",
                "path": "sampling_params",
            },
        ],
        "outputs": [
            {
                "name": "vllm_llama2_7b_chat_output_message",
                "key": "message",
                "path": "vllm_llama2_7b_chat_output_message",
            }
        ],
    },
    {
        "name": "vllm_llama2_7b_chat_dialog_stream",
        "type": "ray_deployment",
        "deployment_name": "vllm_deployment_llama2_7b_chat",
        "data_type": "generator",
        "generator_path": "dialog",
        "method": "chat_stream",
        "inputs": [
            {
                "name": "dialog",
                "key": "dialog",
                "path": "dialog",
                "data_model": ChatDialog,
            },
            {
                "name": "sampling_params",
                "key": "sampling_params",
                "path": "sampling_params",
            },
        ],
        "outputs": [
            {
                "name": "vllm_llama2_7b_chat_output_dialog_stream",
                "key": "text",
                "path": "vllm_llama2_7b_chat_output_dialog_stream",
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
        "name": "download_videos",
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
                "name": "frame_ids",
                "key": "frame_ids",
                "path": "video_batch.videos.[*].frames.[*].id",
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
        "name": "hf_blip2_opt_2_7b_videos",
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
                "name": "videos_captions_hf_blip2_opt_2_7b",
                "key": "captions",
                "path": "video_batch.videos.[*].frames.[*].caption_hf_blip2_opt_2_7b",
                "data_model": VideoCaptionsList,
            }
        ],
    },
    {
        "name": "whisper_params",
        "type": "input",
        "inputs": [],
        "outputs": [
            {
                "name": "whisper_params",
                "key": "whisper_params",
                "path": "video_batch.whisper_params",
                "data_model": WhisperParams,
            }
        ],
    },
    {
        "name": "whisper_medium_transcribe_videos",
        "type": "ray_deployment",
        "deployment_name": "whisper_deployment_medium",
        "method": "transcribe_batch",
        "inputs": [
            {
                "name": "video_objects",
                "key": "media_batch",
                "path": "video_batch.videos.[*].video",
            },
            {
                "name": "whisper_params",
                "key": "params",
                "path": "video_batch.whisper_params",
                "data_model": WhisperParams,
            },
        ],
        "outputs": [
            {
                "name": "videos_transcriptions_segments_whisper_medium",
                "key": "segments",
                "path": "video_batch.videos.[*].segments",
                "data_model": AsrSegmentsList,
            },
            {
                "name": "videos_transcriptions_info_whisper_medium",
                "key": "transcription_info",
                "path": "video_batch.videos.[*].transcription_info",
                "data_model": AsrTranscriptionInfoList,
            },
            {
                "name": "videos_transcriptions_whisper_medium",
                "key": "transcription",
                "path": "video_batch.videos.[*].transcription",
                "data_model": AsrTranscriptionList,
            },
        ],
    },
    {
        "name": "video",
        "type": "input",
        "inputs": [],
        "outputs": [
            {
                "name": "video",
                "key": "video",
                "path": "video.video_input",
                "data_model": VideoInput,
            }
        ],
    },
    {
        "name": "download_video",
        "type": "ray_task",
        "function": "aana.utils.video.download_video",
        "dict_output": False,
        "inputs": [
            {
                "name": "video",
                "key": "video_input",
                "path": "video.video_input",
            },
        ],
        "outputs": [
            {
                "name": "video_object",
                "key": "output",
                "path": "video.video",
            },
        ],
    },
    {
        "name": "generate_frames_for_video",
        "type": "ray_task",
        "function": "aana.utils.video.generate_frames_decord",
        "data_type": "generator",
        "generator_path": "video",
        "inputs": [
            {
                "name": "video_object",
                "key": "video",
                "path": "video.video",
            },
            {"name": "video_params", "key": "params", "path": "video_batch.params"},
        ],
        "outputs": [
            {
                "name": "video_frames",
                "key": "frames",
                "path": "video.frames.[*].image",
            },
            {
                "name": "video_frame_ids",
                "key": "frame_ids",
                "path": "video.frames.[*].id",
            },
            {
                "name": "video_timestamps",
                "key": "timestamps",
                "path": "video.timestamps",
            },
            {
                "name": "video_duration",
                "key": "duration",
                "path": "video.duration",
            },
        ],
    },
    {
        "name": "hf_blip2_opt_2_7b_video",
        "type": "ray_deployment",
        "deployment_name": "hf_blip2_deployment_opt_2_7b",
        "method": "generate_batch",
        "flatten_by": "video.frames.[*]",
        "inputs": [
            {
                "name": "video_frames",
                "key": "images",
                "path": "video.frames.[*].image",
                "partial": True,
            }
        ],
        "outputs": [
            {
                "name": "video_captions_hf_blip2_opt_2_7b",
                "key": "captions",
                "path": "video.frames.[*].caption_hf_blip2_opt_2_7b",
                "data_model": CaptionsList,
            }
        ],
    },
    {
        "name": "whisper_medium_transcribe_video",
        "type": "ray_deployment",
        "deployment_name": "whisper_deployment_medium",
        "data_type": "generator",
        "generator_path": "video",
        "method": "transcribe_stream",
        "inputs": [
            {
                "name": "video_object",
                "key": "media",
                "path": "video.video",
            },
            {
                "name": "whisper_params",
                "key": "params",
                "path": "video_batch.whisper_params",
                "data_model": WhisperParams,
            },
        ],
        "outputs": [
            {
                "name": "video_transcriptions_segments_whisper_medium",
                "key": "segments",
                "path": "video.segments",
                "data_model": AsrSegments,
            },
            {
                "name": "video_transcriptions_info_whisper_medium",
                "key": "transcription_info",
                "path": "video.transcription_info",
                "data_model": AsrTranscriptionInfo,
            },
            {
                "name": "video_transcriptions_whisper_medium",
                "key": "transcription",
                "path": "video.transcription",
                "data_model": AsrTranscription,
            },
        ],
    },
    {
        "name": "vad_params",
        "type": "input",
        "inputs": [],
        "outputs": [
            {
                "name": "vad_params",
                "key": "vad_params",
                "path": "video_batch.vad_params",
                "data_model": VadParams,
            }
        ],
    },
    {
        "name": "vad_batched_inference_video",
        "type": "ray_deployment",
        "deployment_name": "vad_deployment",
        "method": "asr_preprocess_vad",
        "inputs": [
            {
                "name": "video_object",
                "key": "media",
                "path": "video.video",
            },
            {
                "name": "vad_params",
                "key": "params",
                "path": "video_batch.vad_params",
            },
        ],
        "outputs": [
            {
                "name": "video_transcriptions_vad_segments",
                "key": "vad_segments",
                "path": "video.vad_segments",
                "data_model": VadSegments,
            },
        ],
    },
    {
        "name": "whisper_medium_batched_inference_video",
        "type": "ray_deployment",
        "deployment_name": "whisper_deployment_medium",
        "data_type": "generator",
        "generator_path": "video",
        "method": "batched_inference",
        "inputs": [
            {
                "name": "video_object",
                "key": "media",
                "path": "video.video",
            },
            {
                "name": "video_transcriptions_vad_segments",
                "key": "vad_segments",
                "path": "video.vad_segments",
            },
            {
                "name": "whisper_params",
                "key": "params",
                "path": "video_batch.whisper_params",
                "data_model": WhisperParams,
            },
        ],
        "outputs": [
            {
                "name": "video_transcriptions_segments_batched_whisper_medium",
                "key": "segments",
                "path": "video.segments_batched",
                "data_model": AsrSegments,
            },
            {
                "name": "video_transcriptions_info_batched_whisper_medium",
                "key": "transcription_info",
                "path": "video.transcription_info_batched",
                "data_model": AsrTranscriptionInfo,
            },
            {
                "name": "video_transcriptions_batched_whisper_medium",
                "key": "transcription",
                "path": "video.transcription_batched",
                "data_model": AsrTranscription,
            },
        ],
    },
    {
        "name": "media_id",
        "type": "input",
        "inputs": [],
        "outputs": [
            {
                "name": "media_id",
                "key": "media_id",
                "path": "media_id",
            }
        ],
    },
    {
        "name": "question",
        "type": "input",
        "inputs": [],
        "outputs": [
            {
                "name": "question",
                "key": "question",
                "path": "question",
            }
        ],
    },
    {
        "name": "load_video_metadata",
        "type": "function",
        "function": "aana.utils.db.load_video_metadata",
        "dict_output": False,
        "inputs": [
            {
                "name": "media_id",
                "key": "media_id",
                "path": "media_id",
            },
        ],
        "outputs": [
            {
                "name": "video_metadata",
                "key": "output",
                "path": "video_metadata",
                "data_model": VideoMetadata,
            },
        ],
    },
    {
        "name": "load_video_transcription",
        "type": "function",
        "function": "aana.utils.db.load_video_transcription",
        "dict_output": True,
        "kwargs": {
            "model_name": "whisper_medium",
        },
        "inputs": [
            {
                "name": "media_id",
                "key": "media_id",
                "path": "media_id",
            },
        ],
        "outputs": [
            {
                "name": "video_transcriptions_info_whisper_medium_from_db",
                "key": "transcription_info",
                "path": "video_transcriptions_info_whisper_medium_from_db",
                "data_model": AsrTranscriptionInfo,
            },
            {
                "name": "video_transcriptions_segments_whisper_medium_from_db",
                "key": "segments",
                "path": "video_transcriptions_segments_whisper_medium_from_db",
                "data_model": AsrSegments,
            },
            {
                "name": "video_transcriptions_whisper_medium_from_db",
                "key": "transcription",
                "path": "video_transcriptions_whisper_medium_from_db",
                "data_model": AsrTranscription,
            },
        ],
    },
    {
        "name": "load_video_captions_hf_blip2_opt_2_7b",
        "type": "function",
        "function": "aana.utils.db.load_video_captions",
        "dict_output": True,
        "kwargs": {
            "model_name": "hf_blip2_opt_2_7b",
        },
        "inputs": [
            {
                "name": "media_id",
                "key": "media_id",
                "path": "media_id",
            }
        ],
        "outputs": [
            {
                "name": "video_captions_hf_blip2_opt_2_7b_from_db",
                "key": "captions",
                "path": "video_captions_hf_blip2_opt_2_7b_from_db",
            },
            {
                "name": "video_caption_timestamps_hf_blip2_opt_2_7b_from_db",
                "key": "timestamps",
                "path": "video_caption_timestamps_hf_blip2_opt_2_7b_from_db",
            },
            {
                "name": "video_caption_frame_ids_hf_blip2_opt_2_7b_from_db",
                "key": "frame_ids",
                "path": "video_caption_frame_ids_hf_blip2_opt_2_7b_from_db",
            },
        ],
    },
    {
        "name": "generate_combined_timeline",
        "type": "function",
        "function": "aana.utils.video.generate_combined_timeline",
        "inputs": [
            {
                "name": "video_transcriptions_segments_whisper_medium_from_db",
                "key": "transcription_segments",
                "path": "video_transcriptions_segments_whisper_medium_from_db",
            },
            {
                "name": "video_captions_hf_blip2_opt_2_7b_from_db",
                "key": "captions",
                "path": "video_captions_hf_blip2_opt_2_7b_from_db",
            },
            {
                "name": "video_caption_timestamps_hf_blip2_opt_2_7b_from_db",
                "key": "caption_timestamps",
                "path": "video_caption_timestamps_hf_blip2_opt_2_7b_from_db",
            },
        ],
        "outputs": [
            {
                "name": "video_timeline",
                "key": "timeline",
                "path": "video_timeline",
            },
        ],
    },
    {
        "name": "generate_dialog",
        "type": "function",
        "function": "aana.utils.video.generate_dialog",
        "dict_output": False,
        "inputs": [
            {
                "name": "video_metadata",
                "key": "metadata",
                "path": "video_metadata",
            },
            {
                "name": "video_timeline",
                "key": "timeline",
                "path": "video_timeline",
            },
            {
                "name": "question",
                "key": "question",
                "path": "question",
            },
        ],
        "outputs": [
            {
                "name": "video_chat_dialog",
                "key": "output",
                "path": "video_chat_dialog",
            },
        ],
    },
    {
        "name": "vllm_llama2_7b_chat_dialog_stream_video",
        "type": "ray_deployment",
        "deployment_name": "vllm_deployment_llama2_7b_chat",
        "data_type": "generator",
        "generator_path": "dialog",
        "method": "chat_stream",
        "inputs": [
            {
                "name": "video_chat_dialog",
                "key": "dialog",
                "path": "video_chat_dialog",
                "data_model": ChatDialog,
            },
            {
                "name": "sampling_params",
                "key": "sampling_params",
                "path": "sampling_params",
            },
        ],
        "outputs": [
            {
                "name": "vllm_llama2_7b_chat_output_dialog_stream_video",
                "key": "text",
                "path": "vllm_llama2_7b_chat_output_dialog_stream_video",
            }
        ],
    },
    {
        "name": "save_video",
        "type": "function",
        "function": "aana.utils.db.save_video",
        "dict_output": True,
        "inputs": [
            {
                "name": "video_object",
                "key": "video",
                "path": "video.video",
            },
        ],
        "outputs": [
            {
                "name": "video_media_id",
                "key": "media_id",
                "path": "video.media_id",
            },
            {
                "name": "video_id",
                "key": "video_id",
                "path": "video.id",
            },
        ],
    },
    {
        "name": "delete_media",
        "type": "function",
        "function": "aana.utils.db.delete_media",
        "dict_output": True,
        "inputs": [
            {
                "name": "media_id",
                "key": "media_id",
                "path": "media_id",
            },
        ],
        "outputs": [
            {
                "name": "deleted_media_id",
                "key": "media_id",
                "path": "deleted_media_id",
            }
        ],
    },
    {
        "name": "save_videos_info",
        "type": "function",
        "function": "aana.utils.db.save_video_batch",
        "dict_output": True,
        "inputs": [
            {
                "name": "video_objects",
                "key": "videos",
                "path": "video_batch.videos.[*].video",
            },
        ],
        "outputs": [
            {
                "name": "videos_media_ids",
                "key": "media_ids",
                "path": "video_batch.[*].media_id",
            },
            {
                "name": "video_ids",
                "key": "video_ids",
                "path": "video_batch.[*].id",
            },
        ],
    },
    {
        "name": "save_video_transcription",
        "type": "function",
        "function": "aana.utils.db.save_video_transcription",
        "kwargs": {
            "model_name": "whisper_medium",
        },
        "dict_output": True,
        "inputs": [
            {
                "name": "video_media_id",
                "key": "media_id",
                "path": "video.media_id",
            },
            {
                "name": "video_transcriptions_info_whisper_medium",
                "key": "transcription_info",
                "path": "video.transcription_info",
            },
            {
                "name": "video_transcriptions_segments_whisper_medium",
                "key": "segments",
                "path": "video.segments",
            },
            {
                "name": "video_transcriptions_whisper_medium",
                "key": "transcription",
                "path": "video.transcription",
            },
        ],
        "outputs": [
            {
                "name": "transcription_id",
                "key": "transcription_id",
                "path": "video.transcription_id",
            }
        ],
    },
    {
        "name": "save_video_transcription_batched",
        "type": "function",
        "function": "aana.utils.db.save_video_transcription",
        "kwargs": {
            "model_name": "whisper_medium",
        },
        "dict_output": True,
        "inputs": [
            {
                "name": "video_media_id",
                "key": "media_id",
                "path": "video.media_id",
            },
            {
                "name": "video_transcriptions_info_batched_whisper_medium",
                "key": "transcription_info",
                "path": "video.transcription_info_batched",
            },
            {
                "name": "video_transcriptions_segments_batched_whisper_medium",
                "key": "segments",
                "path": "video.segments_batched",
            },
            {
                "name": "video_transcriptions_whisper_medium",
                "key": "transcription",
                "path": "video.transcription_batched",
            },
        ],
        "outputs": [
            {
                "name": "transcription_id_batched",
                "key": "transcription_id",
                "path": "video.transcription_id_batched",
            }
        ],
    },
    {
        "name": "save_transcripts_batch_medium",
        "type": "function",
        "function": "aana.utils.db.save_transcripts_batch",
        "kwargs": {
            "model_name": "whisper_medium",
        },
        "dict_output": True,
        "inputs": [
            {
                "name": "videos_media_ids",
                "key": "media_ids",
                "path": "video_batch.[*].media_id",
            },
            {
                "name": "videos_transcriptions_info_whisper_medium",
                "key": "transcription_info_list",
                "path": "video_batch.videos.[*].transcription_info",
            },
            {
                "name": "videos_transcriptions_segments_whisper_medium",
                "key": "segments_list",
                "path": "video_batch.videos.[*].segments",
            },
            {
                "name": "videos_transcriptions_whisper_medium",
                "key": "transcription_list",
                "path": "video_batch.videos.[*].transcription",
            },
        ],
        "outputs": [
            {
                "name": "videos_transcription_ids",
                "key": "transcription_ids",
                "path": "video_batch.videos.[*].transcription.id",
            }
        ],
    },
    {
        "name": "save_video_captions_hf_blip2_opt_2_7b",
        "type": "function",
        "function": "aana.utils.db.save_video_captions",
        "kwargs": {
            "model_name": "hf_blip2_opt_2_7b",
        },
        "dict_output": True,
        "inputs": [
            {
                "name": "video_media_id",
                "key": "media_id",
                "path": "video.media_id",
            },
            {
                "name": "video_captions_hf_blip2_opt_2_7b",
                "key": "captions",
                "path": "video.frames.[*].caption_hf_blip2_opt_2_7b",
            },
            {
                "name": "video_timestamps",
                "key": "timestamps",
                "path": "video.timestamps",
            },
            {
                "name": "video_frame_ids",
                "key": "frame_ids",
                "path": "video.frames.[*].id",
            },
        ],
        "outputs": [
            {
                "name": "caption_ids",
                "key": "caption_ids",
                "path": "video.frames.[*].caption_hf_blip2_opt_2_7b.id",
            }
        ],
    },
]
