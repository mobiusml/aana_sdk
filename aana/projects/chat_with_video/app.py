from aana.configs.deployments import (
    hf_blip2_opt_2_7b_deployment,
    vad_deployment,
    vllm_llama2_7b_chat_deployment,
    whisper_medium_deployment,
)
from aana.projects.chat_with_video.endpoints import (
    DeleteMediaEndpoint,
    IndexVideoEndpoint,
    LoadVideoMetadataEndpoint,
    VideoChatEndpoint,
)
from aana.sdk import AanaSDK

aana_app = AanaSDK(port=8000, show_logs=True)

aana_app.register_deployment(
    "asr_deployment",
    whisper_medium_deployment,
)

aana_app.register_deployment(
    "vad_deployment",
    vad_deployment,
)

aana_app.register_deployment(
    "captioning_deployment",
    hf_blip2_opt_2_7b_deployment,
)

aana_app.register_deployment(
    "llm_deployment",
    vllm_llama2_7b_chat_deployment,
)

aana_app.register_endpoint(
    name="index_video_stream",
    path="/video/index_stream",
    summary="Index a video and return the captions and transcriptions as a stream",
    endpoint_cls=IndexVideoEndpoint,
)

aana_app.register_endpoint(
    name="video_metadata",
    path="/video/metadata",
    summary="Load video metadata",
    endpoint_cls=LoadVideoMetadataEndpoint,
)

aana_app.register_endpoint(
    name="video_chat_stream",
    path="/video/chat_stream",
    summary="Chat with video using LLaMa2 7B Chat (streaming)",
    endpoint_cls=VideoChatEndpoint,
)

aana_app.register_endpoint(
    name="delete_media",
    path="/video/delete",
    summary="Delete a media",
    endpoint_cls=DeleteMediaEndpoint,
)

if __name__ == "__main__":
    aana_app.deploy(blocking=True)
