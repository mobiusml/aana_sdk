import argparse

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

deployments = [
    {
        "name": "asr_deployment",
        "instance": whisper_medium_deployment,
    },
    {
        "name": "vad_deployment",
        "instance": vad_deployment,
    },
    {
        "name": "captioning_deployment",
        "instance": hf_blip2_opt_2_7b_deployment,
    },
    {
        "name": "llm_deployment",
        "instance": vllm_llama2_7b_chat_deployment,
    },
]

endpoints = [
    {
        "name": "index_video_stream",
        "path": "/video/index_stream",
        "summary": "Index a video and return the captions and transcriptions as a stream",
        "endpoint_cls": IndexVideoEndpoint,
    },
    {
        "name": "video_metadata",
        "path": "/video/metadata",
        "summary": "Load video metadata",
        "endpoint_cls": LoadVideoMetadataEndpoint,
    },
    {
        "name": "video_chat_stream",
        "path": "/video/chat_stream",
        "summary": "Chat with video using LLaMa2 7B Chat (streaming)",
        "endpoint_cls": VideoChatEndpoint,
    },
    {
        "name": "delete_media",
        "path": "/video/delete",
        "summary": "Delete a media",
        "endpoint_cls": DeleteMediaEndpoint,
    },
]

if __name__ == "__main__":
    """Runs the application."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--port", type=int, default=8000)
    arg_parser.add_argument("--host", type=str, default="127.0.0.1")
    args = arg_parser.parse_args()

    aana_app = AanaSDK(port=args.port, host=args.host, show_logs=True)

    for deployment in deployments:
        aana_app.register_deployment(
            name=deployment["name"],
            instance=deployment["instance"],
        )

    for endpoint in endpoints:
        aana_app.register_endpoint(
            name=endpoint["name"],
            path=endpoint["path"],
            summary=endpoint["summary"],
            endpoint_cls=endpoint["endpoint_cls"],
        )

    aana_app.deploy(blocking=True)