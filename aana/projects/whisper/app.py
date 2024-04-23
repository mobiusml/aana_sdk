import argparse

from aana.configs.deployments import vad_deployment, whisper_medium_deployment
from aana.projects.whisper.endpoints import (
    DeleteMediaEndpoint,
    LoadTranscriptionEndpoint,
    TranscribeVideoEndpoint,
    TranscribeVideoInChunksEndpoint,
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
]

endpoints = [
    {
        "name": "whisper_transcribe",
        "path": "/video/transcribe",
        "summary": "Transcribe a video",
        "endpoint_cls": TranscribeVideoEndpoint,
    },
    {
        "name": "whisper_transcribe_in_chunks",
        "path": "/video/transcribe_in_chunks",
        "summary": "Transcribe a video using Whisper by segmenting it into chunks",
        "endpoint_cls": TranscribeVideoInChunksEndpoint,
    },
    {
        "name": "load_transcription",
        "path": "/video/get_transcription",
        "summary": "Load a transcription",
        "endpoint_cls": LoadTranscriptionEndpoint,
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
