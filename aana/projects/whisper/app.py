import argparse
from pathlib import Path

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

aana_app = AanaSDK(name="whisper_app")

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


def main():
    """Main function to run the application."""
    arg_parser = argparse.ArgumentParser()
    subparsers = arg_parser.add_subparsers(dest="command", help="commands")

    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy the application")
    deploy_parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the application"
    )
    deploy_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host address"
    )

    # Build command
    build_parser = subparsers.add_parser("build", help="Build the application")
    build_parser.add_argument(
        "--import_path", type=str, help="Import path", required=True
    )
    build_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",  # noqa: S104
        help="Host address",
    )
    build_parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the application"
    )
    build_parser.add_argument(
        "--output_dir", type=str, help="Output directory", default=Path(__file__).parent
    )

    args = arg_parser.parse_args()

    if args.command == "deploy":
        aana_app.connect(port=args.port, host=args.host, show_logs=True)
        aana_app.deploy(blocking=True)
    elif args.command == "build":
        aana_app.build(
            import_path=args.import_path,
            host=args.host,
            port=args.port,
            output_dir=args.output_dir,
        )
    else:
        arg_parser.print_help()


if __name__ == "__main__":
    main()
