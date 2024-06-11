import argparse

from aana.configs.deployments import hf_blip2_opt_2_7b_deployment
from aana.projects.process_stream.endpoints import (
    CaptionStreamEndpoint,
)
from aana.sdk import AanaSDK

deployments = [
    {
        "name": "captioning_deployment",
        "instance": hf_blip2_opt_2_7b_deployment,
    },
]

endpoints = [
    {
        "name": "caption_live_stream",
        "path": "/stream/caption_stream",
        "summary": "Process a live stream and return the captions",
        "endpoint_cls": CaptionStreamEndpoint,
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
