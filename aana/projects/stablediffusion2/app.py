import argparse

from aana.api.event_handlers.rate_limit_handler import RateLimitHandler
from aana.configs.deployments import (
    stablediffusion2_deployment,
)
from aana.projects.stablediffusion2.endpoints import ImageGenerationEndpoint
from aana.sdk import AanaSDK

deployments = [
    {
        "name": "image_generation_deployment",
        "instance": stablediffusion2_deployment,
    },
]

endpoints = [
    {
        "name": "generate_image",
        "path": "/generate_image",
        "summary": "Generates an image from a text prompt",
        "endpoint_cls": ImageGenerationEndpoint,
    },
    {
        "name": "generate_image_rate_limited",
        "path": "/generate_image_rate_limited",
        "summary": "Generates an image from a text prompt",
        "endpoint_cls": ImageGenerationEndpoint,
        "event_handlers": [RateLimitHandler(1, 30)],
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
            event_handlers=endpoint.get("event_handlers", []),
        )

    aana_app.deploy(blocking=True)
