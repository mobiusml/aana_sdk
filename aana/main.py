import argparse
import sys
import time
import traceback

DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"  # noqa: S104


def run():
    """Runs the application."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    arg_parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    arg_parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Specify the set of endpoints to be deployed.",
    )
    args = arg_parser.parse_args()

    import ray
    from ray import serve

    from aana.api.request_handler import RequestHandler
    from aana.configs.build import get_configuration
    from aana.configs.deployments import deployments as all_deployments
    from aana.configs.endpoints import endpoints as all_endpoints
    from aana.configs.pipeline import nodes as all_nodes

    configuration = get_configuration(
        args.target,
        endpoints=all_endpoints,
        nodes=all_nodes,
        deployments=all_deployments,
    )
    endpoints = configuration["endpoints"]
    pipeline_nodes = configuration["nodes"]
    deployments = configuration["deployments"]

    # Setup ray environment and serve
    ray.init(ignore_reinit_error=True)

    context = {
        "deployments": {
            name: deployment.bind() for name, deployment in deployments.items()
        }
    }
    try:
        server = RequestHandler.bind(endpoints, pipeline_nodes, context)
        serve.run(server, port=args.port, host=args.host)
        # TODO: add logging
        print("Deployed Serve app successfully.")
        while True:
            time.sleep(10)

    except KeyboardInterrupt:
        print("Got KeyboardInterrupt, shutting down...")
        serve.shutdown()
        sys.exit()

    except Exception:
        traceback.print_exc()
        print(
            "Received unexpected error, see console logs for more details. Shutting "
            "down..."
        )
        serve.shutdown()
        sys.exit()
