import argparse
import sys
import time
import traceback

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--port", type=int, default=8000)
    arg_parser.add_argument("--host", type=str, default="0.0.0.0")
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

    configuration = get_configuration(args.target)
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
        handle = serve.run(server, port=args.port, host=args.host)
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
