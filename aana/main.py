import argparse
from pathlib import Path

from mobius_pipeline.utils.general import import_from_path

from aana.sdk import AanaSDK


def run():
    """Main function to run the application."""
    arg_parser = argparse.ArgumentParser()
    subparsers = arg_parser.add_subparsers(
        dest="command", help="commands (deploy or build)", required=True
    )

    arg_parser.add_argument(
        "app_path",
        type=str,
        help="Path to the application (module:app)",
    )

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

    try:
        aana_app = import_from_path(args.app_path)
    except:
        print(f"Error: Could not import {args.app_path}")
        return

    if not isinstance(aana_app, AanaSDK):
        print(f"Error: {args.app_path} is not an AanaSDK instance")
        return

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
