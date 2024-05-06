import sys

import click

from aana.sdk import AanaSDK
from aana.utils.general import import_from_path


@click.group()
def cli():
    """Aana CLI.

    It provides commands to deploy and build AanaSDK applications.
    """
    pass


def load_app(app_path: str):
    """Load the AanaSDK app from the given path.

    Args:
        app_path (str): Path to the application (module:app).
    """
    sys.path.insert(0, ".")  # to be able import the app from the current directory
    try:
        aana_app = import_from_path(app_path)
    except Exception as e:
        raise ImportError(f"Could not import {app_path}") from e  # noqa: TRY003

    if not isinstance(aana_app, AanaSDK):
        raise TypeError(f"{app_path} is not an AanaSDK instance")  # noqa: TRY003

    return aana_app


@cli.command()
@click.argument("app_path", type=str)
@click.option("--host", default="127.0.0.1", type=str, help="Host address")
@click.option("--port", default=8000, type=int, help="Port to run the application")
def deploy(app_path: str, host: str, port: int):
    """Deploy the application.

    APP_PATH: Path to the application (module:app).
    """
    aana_app = load_app(app_path)
    aana_app.connect(port=port, host=host, show_logs=True)
    aana_app.deploy(blocking=True)


@cli.command()
@click.argument("app_path", type=str)
@click.option("--host", default="0.0.0.0", type=str, help="Host address")  # noqa: S104
@click.option("--port", default=8000, type=int, help="Port to run the application")
@click.option(
    "--app_config_name",
    default="app_config",
    type=str,
    help="App config name. Default is app_config so app config will be saved under app_config.py",
)
@click.option(
    "--config_name",
    default="config",
    type=str,
    help="Config name. Default is config so config will be saved under config.yaml",
)
def build(app_path: str, host: str, port: int, app_config_name: str, config_name: str):
    """Build the application.

    APP_PATH: Path to the application (module:app).
    """
    aana_app = load_app(app_path)
    aana_app.build(
        import_path=app_path,
        host=host,
        port=port,
        app_config_name=app_config_name,
        config_name=config_name,
    )
