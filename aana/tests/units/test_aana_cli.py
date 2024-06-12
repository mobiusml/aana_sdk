# ruff: noqa: S101, S113
import importlib
import uuid
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from aana.cli import cli

test_app_path = "aana.projects.lowercase.app:aana_app"
port = 5000


@pytest.fixture
def config_paths():
    """Fixture to create config paths and clean them up after the test."""
    # Temporary file paths creation
    app_module, _ = test_app_path.split(":")
    output_dir = Path(importlib.util.find_spec(app_module).origin).parent

    app_config_name = f"app_config_{uuid.uuid4().hex}"
    config_name = f"config_{uuid.uuid4().hex}"

    app_config_path = output_dir / f"{app_config_name}.py"
    config_path = output_dir / f"{config_name}.yaml"

    # Yield paths to use in the test
    yield app_config_path, config_path

    # Cleanup code after test execution
    app_config_path.unlink(missing_ok=True)
    config_path.unlink(missing_ok=True)


# TODO: Fix the test
# class TestAppManager:
#     """Context manager to handle the test app process."""

#     def __init__(self, target):
#         """Initialize the context manager."""
#         self.process = multiprocessing.Process(target=target)

#     def __enter__(self):
#         """Start the process when entering the context manager."""
#         self.process.start()
#         return self.process

#     def __exit__(self, exc_type, exc_value, traceback):
#         """Terminate the process when exiting the context manager."""
#         self.process.terminate()


# def deploy_test_app():
#     """Starting the CLI application deployment."""
#     CliRunner().invoke(cli, ["deploy", test_app_path, "--port", str(port)])


# @pytest.mark.asyncio
# async def test_aana_deploy():
#     """Test aana deploy command."""
#     with TestAppManager(deploy_test_app):
#         timeout = 30  # timeout in seconds
#         url = f"http://localhost:{port}/api/ready"

#         # Polling the endpoint until the app is ready or timeout expires
#         for _ in range(timeout):
#             try:
#                 response = requests.get(url, timeout=1)
#                 if response.status_code == 200:
#                     break
#             except requests.exceptions.RequestException:
#                 pass

#             await asyncio.sleep(1)
#         else:
#             # Only raises if the for loop completes without breaking (i.e. timeout)
#             raise Exception("App not started within the given timeout")

#         assert response.status_code == 200
#         assert response.json() == {"ready": True}

#         # Test lowercase endpoint
#         data = {"text": ["Hello World!", "This is a test."]}
#         response = requests.post(
#             f"http://localhost:{port}/lowercase",
#             data={"body": json.dumps(data)},
#         )
#         assert response.status_code == 200
#         lowercase_text = response.json().get("text")
#         assert lowercase_text == ["hello world!", "this is a test."]


def test_aana_build(config_paths):
    """Test aana build command."""
    expected_app_config = (
        "from aana.projects.lowercase.app import aana_app\n"
        "\n"
        "task_queue_deployment = aana_app.get_deployment_app('task_queue_deployment')\n"
        "lowercase_deployment = aana_app.get_deployment_app('lowercase_deployment')\n"
        "lowercase_app = aana_app.get_main_app()\n"
    )

    app_config_path, config_path = config_paths

    result = CliRunner().invoke(
        cli,
        [
            "build",
            test_app_path,
            "--port",
            str(port),
            "--app-config-name",
            app_config_path.stem,
            "--config-name",
            config_path.stem,
        ],
    )

    assert str(app_config_path) in result.output
    assert str(config_path) in result.output

    assert app_config_path.exists()
    assert config_path.exists()

    with app_config_path.open() as f:
        app_config = f.read()

    assert app_config == expected_app_config

    with config_path.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)  # noqa: S506

    assert config["http_options"]["port"] == port
    assert "applications" in config
    assert len(config["applications"]) == 3
    for app in config["applications"]:
        assert app["name"] in [
            "lowercase_deployment",
            "lowercase_app",
            "task_queue_deployment",
        ]


def test_aana_missing_app_path():
    """Test that the CLI commands fail when the APP_PATH argument is missing."""
    result = CliRunner().invoke(cli, ["deploy"])
    assert result.exit_code == 2
    assert "Error: Missing argument 'APP_PATH'." in result.output

    result = CliRunner().invoke(cli, ["build"])
    assert result.exit_code == 2
    assert "Error: Missing argument 'APP_PATH'." in result.output

    result = CliRunner().invoke(cli, ["migrate"])
    assert result.exit_code == 2
    assert "Error: Missing argument 'APP_PATH'." in result.output
