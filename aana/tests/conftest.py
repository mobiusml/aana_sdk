# This file is used to define fixtures that are used in the integration tests.
# The fixtures are used to setup Ray and Ray Serve, and to call the endpoints.
# The fixtures depend on each other, to setup the environment for the tests.
# Here is a dependency graph of the fixtures:
# app_setup (module scope, starts Ray and Ray Serve app for a specific target, args: deployments, endpoints)
#     -> call_endpoint (module scope, calls endpoint, args: endpoint_path, data, ignore_expected_output, expected_error)

# ruff: noqa: S101
import importlib
import os
import tempfile
from pathlib import Path

import pytest

from aana.configs.db import DbSettings, SQLiteConfig
from aana.configs.settings import settings as aana_settings
from aana.sdk import AanaSDK
from aana.storage.op import DbType
from aana.tests.utils import (
    call_and_check_endpoint,
    clear_database,
    is_gpu_available,
    is_using_deployment_cache,
)
from aana.utils.json import jsonify


@pytest.fixture(scope="module")
def app_setup():
    """Setup Ray Serve app for given deployments and endpoints."""
    # create temporary database
    tmp_database_path = Path(tempfile.mkstemp(suffix=".db")[1])
    db_config = DbSettings(
        datastore_type=DbType.SQLITE,
        datastore_config=SQLiteConfig(path=tmp_database_path),
    )
    # set environment variable for the database config so Ray can find it
    os.environ["DB_CONFIG"] = jsonify(db_config)

    # set database config in aana settings
    aana_settings.db_config = db_config

    # clear database before running the test
    clear_database(aana_settings)

    app = AanaSDK()
    app.connect(
        port=8000, show_logs=True, num_cpus=10
    )  # pretend we have 10 cpus for testing

    def start_app(deployments, endpoints):
        for deployment in deployments:
            deployment_instance = deployment["instance"]
            if not is_gpu_available() and is_using_deployment_cache():
                # if GPU is not available and we are using deployment cache,
                # then we don't want to request GPU resources
                deployment_instance = deployment_instance.options(
                    ray_actor_options={"num_gpus": 0}
                )

            app.register_deployment(
                name=deployment["name"], instance=deployment_instance
            )

        for endpoint in endpoints:
            app.register_endpoint(
                name=endpoint["name"],
                path=endpoint["path"],
                summary=endpoint["summary"],
                endpoint_cls=endpoint["endpoint_cls"],
                event_handlers=endpoint.get("event_handlers", []),
            )

        app.deploy(blocking=False)

        return app

    yield start_app

    # delete temporary database
    tmp_database_path.unlink()

    app.shutdown()


@pytest.fixture(scope="module")
def call_endpoint(app_setup, request):  # noqa: D417
    """Call endpoint.

    Args:
        endpoint_path: The endpoint path.
        data: The data to send.
        ignore_expected_output: Whether to ignore the expected output. Defaults to False.
        expected_error: The expected error. Defaults to None.
    """
    target = request.param

    module = importlib.import_module(f"aana.projects.{target}.app")
    deployments = module.deployments
    endpoints = module.endpoints

    aana_app = app_setup(deployments, endpoints)

    port = aana_app.port
    route_prefix = ""

    def _call_endpoint(
        endpoint_path: str,
        data: dict,
        ignore_expected_output: bool = False,
        expected_error: str | None = None,
    ) -> dict | list:
        endpoint = None
        for e in endpoints:
            if e["path"] == endpoint_path:
                endpoint = e
                break
        if endpoint is None:
            raise ValueError(f"Endpoint with path {endpoint_path} not found")  # noqa: TRY003
        is_streaming = endpoint["endpoint_cls"].is_streaming_response()

        return call_and_check_endpoint(
            target=target,
            port=port,
            route_prefix=route_prefix,
            endpoint_path=endpoint_path,
            data=data,
            is_streaming=is_streaming,
            expected_error=expected_error,
            ignore_expected_output=ignore_expected_output,
        )

    return _call_endpoint


@pytest.fixture(scope="module")
def one_request_worker():
    """Fixture to update settings to only run one request worker."""
    aana_settings.num_workers = 1
    yield
