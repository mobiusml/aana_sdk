# ruff: noqa: S101
import json
import os
import tempfile
import threading
from pathlib import Path

import portpicker
import pytest
import ray
from ray import serve

from aana.api.request_handler import RequestHandler
from aana.configs.build import get_configuration
from aana.configs.db import (
    DBConfig,
    DbType,
    SQLiteConfig,
)
from aana.configs.deployments import deployments as all_deployments
from aana.configs.endpoints import endpoints as all_endpoints
from aana.configs.pipeline import nodes as all_nodes
from aana.configs.settings import settings as aana_settings
from aana.tests.utils import call_and_check_endpoint, clear_database
from aana.utils.json import json_serializer_default

gpu_lock = threading.Lock()


@pytest.fixture(scope="session")
def ray_setup():
    """Setup Ray instance."""
    ray.init()
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def setup_deployment(ray_setup, request):
    """Setup Ray Deployment."""

    def start_deployment(deployment):
        """Start deployment."""
        port = portpicker.pick_unused_port()
        name = request.node.name.replace("/", "_")
        route_prefix = f"/test/{name}"
        print(
            f"Starting deployment {name} on port {port} with route prefix {route_prefix}"
        )
        handle = serve.run(deployment, port=port, name=name, route_prefix=route_prefix)
        return handle, port, route_prefix

    yield start_deployment

    serve.shutdown()


@pytest.fixture(scope="module")
def ray_serve_setup(setup_deployment, request):
    """Setup the Ray Serve from specified endpoints and nodes."""

    def start_ray_serve(endpoints, nodes, context, runtime_env=None):
        if runtime_env is None:
            runtime_env = {}
        server = RequestHandler.options(
            ray_actor_options={"runtime_env": runtime_env}
        ).bind(endpoints, nodes, context)
        return setup_deployment(server)

    yield start_ray_serve

    serve.shutdown()


@pytest.fixture(scope="module")
def app_setup(ray_serve_setup):
    """Setup app for a specific target."""
    # create temporary database
    tmp_database_path = Path(tempfile.mkstemp(suffix=".db")[1])
    db_config = DBConfig(
        datastore_type=DbType.SQLITE,
        datastore_config=SQLiteConfig(path=tmp_database_path),
    )
    # set environment variable for the database config so Ray can find it
    os.environ["DB_CONFIG"] = json.dumps(db_config, default=json_serializer_default)
    # set database config in aana settings
    aana_settings.db_config = db_config

    # clear database before running the test
    clear_database(aana_settings)

    def start_app(target):
        # get configuration for the target deployment
        configuration = get_configuration(
            target,
            endpoints=all_endpoints,
            nodes=all_nodes,
            deployments=all_deployments,
        )
        endpoints = configuration["endpoints"]
        pipeline_nodes = configuration["nodes"]
        deployments = configuration["deployments"]
        runtime_env = {
            "env_vars": {
                "DB_CONFIG": json.dumps(db_config, default=json_serializer_default)
            }
        }
        context = {
            "deployments": {
                name: deployment.bind() for name, deployment in deployments.items()
            }
        }
        return ray_serve_setup(endpoints, pipeline_nodes, context, runtime_env)

    yield start_app

    # delete temporary database
    tmp_database_path.unlink()


@pytest.fixture(scope="module")
def call_endpoint(app_setup, request):
    """Call endpoint."""
    target = request.param
    handle, port, route_prefix = app_setup(target)

    def _call_endpoint(
        endpoint_path: str,
        data: dict,
        ignore_expected_output: bool = False,
        expected_error: str | None = None,
    ) -> dict | list:
        return call_and_check_endpoint(
            target,
            port,
            route_prefix,
            endpoint_path,
            data,
            expected_error=expected_error,
            ignore_expected_output=ignore_expected_output,
        )

    return _call_endpoint
