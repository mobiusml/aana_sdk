# This file is used to define fixtures that are used in the integration tests.
# The fixtures are used to setup Ray and Ray Serve, and to call the endpoints.
# The fixtures depend on each other, to setup the environment for the tests.
# Here is a dependency graph of the fixtures:
# ray_setup (session scope, starts Ray cluster)
#   -> setup_deployment (module scope, starts Ray deployment, args: deployment)
#     -> ray_serve_setup (module scope, starts Ray Serve app, args: endpoints, nodes, context, runtime_env)
#       -> app_setup (module scope, starts Ray Serve app for a specific target, args: target)
#         -> call_endpoint (module scope, calls endpoint, args: endpoint_path, data, ignore_expected_output, expected_error)


# ruff: noqa: S101
import os
import tempfile
from pathlib import Path

import portpicker
import pytest
import ray
from ray import serve

from aana.api.request_handler import RequestHandler
from aana.configs.build import get_configuration
from aana.configs.db import (
    DbSettings,
    DbType,
    SQLiteConfig,
)
from aana.configs.deployments import deployments as all_deployments
from aana.configs.endpoints import endpoints as all_endpoints
from aana.configs.pipeline import nodes as all_nodes
from aana.configs.settings import settings as aana_settings
from aana.tests.utils import (
    call_and_check_endpoint,
    clear_database,
    is_gpu_available,
    is_using_deployment_cache,
)
from aana.utils.general import jsonify


@pytest.fixture(scope="session")
def ray_setup():
    """Setup Ray cluster."""
    ray.init(num_cpus=10)  # pretend we have 10 cpus
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def setup_deployment(ray_setup, request):  # noqa: D417
    """Setup Ray Deployment.

    Args:
        deployment: The deployment to start.
        bind (bool): Whether to bind the deployment. Defaults to False.
    """

    def start_deployment(deployment, bind=False):
        """Start deployment."""
        port = portpicker.pick_unused_port()
        name = request.node.name.replace("/", "_")
        route_prefix = f"/test/{name}"
        if bind:
            if not is_gpu_available() and is_using_deployment_cache():
                # if GPU is not available and we are using deployment cache,
                # then we don't want to request GPU resources
                deployment = deployment.options(ray_actor_options={"num_gpus": 0})
            deployment = deployment.bind()
        handle = serve.run(deployment, port=port, name=name, route_prefix=route_prefix)
        return handle, port, route_prefix

    yield start_deployment

    serve.shutdown()


@pytest.fixture(scope="module")
def ray_serve_setup(setup_deployment, request):  # noqa: D417
    """Setup the Ray Serve app from specified endpoints and nodes.

    Args:
        endpoints: App endpoints.
        nodes: App nodes.
        context: App context.
        runtime_env: The runtime environment. Defaults to None.
    """

    def start_ray_serve(endpoints, nodes, context, runtime_env=None):
        if runtime_env is None:
            runtime_env = {}
        server = RequestHandler.options(
            num_replicas=aana_settings.num_workers,
            ray_actor_options={"runtime_env": runtime_env},
        ).bind(endpoints, nodes, context)
        return setup_deployment(server)

    yield start_ray_serve

    serve.shutdown()


@pytest.fixture(scope="module")
def app_setup(ray_serve_setup):  # noqa: D417
    """Setup Ray Serve app for a specific target.

    Args:
        target: The target deployment.
    """
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
                "DB_CONFIG": jsonify(db_config),
            }
        }
        context = {"deployments": {}}
        for name, deployment in deployments.items():
            if not is_gpu_available() and is_using_deployment_cache():
                # if GPU is not available and we are using deployment cache,
                # then we don't want to request GPU resources
                context["deployments"][name] = deployment.options(
                    ray_actor_options={"num_gpus": 0}
                ).bind()
            else:
                context["deployments"][name] = deployment.bind()

        return ray_serve_setup(endpoints, pipeline_nodes, context, runtime_env)

    yield start_app

    # delete temporary database
    tmp_database_path.unlink()


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
