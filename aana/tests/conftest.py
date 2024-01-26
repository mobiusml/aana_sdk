# ruff: noqa: S101
import json
import os
import tempfile
import threading
from pathlib import Path

import pytest
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
from aana.tests.utils import clear_database, ray_init, ray_lock, ray_shutdown
from aana.utils.json import json_serializer_default

gpu_lock = threading.Lock()


@pytest.fixture(scope="module")
def ray_setup():
    """Setup the Ray environment and serve the endpoints.

    Returns:
        tuple: A tuple containing the handle to the Ray Serve app
        and the port on which the app is running.
    """

    def _ray_setup(endpoints, nodes, context):
        with ray_lock():
            ray_init()
            server = RequestHandler.bind(endpoints, nodes, context)
            port = 34422
            route_prefix = "/test"
            handle = serve.run(
                server, port=port, name="test", route_prefix=route_prefix
            )
            yield handle, port, route_prefix
            ray_shutdown()

    return _ray_setup


@pytest.fixture(scope="module")
def app_setup():
    """Setup for the target test.

    Call it like this in the test:
    port, route_prefix, target = next(setup(TARGET))
    """

    def _app_setup(target: str):
        """Set up Ray for the test.

        Args:
            target (str): the name of the target deployment

        Yields:
            tuple: port, route_prefix
        """
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

        with ray_lock():
            # start Ray
            ray_init()

            # initialize the deployments
            context = {
                "deployments": {
                    name: deployment.bind() for name, deployment in deployments.items()
                }
            }

            # start the server
            port = 34422
            test_name = target
            route_prefix = f"/{test_name}"
            server = RequestHandler.bind(endpoints, pipeline_nodes, context)
            serve.run(server, port=port, name=test_name, route_prefix=route_prefix)

            yield port, route_prefix

            # shutdown Ray
            ray_shutdown()

        # delete temporary database
        tmp_database_path.unlink()

    return _app_setup
