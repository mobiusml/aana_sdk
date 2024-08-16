# This file is used to define fixtures that are used in the integration tests.
# ruff: noqa: S101
import os
import tempfile
from pathlib import Path
from typing import Any

import portpicker
import pytest
from pytest_postgresql import factories
from sqlalchemy.orm import Session

from aana.configs.db import DbSettings, PostgreSQLConfig, SQLiteConfig
from aana.configs.settings import settings as aana_settings
from aana.sdk import AanaSDK
from aana.storage.op import DbType, run_alembic_migrations
from aana.tests.utils import (
    is_gpu_available,
    send_api_request,
    verify_output,
)
from aana.utils.core import import_from
from aana.utils.json import jsonify

# Change file permission if the user is root
if os.geteuid() == 0:
    postgresql_proc = factories.postgresql_proc(executable="./pg_ctl.sh")
    postgresql = factories.postgresql("postgresql_proc")


@pytest.fixture(scope="module")
def app_factory():
    """Factory fixture to create and configure the app."""

    def create_app(app_module, app_name):
        # Create a temporary database for testing
        tmp_database_path = Path(tempfile.mkstemp(suffix=".db")[1])
        db_config = DbSettings(
            datastore_type=DbType.SQLITE,
            datastore_config=SQLiteConfig(path=tmp_database_path),
        )
        os.environ["DB_CONFIG"] = jsonify(db_config)

        aana_settings.db_config = db_config
        aana_settings.db_config._engine = None

        # Import and start the app
        app = import_from(app_module, app_name)
        app.connect(port=portpicker.pick_unused_port(), show_logs=True, num_cpus=10)
        app.migrate()
        app.deploy()

        return app, tmp_database_path

    return create_app


@pytest.fixture(scope="module")
def create_app():
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

    run_alembic_migrations(aana_settings)

    app = AanaSDK()
    app.connect(
        port=portpicker.pick_unused_port(), show_logs=True, num_cpus=10
    )  # pretend we have 10 cpus for testing
    app.migrate()

    def start_app(deployments, endpoints):
        for deployment in deployments:
            deployment_instance = deployment["instance"]

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


@pytest.fixture(scope="class")
def setup_deployment(create_app, request):
    """Start the app with provided deployment."""
    deployment_name = request.param[0]
    deployment = request.param[1]

    # skip the test if GPU is not available and the deployment requires GPU
    num_gpus = deployment.ray_actor_options.get("num_gpus", 0)
    if not is_gpu_available() and num_gpus > 0:
        pytest.skip("GPU is not available")

    # keep it the same for all tests so the deployment is replaced
    # when the fixture is called again
    handle_name = "test_deployment"

    deployments = [
        {
            "name": handle_name,
            "instance": deployment,
        }
    ]
    endpoints = []

    return deployment_name, handle_name, create_app(deployments, endpoints)


@pytest.fixture(scope="module")
def call_endpoint(app_setup):
    """Call an endpoint and verify the output."""
    app = app_setup

    def _call_endpoint(
        endpoint_path: str,
        data: dict[str, Any],
        expected_error: str | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        endpoint = next(
            (e for e in app.endpoints.values() if e.path == endpoint_path), None
        )
        if endpoint is None:
            raise ValueError(f"Endpoint with path {endpoint_path} not found")  # noqa: TRY003
        response = send_api_request(endpoint=endpoint, app=app, data=data)
        verify_output(
            endpoint=endpoint,
            response=response,
            expected_error=expected_error,
        )
        return response

    return _call_endpoint


@pytest.fixture(scope="module")
def one_request_worker():
    """Fixture to update settings to only run one request worker."""
    aana_settings.num_workers = 1
    yield


@pytest.fixture(scope="function")
def sqlite_db_session():
    """Creates a new sql database file and session for each test."""
    with tempfile.NamedTemporaryFile(dir=aana_settings.tmp_data_dir) as tmp:
        # Configure the database to use the temporary file
        aana_settings.db_config.datastore_type = DbType.SQLITE
        aana_settings.db_config.datastore_config = SQLiteConfig(path=tmp.name)
        os.environ["DB_CONFIG"] = jsonify(aana_settings.db_config)

        # Reset the engine
        aana_settings.db_config._engine = None

        run_alembic_migrations(aana_settings)

        # Create a new session
        engine = aana_settings.db_config.get_engine()
        with Session(engine) as session:
            yield session


@pytest.fixture(scope="function")
def postgres_db_session(postgresql):
    """Creates a new postgres database and session for each test."""
    aana_settings.db_config.datastore_type = DbType.POSTGRESQL
    aana_settings.db_config.datastore_config = PostgreSQLConfig(
        host=postgresql.info.host,
        port=postgresql.info.port,
        user=postgresql.info.user,
        password=postgresql.info.password,
        database=postgresql.info.dbname,
    )
    os.environ["DB_CONFIG"] = jsonify(aana_settings.db_config)

    # Reset the engine
    aana_settings.db_config._engine = None

    # Run migrations to set up the schema
    run_alembic_migrations(aana_settings)

    # Create a new session
    engine = aana_settings.db_config.get_engine()
    with Session(engine) as session:
        yield session


@pytest.fixture(params=["sqlite_db_session", "postgres_db_session"])
def db_session(request):
    """Iterate over different database type for db tests."""
    return request.getfixturevalue(request.param)
