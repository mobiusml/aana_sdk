# This file is used to define fixtures that are used in the integration tests.
# ruff: noqa: S101
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from pytest_postgresql import factories

from aana.configs.db import DbSettings, PostgreSQLConfig, SQLiteConfig
from aana.configs.settings import settings as aana_settings
from aana.sdk import AanaSDK
from aana.storage.models.api_key import ApiServiceBase
from aana.storage.op import DatabaseSessionManager, DbType, run_alembic_migrations
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


def random_port() -> int:
    """Return a random port."""
    return random.randint(30000, 60000)  # noqa: S311


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

        # Import and start the app
        app = import_from(app_module, app_name)
        try:
            # pretend we have 10 cpus for testing
            app.connect(port=random_port(), show_logs=True, num_cpus=10)
        except ValueError:
            # if the port is already in use, try again
            app.shutdown()
            app = import_from(app_module, app_name)
            app.connect(port=random_port(), show_logs=True, num_cpus=10)
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

    os.environ["API_SERVICE__ENABLED"] = "False"
    aana_settings.api_service.enabled = False

    app = AanaSDK()
    try:
        # pretend we have 10 cpus for testing
        app.connect(port=random_port(), show_logs=True, num_cpus=10)
    except ValueError:
        # if the port is already in use, try again
        app.shutdown()
        app = AanaSDK()
        app.connect(port=random_port(), show_logs=True, num_cpus=10)
    app.migrate()

    def start_app(deployments, endpoints):
        for deployment in deployments:
            deployment_instance = deployment["instance"]

            app.register_deployment(
                name=deployment["name"], instance=deployment_instance
            )

        for endpoint in endpoints:
            app.register_endpoint(**endpoint)

        app.deploy(blocking=False)

        return app

    try:
        yield start_app
    finally:
        # delete temporary database
        tmp_database_path.unlink()

        app.shutdown()


@pytest.fixture(scope="module")
def create_app_with_api_service():
    """Setup Ray Serve app for given deployments and endpoints with API service enabled."""
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

    # Setup API service database
    tmp_api_service_database_path = Path(tempfile.mkstemp(suffix=".db")[1])
    api_service_db_config = DbSettings(
        datastore_type=DbType.SQLITE,
        datastore_config=SQLiteConfig(path=tmp_api_service_database_path),
    )
    os.environ["API_SERVICE_DB_CONFIG"] = jsonify(api_service_db_config)
    aana_settings.api_service_db_config = api_service_db_config

    os.environ["API_SERVICE__ENABLED"] = "True"
    aana_settings.api_service.enabled = True

    app = AanaSDK()
    try:
        # pretend we have 10 cpus for testing
        app.connect(port=random_port(), show_logs=True, num_cpus=10)
    except ValueError:
        # if the port is already in use, try again
        app.shutdown()
        app = AanaSDK()
        app.connect(port=random_port(), show_logs=True, num_cpus=10)
    app.migrate()

    def start_app(deployments, endpoints):
        for deployment in deployments:
            deployment_instance = deployment["instance"]

            app.register_deployment(
                name=deployment["name"], instance=deployment_instance
            )

        for endpoint in endpoints:
            app.register_endpoint(**endpoint)

        app.deploy(blocking=False)

        return app

    try:
        yield start_app
    finally:
        # delete temporary databases
        tmp_database_path.unlink()
        tmp_api_service_database_path.unlink()

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


@pytest.fixture
def db_sqlite_config():
    """Create a SQLite database config."""
    with tempfile.NamedTemporaryFile(dir=aana_settings.tmp_data_dir) as tmp:
        db_config = DbSettings(
            datastore_type=DbType.SQLITE, datastore_config=SQLiteConfig(path=tmp.name)
        )
        yield db_config


@pytest.fixture
def db_postgres_config(postgresql):
    """Create a PostgreSQL database config."""
    db_config = DbSettings(
        datastore_type=DbType.POSTGRESQL,
        datastore_config=PostgreSQLConfig(
            host=postgresql.info.host,
            port=str(postgresql.info.port),
            user=postgresql.info.user,
            password=postgresql.info.password,
            database=postgresql.info.dbname,
        ),
    )

    yield db_config


@pytest.fixture(params=["db_sqlite_config", "db_postgres_config"])
def db_config(request):
    """Fixture to provide both SQLite and PostgreSQL database configs."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def api_service_db_sqlite_config():
    """Create a SQLite database config."""
    with tempfile.NamedTemporaryFile(dir=aana_settings.tmp_data_dir) as tmp:
        db_config = DbSettings(
            datastore_type=DbType.SQLITE, datastore_config=SQLiteConfig(path=tmp.name)
        )
        yield db_config


@pytest.fixture
def api_service_db_postgres_config(postgresql):
    """Create a PostgreSQL database config."""
    db_config = DbSettings(
        datastore_type=DbType.POSTGRESQL,
        datastore_config=PostgreSQLConfig(
            host=postgresql.info.host,
            port=str(postgresql.info.port),
            user=postgresql.info.user,
            password=postgresql.info.password,
            database=postgresql.info.dbname,
        ),
    )

    yield db_config


@pytest.fixture(
    params=["api_service_db_sqlite_config", "api_service_db_postgres_config"]
)
def api_service_db_config(request):
    """Fixture to provide both SQLite and PostgreSQL database configs for the API service."""
    return request.getfixturevalue(request.param)


@pytest_asyncio.fixture
async def db_session_manager(db_config):
    """Create a new session for each test."""
    aana_settings.db_config = db_config

    run_alembic_migrations(aana_settings)

    session_manager = DatabaseSessionManager(aana_settings)

    try:
        yield session_manager
    finally:
        await session_manager.close()


@pytest_asyncio.fixture
async def db_session_manager_with_api_service(db_config, api_service_db_config):
    """Create a new session with both main and api service databases."""
    aana_settings.api_service.enabled = True
    aana_settings.db_config = db_config
    aana_settings.api_service_db_config = api_service_db_config

    run_alembic_migrations(aana_settings)

    session_manager = DatabaseSessionManager(aana_settings)

    async with session_manager._api_service_engine.begin() as conn:
        await conn.run_sync(ApiServiceBase.metadata.drop_all)
        await conn.run_sync(ApiServiceBase.metadata.create_all)

    try:
        yield session_manager
    finally:
        await session_manager.close()
