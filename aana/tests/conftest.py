# This file is used to define fixtures that are used in the integration tests.
# ruff: noqa: S101
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import pytest
from pytest_postgresql import factories
from sqlalchemy.orm import sessionmaker

from aana.configs.db import DbSettings, PostgreSQLConfig, SQLiteConfig
from aana.configs.settings import settings as aana_settings
from aana.sdk import AanaSDK
from aana.storage.models.api_key import ApiServiceBase
from aana.storage.models.base import BaseEntity
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
        aana_settings.db_config._engine = None

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

    # Setup API service database
    tmp_api_service_database_path = Path(tempfile.mkstemp(suffix=".db")[1])
    api_service_db_config = DbSettings(
        datastore_type=DbType.SQLITE,
        datastore_config=SQLiteConfig(path=tmp_api_service_database_path),
    )
    os.environ["API_SERVICE_DB_CONFIG"] = jsonify(api_service_db_config)

    aana_settings.api_service_db_config = api_service_db_config

    ApiServiceBase.metadata.create_all(api_service_db_config.get_engine())

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


def create_db_engine(config, db_type, tmp_path=None, postgresql_info=None):
    """Helper to create engine for different configs and db types."""
    config.datastore_type = db_type

    if db_type == DbType.SQLITE:
        config.datastore_config = SQLiteConfig(path=tmp_path)
    elif db_type == DbType.POSTGRESQL:
        config.datastore_config = PostgreSQLConfig(
            host=postgresql_info.host,
            port=postgresql_info.port,
            user=postgresql_info.user,
            password=postgresql_info.password,
            database=postgresql_info.dbname,
        )

    os.environ["DB_CONFIG"] = jsonify(config)
    config._engine = None  # Reset engine

    return config.get_engine()


@pytest.fixture(scope="function")
def sqlite_db_engine():
    """Create a SQLite database engine."""
    with tempfile.NamedTemporaryFile(dir=aana_settings.tmp_data_dir) as tmp:
        yield create_db_engine(
            config=aana_settings.db_config, db_type=DbType.SQLITE, tmp_path=tmp.name
        )


@pytest.fixture(scope="function")
def postgres_db_engine(postgresql):
    """Create a PostgreSQL database engine."""
    yield create_db_engine(
        config=aana_settings.db_config,
        db_type=DbType.POSTGRESQL,
        postgresql_info=postgresql.info,
    )


@pytest.fixture(scope="function")
def api_service_sqlite_db_engine():
    """Create a SQLite database engine for the API service."""
    with tempfile.NamedTemporaryFile(dir=aana_settings.tmp_data_dir) as tmp:
        yield create_db_engine(
            config=aana_settings.api_service_db_config,
            db_type=DbType.SQLITE,
            tmp_path=tmp.name,
        )


@pytest.fixture(scope="function")
def api_service_postgres_db_engine(postgresql):
    """Create a PostgreSQL database engine for the API service."""
    yield create_db_engine(
        config=aana_settings.api_service_db_config,
        db_type=DbType.POSTGRESQL,
        postgresql_info=postgresql.info,
    )


@pytest.fixture(params=["sqlite_db_engine", "postgres_db_engine"])
def db_engine(request):
    """Fixture to provide both SQLite and PostgreSQL database engines."""
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=["api_service_sqlite_db_engine", "api_service_postgres_db_engine"]
)
def api_service_db_engine(request):
    """Fixture to provide both SQLite and PostgreSQL database engines for the API service."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def db_session(db_engine):
    """Create a new session for each test."""
    run_alembic_migrations(aana_settings)

    SessionLocal = sessionmaker(bind=db_engine)
    with SessionLocal() as session:
        yield session


@pytest.fixture
def db_session_with_api_service(db_engine, api_service_db_engine):
    """Create a new session with both main and api service databases."""
    run_alembic_migrations(aana_settings)
    ApiServiceBase.metadata.create_all(api_service_db_engine)

    SessionLocal = sessionmaker(
        binds={ApiServiceBase: api_service_db_engine, BaseEntity: db_engine},
        bind=db_engine,
    )
    with SessionLocal() as session:
        yield session
