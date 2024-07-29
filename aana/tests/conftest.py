# This file is used to define fixtures that are used in the integration tests.
# The fixtures are used to setup Ray and Ray Serve, and to call the endpoints.
# The fixtures depend on each other, to setup the environment for the tests.
# Here is a dependency graph of the fixtures:
# app_setup (module scope, starts Ray and Ray Serve app for a specific target, args: deployments, endpoints)
#     -> call_endpoint (module scope, calls endpoint, args: endpoint_path, data, ignore_expected_output, expected_error)

# ruff: noqa: S101
import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import requests
from pydantic import ValidationError
from sqlalchemy.orm import Session

from aana.api.api_generation import Endpoint
from aana.configs.db import DbSettings, SQLiteConfig
from aana.configs.settings import settings
from aana.configs.settings import settings as aana_settings
from aana.sdk import AanaSDK
from aana.storage.op import DbType, run_alembic_migrations
from aana.utils.core import import_from
from aana.utils.json import jsonify


def send_api_request(
    endpoint: Endpoint,
    app: AanaSDK,
    data: dict[str, Any],
    timeout: int = 30,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Call an endpoint, handling both streaming and non-streaming responses."""
    url = f"http://localhost:{app.port}{endpoint.path}"
    payload = {"body": json.dumps(data)}

    if endpoint.is_streaming_response():
        output = []
        with requests.post(url, data=payload, timeout=timeout, stream=True) as r:
            for chunk in r.iter_content(chunk_size=None):
                chunk_output = json.loads(chunk.decode("utf-8"))
                output.append(chunk_output)
                if "error" in chunk_output:
                    return [chunk_output]
        return output
    else:
        response = requests.post(url, data=payload, timeout=timeout)
        return response.json()


def verify_output(
    endpoint: Endpoint,
    response: dict[str, Any] | list[dict[str, Any]],
    expected_error: str | None = None,
) -> None:
    """Verify the output of an endpoint call."""
    is_streaming = endpoint.is_streaming_response()
    ResponseModel = endpoint.get_response_model()
    if expected_error:
        error = response[0]["error"] if is_streaming else response["error"]
        assert error == expected_error, response
    else:
        try:
            if is_streaming:
                for item in response:
                    ResponseModel.model_validate(item, strict=True)
            else:
                ResponseModel.model_validate(response, strict=True)
        except ValidationError as e:
            raise AssertionError(  # noqa: TRY003
                f"Validation failed. Errors:\n{e}\n\nResponse: {response}"
            ) from e


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

        # Reload the settings to update the database path
        import aana.configs.settings

        importlib.reload(aana.configs.settings)

        # Import and start the app
        app = import_from(app_module, app_name)
        app.connect(port=8000, show_logs=True, num_cpus=10)
        app.migrate()
        app.deploy()

        return app, tmp_database_path

    return create_app


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
def db_session():
    """Creates a new database file and session for each test."""
    with tempfile.NamedTemporaryFile(dir=settings.tmp_data_dir) as tmp:
        # Configure the database to use the temporary file
        settings.db_config.datastore_config = SQLiteConfig(path=tmp.name)
        # Reset the engine
        settings.db_config.engine = None

        # Run migrations to set up the schema
        run_alembic_migrations(settings)

        # Create a new session
        engine = settings.db_config.get_engine()
        with Session(engine) as session:
            yield session


# TODO: add support for postgresql using pytest-postgresql
# @pytest.fixture(scope="function")
# def db_session(postgresql):
#     """Creates a new database file and session for each test."""
#     settings.db_config.datastore_type = DbType.POSTGRESQL
#     settings.db_config.datastore_config = PostgreSQLConfig(
#         host=postgresql.info.host,
#         port=postgresql.info.port,
#         user=postgresql.info.user,
#         password=postgresql.info.password,
#         database=postgresql.info.dbname,
#     )

#     # Reset the engine
#     settings.db_config.engine = None

#     # Run migrations to set up the schema
#     run_alembic_migrations(settings)

#     # Create a new session
#     engine = settings.db_config.get_engine()
#     with Session(engine) as session:
#         yield session
