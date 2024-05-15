# ruff: noqa: S101
import hashlib
import json
from importlib import resources

import rapidfuzz
import requests
from deepdiff.operator import BaseOperator

from aana.configs.settings import Settings
from aana.storage.op import drop_all_tables, run_alembic_migrations
from aana.tests.const import ALLOWED_LEVENSTEIN_ERROR_RATE
from aana.utils.json import jsonify


def is_gpu_available() -> bool:
    """Check if a GPU is available.

    Returns:
        bool: True if a GPU is available, False otherwise.
    """
    import torch

    # TODO: find the way to check if GPU is available without importing torch
    return torch.cuda.is_available()


def is_using_deployment_cache() -> bool:
    """Check if the deployment cache is being used.

    Returns:
        bool: True if the deployment cache is being used, False otherwise.
    """
    from aana.configs.settings import settings

    return settings.test.use_deployment_cache


def compare_texts(expected_text: str, text: str):
    """Compare two texts using Levenshtein distance.

    The error rate is allowed to be less than ALLOWED_LEVENSTEIN_ERROR_RATE.

    Args:
        expected_text (str): the expected text
        text (str): the actual text

    Raises:
        AssertionError: if the error rate is too high
    """
    dist = rapidfuzz.distance.Levenshtein.distance(text, expected_text)
    assert dist < len(expected_text) * ALLOWED_LEVENSTEIN_ERROR_RATE, (
        expected_text,
        text,
        dist,
    )


class LevenshteinOperator(BaseOperator):
    """Deepdiff operator class for Levenshtein distances."""

    def give_up_diffing(self, level, diff_instance) -> bool:
        """Short-circuit if we're certain to exceed error rate based on length."""
        dist = rapidfuzz.distance.Levenshtein.distance(level.t1, level.t2)
        if dist < len(level.t1) * ALLOWED_LEVENSTEIN_ERROR_RATE:
            return True
        return False


def call_streaming_endpoint(
    port: int, route_prefix: str, endpoint: str, data: dict
) -> list:
    """Call a streaming endpoint.

    Args:
        port (int): Port of the server.
        route_prefix (str): Route prefix of the server.
        endpoint (str): Endpoint to call.
        data (dict): Data to send to the endpoint.

    Returns:
        list: List of output chunks. If an error occurs, the list will contain
            only one element, which is the error response.
    """
    output = []
    r = requests.post(
        f"http://localhost:{port}{route_prefix}{endpoint}",
        data={"body": json.dumps(data)},
        timeout=30,
        stream=True,
    )
    for chunk in r.iter_content(chunk_size=None):
        chunk_output = json.loads(chunk.decode("utf-8"))
        output.append(chunk_output)
        if "error" in chunk_output:
            return [chunk_output]
    return output


def call_endpoint(
    target: str,
    port: int,
    route_prefix: str,
    endpoint_path: str,
    is_streaming: bool,
    data: dict,
) -> dict | list:
    """Call an endpoint.

    Args:
        target (str): the name of the target.
        port (int): Port of the server.
        route_prefix (str): Route prefix of the server.
        endpoint_path (str): Endpoint to call.
        is_streaming (bool): If True, the endpoint is a streaming endpoint.
        data (dict): Data to send to the endpoint.

    Returns:
        dict | list: Output of the endpoint. If the endpoint is a streaming endpoint, the output will be a list of output chunks.
            If the endpoint is not a streaming endpoint, the output will be a dict.
            If an error occurs, the output will be a dict with the error message.
    """
    if is_streaming:
        return call_streaming_endpoint(port, route_prefix, endpoint_path, data)
    else:
        r = requests.post(
            f"http://localhost:{port}{route_prefix}{endpoint_path}",
            data={"body": json.dumps(data)},
            timeout=30,
        )
        return r.json()


def compare_streaming_output(expected_output: list[dict], output: list[dict]):
    """Compare streaming output to expected output.

    Args:
        expected_output (list[dict]): Expected output.
        output (list[dict]): Actual output.

    Raises:
        AssertionError: if the output is different from the expected output.
    """
    # check that the output and expected output have the same length
    assert len(expected_output) == len(output), (
        len(expected_output),
        len(output),
        output,
    )

    # if error is expected or occurs, compare the errors
    if "error" in expected_output[0] or "error" in output[0]:
        # check that the expected output contains an error
        assert "error" in expected_output[0]
        # check that the output contains an error
        assert "error" in output[0]
        # check that the output and expected output have the same error
        assert expected_output[0]["error"] == output[0]["error"]
        return

    # Streaming output can come from multiple generators
    # (e.g. from transcription and captioning for video indexing)
    # so we need to make sure that we compare the output from each generator
    # to the corresponding expected output.
    # Simplest way to do this is to sort the output and expected output by the keys.
    expected_output.sort(key=lambda x: sorted(x.keys()))
    output.sort(key=lambda x: sorted(x.keys()))

    # compare the output to the expected output
    for expected, actual in zip(expected_output, output, strict=True):
        # check that the output and expected output are dicts
        assert isinstance(expected, dict)
        assert isinstance(actual, dict)
        # compare the keys
        assert expected.keys() == actual.keys()
        # compare the values (might need to update to a more intelligent comparison
        # of each value in the dict later, for now just compare the json strings with levenshtein distance)
        expected_json = json.dumps(expected, sort_keys=True)
        actual_json = json.dumps(actual, sort_keys=True)
        compare_texts(expected_json, actual_json)


def compare_output(expected_output: dict, output: dict):
    """Compare output to expected output.

    Args:
        expected_output (dict): Expected output.
        output (dict): Actual output.

    Raises:
        AssertionError: if the output is different from the expected output.
    """
    # check that the output and expected output are dicts
    assert isinstance(expected_output, dict)
    assert isinstance(output, dict)
    # if error is expected or occurs, compare the errors
    if "error" in expected_output or "error" in output:
        # check that the expected output contains an error
        assert "error" in expected_output
        # check that the output contains an error
        assert "error" in output
        # check that the output and expected output have the same error
        assert expected_output["error"] == output["error"]
        return

    # compare the keys
    assert expected_output.keys() == output.keys()
    # compare the values (might need to update to a more intelligent comparison
    # of each value in the dict later, for now just compare the json strings with levenshtein distance)
    expected_json = json.dumps(expected_output, sort_keys=True)
    actual_json = json.dumps(output, sort_keys=True)
    compare_texts(expected_json, actual_json)


def clear_database(aana_settings: Settings):
    """Clear the database.

    It drops all tables and runs alembic migrations to create the tables again.

    Args:
        aana_settings (Settings): AANA settings.
    """
    drop_all_tables(aana_settings)
    run_alembic_migrations(aana_settings)


def check_output(
    target: str,
    endpoint_path: str,
    key: str,
    output: dict | list,
    is_streaming: bool,
    ignore_expected_output: bool = False,
    expected_error: str | None = None,
):
    """Compare output with expected output.

    Args:
        target (str): the name of the target.
        endpoint_path (str): Endpoint path.
        key (str): Key of the expected output.
        output (dict | list): Output of the endpoint.
        is_streaming (bool): If True, the endpoint is a streaming endpoint.
        ignore_expected_output (bool, optional): If True, do not compare the output with the expected output. Defaults to False.
        expected_error (str | None, optional): Expected error. If not None, the output will be compared with the expected error
            and the expected output will be ignored. Defaults to None.

    Raises:
        AssertionError: if the output is different from the expected output.
    """
    # if we expect an error, then we only check the error
    if expected_error:
        if is_streaming:
            assert output[0]["error"] == expected_error, output
        else:
            assert output["error"] == expected_error, output
    # if we don't expect an error and we don't ignore the expected output, then we compare
    elif not ignore_expected_output:
        endpoint_key = endpoint_path.replace("/", "_")[
            1:
        ]  # e.g. /video/indexing -> video_indexing
        expected_output_path = resources.path(
            f"aana.tests.files.expected.endpoints.{target}",
            f"{endpoint_key}_{key}.json",
        )
        # Below block stores expected endpoint results as json files (when path does not exist yet).
        # if not expected_output_path.exists():
        #    with expected_output_path.open("w") as f:
        #        json.dump(output, f, indent=4, sort_keys=True)

        if not expected_output_path.exists():
            raise FileNotFoundError(expected_output_path)

        expected_output = json.loads(expected_output_path.read_text())
        try:
            if is_streaming:
                compare_streaming_output(expected_output, output)
            else:
                compare_output(expected_output, output)
        except AssertionError as e:
            raise AssertionError(  # noqa: TRY003
                f"Output of {endpoint_path} with key {key} is different from the expected output: {e}"
            ) from e

    # if we don't expect an error and we ignore the expected output,
    # then only check that the output does not contain an error
    else:
        if is_streaming:
            for chunk in output:
                assert "error" not in chunk, chunk
        else:
            assert "error" not in output, output


def call_and_check_endpoint(
    target: str,
    port: int,
    route_prefix: str,
    endpoint_path: str,
    data: dict,
    is_streaming: bool,
    ignore_expected_output: bool = False,
    expected_error: str | None = None,
) -> dict | list:
    """Call endpoint and compare the output with the expected output.

    Args:
        target (str): the name of the target.
        port (int): Port of the server.
        route_prefix (str): Route prefix of the server.
        endpoint_path (str): Endpoint to call.
        data (dict): Data to send to the endpoint.
        is_streaming (bool): If True, the endpoint is a streaming endpoint.
        ignore_expected_output (bool, optional): If True, do not compare the output with the expected output. Defaults to False.
        expected_error (str | None, optional): Expected error. If not None, the output will be compared with the expected error
            and the expected output will be ignored. Defaults to None.

    Returns:
        dict | list: Output of the endpoint. If the endpoint is a streaming endpoint, the output will be a list of output chunks.
            If the endpoint is not a streaming endpoint, the output will be a dict.
            If an error occurs, the output will be a dict with the error message.
    """
    data_json = jsonify(data)
    # "aana.tests.files.videos" will be resolved to a different path on different systems
    # so we need to replace it with a path that is the same on all systems
    # to make sure that the hash of the data is the same on all systems
    data_json = data_json.replace(
        str(resources.path("aana.tests.files", "")),
        "/aana/tests/files/",
    )
    data_hash = hashlib.md5(
        data_json.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    output = call_endpoint(
        target=target,
        port=port,
        route_prefix=route_prefix,
        endpoint_path=endpoint_path,
        is_streaming=is_streaming,
        data=data,
    )
    check_output(
        target=target,
        endpoint_path=endpoint_path,
        key=data_hash,
        output=output,
        is_streaming=is_streaming,
        ignore_expected_output=ignore_expected_output,
        expected_error=expected_error,
    )
    return output


def get_deployments_by_type(deployment_type: str) -> list:
    """Get deployments by type from the list of all available deployments.

    Args:
        deployment_type (str): Deployment type. For example, "VLLMDeployment".

    Returns:
        list: List of deployments with the given type.
    """
    from aana.configs.deployments import available_deployments

    return [
        (name, deployment)
        for name, deployment in available_deployments.items()
        if deployment.name == deployment_type
    ]
