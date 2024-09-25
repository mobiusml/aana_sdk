# ruff: noqa: S101
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import rapidfuzz
import requests
from pydantic import ValidationError

from aana.api.api_generation import Endpoint
from aana.configs.settings import settings
from aana.core.models.base import pydantic_to_dict
from aana.sdk import AanaSDK
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


def round_floats(obj, decimals=2):
    """Round floats in a nested structure."""
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, defaultdict):
        return defaultdict(
            obj.default_factory, {k: round_floats(v, decimals) for k, v in obj.items()}
        )
    elif isinstance(obj, Mapping):
        return type(obj)((k, round_floats(v, decimals)) for k, v in obj.items())
    elif isinstance(obj, Sequence) and not isinstance(obj, str):  # noqa: SIM114
        return type(obj)(round_floats(x, decimals) for x in obj)
    elif isinstance(obj, set):
        return type(obj)(round_floats(x, decimals) for x in obj)
    elif hasattr(obj, "__dict__"):
        obj.__dict__.update(
            (k, round_floats(v, decimals)) for k, v in obj.__dict__.items()
        )
        return obj
    else:
        return obj


def compare_texts(
    expected_text: str,
    text: str,
    allowed_error_rate: float = ALLOWED_LEVENSTEIN_ERROR_RATE,
):
    """Compare two texts using Levenshtein distance.

    The error rate is allowed to be less than ALLOWED_LEVENSTEIN_ERROR_RATE.

    Args:
        expected_text (str): the expected text
        text (str): the actual texts
        allowed_error_rate (float): the allowed error rate (default: ALLOWED_LEVENSTEIN_ERROR_RATE)

    Raises:
        AssertionError: if the error rate is too high
    """
    dist = rapidfuzz.distance.Levenshtein.distance(text, expected_text)
    assert dist < len(expected_text) * allowed_error_rate, (
        expected_text,
        text,
        dist,
    )


def compare_results(
    expected_result: dict[str, Any],
    result: dict[str, Any],
    allowed_error_rate: float = ALLOWED_LEVENSTEIN_ERROR_RATE,
    apply_float_rounding=True,
    round_decimals=2,
):
    """Compare expected and actual results by jsonifying them and comparing the strings using Levenshtein distance.

    Args:
        expected_result (dict): the expected result
        result (dict): the actual result
        allowed_error_rate (float): the allowed error rate (default: ALLOWED_LEVENSTEIN_ERROR_RATE)
        apply_float_rounding (bool): whether to round floats (default: True)
        round_decimals (int): the number of decimals to round floats to (default: 2)

    Raises:
        AssertionError: if the error rate is too high
    """
    if apply_float_rounding:
        expected_json = jsonify(round_floats(expected_result, round_decimals))
        result_json = jsonify(round_floats(result, round_decimals))
    else:
        expected_json = jsonify(expected_result)
        result_json = jsonify(result)
    compare_texts(expected_json, result_json, allowed_error_rate)


def verify_deployment_results(
    expected_output_path: Path,
    results: dict[str, Any],
    allowed_error_rate: float = ALLOWED_LEVENSTEIN_ERROR_RATE,
):
    """Verify the output of a deployment call against an expected output file.

    If the expected output file doesn't exist and the SAVE_EXPECTED_OUTPUT
    environment variable is set, save the current results as the expected output.
    Creates parent directories if they don't exist.

    Args:
        expected_output_path (Path): Path to the expected output file.
        results (dict): The actual results.
        allowed_error_rate (float): The allowed error rate (default: ALLOWED_LEVENSTEIN_ERROR_RATE)
    """
    # Convert pydantic models to dictionaries before comparison
    results = pydantic_to_dict(results)

    if not expected_output_path.exists():
        if settings.test.save_expected_output:
            # Ensure the parent directory exists
            expected_output_path.parent.mkdir(parents=True, exist_ok=True)

            with expected_output_path.open("w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved results as expected output to: {expected_output_path}")
            return
        else:
            raise FileNotFoundError(  # noqa: TRY003
                f"Expected output not found: {expected_output_path}"
            )

    with expected_output_path.open() as f:
        expected_output = json.load(f)

    compare_results(expected_output, results, allowed_error_rate)


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
