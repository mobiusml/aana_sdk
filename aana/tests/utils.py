# ruff: noqa: S101
import json
from typing import Any

import rapidfuzz
import requests
from deepdiff.operator import BaseOperator
from pydantic import ValidationError

from aana.api.api_generation import Endpoint
from aana.sdk import AanaSDK
from aana.tests.const import ALLOWED_LEVENSTEIN_ERROR_RATE


def is_gpu_available() -> bool:
    """Check if a GPU is available.

    Returns:
        bool: True if a GPU is available, False otherwise.
    """
    import torch

    # TODO: find the way to check if GPU is available without importing torch
    return torch.cuda.is_available()


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
