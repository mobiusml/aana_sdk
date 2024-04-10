# ruff: noqa: S101
# Test whisper endpoints.

import time

import pytest

from aana.configs.settings import settings as aana_settings
from aana.tests.utils import is_gpu_available, is_using_deployment_cache

TARGET = "stablediffusion2"

IMAGE_GENERATE_ENDPOINT = "/generate_image"


@pytest.fixture(scope="module")
def one_request_worker():
    """Fixture to update settings to only run one request worker."""
    aana_settings.num_workers = 1
    yield


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.parametrize("call_endpoint", [TARGET], indirect=True)
@pytest.mark.parametrize(
    "prompt, endpoint",
    [
        (
            "Les Demoiselles d'Avignon but by Hans Holbein the Younger",
            IMAGE_GENERATE_ENDPOINT,
        ),
    ],
)
def test_image_generate(one_request_worker, call_endpoint, prompt, endpoint):
    """Test image generate endpoint. Also tests rate limiting."""
    # Generate image. Ignore output for this because it's meaningless.
    call_endpoint(endpoint, {"prompt": prompt}, ignore_expected_output=True)

    # Call again, it should trigger TooManyRequestsException
    call_endpoint(
        endpoint,
        {"prompt": prompt},
        expected_error="TooManyRequestsException",
    )

    # Wait 30s and try again
    time.sleep(30)

    # Call again, it should work again (still ignoring output).
    call_endpoint(endpoint, {"prompt": prompt}, ignore_expected_output=True)
