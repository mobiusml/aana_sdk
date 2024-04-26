# ruff: noqa: S101
# Test whisper endpoints.

import time

import pytest

from aana.tests.utils import is_gpu_available, is_using_deployment_cache

TARGET = "stablediffusion2"

IMAGE_GENERATE_ENDPOINT = "/generate_image_rate_limited"
NON_RATE_LIMITED_ENDPOINT = "/generate_image"


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.parametrize("call_endpoint", [TARGET], indirect=True)
@pytest.mark.parametrize(
    "prompt, rate_limited_endpoint, non_rate_limited_endpoint",
    [
        (
            "Les Demoiselles d'Avignon but by Hans Holbein the Younger",
            IMAGE_GENERATE_ENDPOINT,
            NON_RATE_LIMITED_ENDPOINT,
        ),
    ],
)
def test_image_generate(
    one_request_worker,
    call_endpoint,
    prompt,
    rate_limited_endpoint,
    non_rate_limited_endpoint,
):
    """Test image generate2 endpoint. Also tests rate limiting."""
    # Generate image. Ignore output for this because it's meaningless.
    call_endpoint(
        rate_limited_endpoint, {"prompt": prompt}, ignore_expected_output=True
    )

    # Call non-rate-limited endpoint, it should not trigger TooManyRequestsException
    call_endpoint(
        non_rate_limited_endpoint, {"prompt": prompt}, ignore_expected_output=True
    )

    # Call again, it should trigger TooManyRequestsException
    call_endpoint(
        rate_limited_endpoint,
        {"prompt": prompt},
        expected_error="TooManyRequestsException",
    )

    # Wait 30s and try again
    time.sleep(30)

    # Call again, it should work again (still ignoring output).
    call_endpoint(
        rate_limited_endpoint, {"prompt": prompt}, ignore_expected_output=True
    )
