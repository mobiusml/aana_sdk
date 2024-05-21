# ruff: noqa: S101
import json
from importlib import resources
from pathlib import Path

import pytest
from ray import serve

from aana.core.models.audio import Audio
from aana.core.models.base import pydantic_to_dict
from aana.core.models.vad import VadParams
from aana.tests.utils import (
    get_deployments_by_type,
    is_gpu_available,
    is_using_deployment_cache,
)


def compare_vad_outputs(expected_output, predictions):
    """Compare two vad outputs.

    Start and end positions are compared using Levenshtein distance.

    Args:
        expected_output (dict(list[dict])): the expected vad_output
        predictions (dict(list[dict])): the predictions of vad

    Raises:
        AssertionError: if vad_outputs differ too much
    """
    # Number of 30 sec segments should not change.
    assert len(expected_output["segments"]) == len(predictions["segments"])

    if len(expected_output["segments"]) != 0:  # non-empty files
        # all start and end times should be similar and number of segments should be same.
        for expected, predicted in zip(
            expected_output["segments"], predictions["segments"], strict=False
        ):
            assert (
                abs(
                    expected["time_interval"]["start"]
                    - predicted["time_interval"]["start"]
                )
                < 2.0
            )  # all segment starts within 2 sec
            assert (
                abs(
                    expected["time_interval"]["end"] - predicted["time_interval"]["end"]
                )
                < 2.0
            )  # all segment ends within 2 sec

            # check same number of small voiced segments within each vad segment
            assert len(expected["segments"]) == len(predicted["segments"])


@pytest.fixture(scope="function", params=get_deployments_by_type("VadDeployment"))
def setup_vad_deployment(app_setup, request):
    """Setup vad deployment."""
    name, deployment = request.param
    deployments = [
        {
            "name": "vad_deployment",
            "instance": deployment,
        }
    ]
    endpoints = []

    return name, deployment, app_setup(deployments, endpoints)


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("audio_file", ["physicsworks.wav", "squirrel.wav"])
async def test_vad_deployment(setup_vad_deployment, audio_file):
    """Test vad deployment."""
    handle = serve.get_app_handle("vad_deployment")

    audio_file_name = Path(audio_file).stem
    expected_output_path = resources.path(
        "aana.tests.files.expected.vad", f"{audio_file_name}_vad.json"
    )
    assert (
        expected_output_path.exists()
    ), f"Expected output not found: {expected_output_path}"
    with Path(expected_output_path) as path, path.open() as f:
        expected_output = json.load(f)

    # asr_preprocess_vad method to test
    path = resources.path("aana.tests.files.audios", audio_file)
    assert path.exists(), f"Audio not found: {path}"

    audio = Audio(path=path)

    output = await handle.asr_preprocess_vad.remote(audio=audio, params=VadParams())
    output = pydantic_to_dict(output)
    compare_vad_outputs(expected_output, output)
