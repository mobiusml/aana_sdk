import json
import os
from importlib import resources
from pathlib import Path

import pytest
import ray
from deepdiff import DeepDiff
from ray import serve

from aana.configs.deployments import deployments
from aana.models.core.audio import Audio
from aana.models.pydantic.vad_params import VadParams
from aana.tests.utils import (
    LevenshteinOperator,
    get_deployments_by_type,
    is_gpu_available,
    is_using_deployment_cache,
)
from aana.utils.general import pydantic_to_dict

EPSILON = 0.01


def compare_vad_outputs(expected_output, predictions):
    """Compare two vad outputs.

    Start and end positions are compared using Levenshtein distance.

    Args:
        expected_output (dict(list[dict])): the expected vad_output
        predictions (dict(list[dict])): the predictions of vad

    Raises:
        AssertionError: if vad_outputs differ too much
    """
    for expected_output_seg, predictions_seg in zip(
        expected_output, predictions, strict=False
    ):
        diff = DeepDiff(
            expected_output_seg,
            predictions_seg,
            math_epsilon=EPSILON,
            ignore_numeric_type_changes=True,
            custom_operators=[LevenshteinOperator([r"\['start'\]$", r"\['end'\]$"])],
        )
        assert not diff, diff  # noqa: S101


@pytest.fixture(scope="function", params=get_deployments_by_type("VadDeployment"))
def setup_vad_deployment(setup_deployment, request):
    """Setup vad deployment."""
    name, deployment = request.param
    return name, deployment, *setup_deployment(deployment, bind=True)


@pytest.mark.skipif(
    not is_gpu_available() and not is_using_deployment_cache(),
    reason="GPU is not available",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("audio_file", ["physicsworks.wav"])
async def test_vad_deployment(setup_vad_deployment, audio_file):
    """Test vad deployment."""
    name, deployment, handle, port, route_prefix = setup_vad_deployment
    audio_file_path = os.path.splitext(audio_file)[0]  # noqa: PTH122
    expected_output_path = resources.path(
        "aana.tests.files.expected.vad", f"{audio_file_path}_vad.json"
    )
    assert (  # noqa: S101
        expected_output_path.exists()
    ), f"Expected output not found: {expected_output_path}"
    with Path(expected_output_path) as path, path.open() as f:
        expected_output = json.load(f)

    # asr_preprocess_vad method
    path = resources.path("aana.tests.files.audios", audio_file)
    assert path.exists(), f"Audio not found: {path}"
    audio = Audio(path=path)

    output = await handle.asr_preprocess_vad.remote(media=audio, params=VadParams())
    output = pydantic_to_dict(output)
    compare_vad_outputs(expected_output, output)