import json
from importlib import resources
from pathlib import Path

import pytest
import ray
from deepdiff import DeepDiff
from ray import serve

from aana.configs.deployments import deployments
from aana.models.core.audio import Audio
from aana.models.pydantic.vad_params import VadParams
from aana.tests.utils import LevenshteinOperator, is_gpu_available
from aana.utils.general import pydantic_to_dict

EPSILON = 0.01


def compare_vad_outputs(expected_output, predictions):
    """Compare two vad outputs.

    Start, end and number of segments inside are compared using Levenshtein distance.

    Args:
        expected_output (list[dict]): the expected vad_output
        predictions (list[dict]): the predictions of vad

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
        print(diff)
        assert not diff, diff  # noqa: S101


def ray_setup(deployment):
    """Setup Ray instance for the test."""
    # Setup ray environment and serve
    ray.init(ignore_reinit_error=True)
    app = deployment.bind()
    port = 34422
    test_name = deployment.name
    route_prefix = f"/{test_name}"
    handle = serve.run(app, port=port, name=test_name, route_prefix=route_prefix)
    return handle


@pytest.mark.skipif(not is_gpu_available(), reason="GPU is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("video_file", ["physicsworks_16k.wav"])  # need this.
async def test_whisper_deployment(video_file):
    """Test whisper deployment."""
    for deployment in deployments.values():
        # skip if not vad deployment
        if deployment.name != "VadDeployment":
            continue

        handle = ray_setup(deployment)

        expected_output_path = resources.path(
            "aana.tests.files.expected.vad", f"{video_file}_vad.json"
        )
        assert (  # noqa: S101
            expected_output_path.exists()
        ), f"Expected output not found: {expected_output_path}"
        with Path(expected_output_path) as path, path.open() as f:
            expected_output = json.load(f)

        # asr_preprocess_vad method
        path = resources.path("aana.tests.files.audios", video_file)
        assert path.exists(), f"Audio not found: {path}"
        audio = Audio(path=path)

        output = await handle.asr_preprocess_vad.remote(media=audio, params=VadParams)
        output = pydantic_to_dict(output)

        compare_vad_outputs(expected_output, output)
