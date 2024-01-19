# ruff: noqa: S101
import json
from collections import defaultdict
from importlib import resources
from pathlib import Path

import pytest
import ray
from deepdiff import DeepDiff
from ray import serve

from aana.configs.deployments import deployments
from aana.models.core.video import Video
from aana.models.pydantic.whisper_params import WhisperParams
from aana.tests.utils import LevenshteinOperator, is_gpu_available
from aana.utils.general import pydantic_to_dict

EPSILON = 0.01


def compare_transcriptions(expected_transcription, transcription):
    """Compare two transcriptions.

    Texts and words are compared using Levenshtein distance.

    Args:
        expected_transcription (dict): the expected transcription
        transcription (dict): the actual transcription

    Raises:
        AssertionError: if transcriptions differ too much
    """
    diff = DeepDiff(
        expected_transcription,
        transcription,
        math_epsilon=EPSILON,
        ignore_numeric_type_changes=True,
        custom_operators=[LevenshteinOperator([r"\['text'\]$", r"\['word'\]$"])],
    )
    assert not diff, diff


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
@pytest.mark.parametrize("video_file", ["physicsworks.webm"])
async def test_whisper_deployment(video_file):
    """Test whisper deployment."""
    for deployment in deployments.values():
        # skip if not a VLLM deployment
        if deployment.name != "WhisperDeployment":
            continue

        handle = ray_setup(deployment)

        model_size = deployment.user_config["model_size"]

        expected_output_path = resources.path(
            f"aana.tests.files.expected.whisper.{model_size}", f"{video_file}.json"
        )
        assert (
            expected_output_path.exists()
        ), f"Expected output not found: {expected_output_path}"
        with Path(expected_output_path) as path, path.open() as f:
            expected_output = json.load(f)

        # Test transcribe method
        path = resources.path("aana.tests.files.videos", video_file)
        assert path.exists(), f"Video not found: {path}"
        video = Video(path=path)

        output = await handle.transcribe.remote(
            media=video, params=WhisperParams(word_timestamps=True)
        )
        output = pydantic_to_dict(output)

        compare_transcriptions(expected_output, output)

        # Test transcribe_stream method
        path = resources.path("aana.tests.files.videos", video_file)
        assert path.exists(), f"Video not found: {path}"
        video = Video(path=path)

        stream = handle.options(stream=True).transcribe_stream.remote(
            media=video, params=WhisperParams(word_timestamps=True)
        )

        # Combine individual segments and compare with the final dict
        grouped_dict = defaultdict(list)
        transcript = ""
        async for chunk in stream:
            chunk = await chunk
            output = pydantic_to_dict(chunk)
            transcript += output["transcription"]["text"]
            grouped_dict["segments"].append(output.get("segments")[0])

        grouped_dict["transcription"] = {"text": transcript}
        grouped_dict["transcription_info"] = output.get("transcription_info")
        compare_transcriptions(expected_output, dict(grouped_dict))

        # Test transcribe_batch method
        videos = [video, video]

        batch_output = await handle.transcribe_batch.remote(
            media_batch=videos, params=WhisperParams(word_timestamps=True)
        )
        batch_output = pydantic_to_dict(batch_output)

        for i in range(len(videos)):
            output = {k: v[i] for k, v in batch_output.items()}
            compare_transcriptions(expected_output, output)
