# ruff: noqa: S101
import json
from collections import defaultdict
from importlib import resources
from pathlib import Path

import pytest
from deepdiff import DeepDiff
from ray import serve

from aana.core.models.audio import Audio
from aana.core.models.base import pydantic_to_dict
from aana.core.models.vad import VadSegment
from aana.core.models.whisper import BatchedWhisperParams, WhisperParams
from aana.tests.utils import (
    LevenshteinOperator,
    get_deployments_by_type,
    is_gpu_available,
    is_using_deployment_cache,
)

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
    custom_operators = [LevenshteinOperator([r"\['text'\]$", r"\['word'\]$"])]

    diff = DeepDiff(
        expected_transcription,
        transcription,
        math_epsilon=EPSILON,
        ignore_numeric_type_changes=True,
        custom_operators=custom_operators,
    )
    assert not diff, diff


@pytest.fixture(scope="function", params=get_deployments_by_type("WhisperDeployment"))
def setup_whisper_deployment(app_setup, request):
    """Setup whisper deployment."""
    name, deployment = request.param
    deployments = [
        {
            "name": "whisper_deployment",
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
@pytest.mark.parametrize("audio_file", ["squirrel.wav", "physicsworks.wav"])
async def test_whisper_deployment(setup_whisper_deployment, audio_file):
    """Test whisper deployment."""
    name, deployment, app = setup_whisper_deployment

    handle = serve.get_app_handle("whisper_deployment")

    model_size = deployment.user_config["model_size"]
    audio_file_name = Path(audio_file).stem
    expected_output_path = resources.path(
        f"aana.tests.files.expected.whisper.{model_size}", f"{audio_file_name}.json"
    )
    assert (
        expected_output_path.exists()
    ), f"Expected output not found: {expected_output_path}"
    with Path(expected_output_path) as path, path.open() as f:
        expected_output = json.load(f)

    # Test transcribe method
    path = resources.path("aana.tests.files.audios", audio_file)
    assert path.exists(), f"Audio not found: {path}"
    audio = Audio(path=path, media_id=audio_file)

    output = await handle.transcribe.remote(
        audio=audio, params=WhisperParams(word_timestamps=True, temperature=0.0)
    )
    output = pydantic_to_dict(output)

    compare_transcriptions(expected_output, output)

    # Test transcribe_stream method)
    stream = handle.options(stream=True).transcribe_stream.remote(
        audio=audio, params=WhisperParams(word_timestamps=True, temperature=0.0)
    )

    # Combine individual segments and compare with the final dict
    grouped_dict = defaultdict(list)
    transcript = ""
    async for chunk in stream:
        output = pydantic_to_dict(chunk)
        transcript += output["transcription"]["text"]
        grouped_dict["segments"].extend(output.get("segments", []))

    grouped_dict["transcription"] = {"text": transcript}
    grouped_dict["transcription_info"] = output.get("transcription_info")
    compare_transcriptions(expected_output, dict(grouped_dict))

    # Test transcribe_batch method

    # Test transcribe_in_chunks method: Note that the expected asr output is different
    expected_batched_output_path = resources.path(
        f"aana.tests.files.expected.whisper.{model_size}",
        f"{audio_file_name}_batched.json",
    )
    assert (
        expected_batched_output_path.exists()
    ), f"Expected output not found: {expected_batched_output_path}"
    with Path(expected_batched_output_path) as path, path.open() as f:
        expected_output_batched = json.load(f)

    # Get expected vad segments
    vad_path = resources.path(
        "aana.tests.files.expected.vad", f"{audio_file_name}_vad.json"
    )
    assert vad_path.exists(), f"vad expected predictions not found: {vad_path}"

    with Path(vad_path) as path, path.open() as f:
        expected_output_vad = json.load(f)

    final_input = [
        VadSegment(time_interval=seg["time_interval"], segments=seg["segments"])
        for seg in expected_output_vad["segments"]
    ]

    batched_stream = handle.options(stream=True).transcribe_in_chunks.remote(
        audio=audio,
        segments=final_input,
        batch_size=16,
        params=BatchedWhisperParams(temperature=0.0),
    )

    # Combine individual segments and compare with the final dict
    transcript = ""
    grouped_dict = defaultdict(list)
    async for chunk in batched_stream:
        output = pydantic_to_dict(chunk)
        transcript += output["transcription"]["text"]
        grouped_dict["segments"].extend(output.get("segments", []))

    grouped_dict["transcription"] = {"text": transcript}
    grouped_dict["transcription_info"] = output.get("transcription_info")

    compare_transcriptions(
        expected_output_batched,
        dict(grouped_dict),
    )
