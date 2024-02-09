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
from aana.models.core.audio import Audio
from aana.models.pydantic.whisper_params import WhisperParams
from aana.tests.utils import LevenshteinOperator, is_gpu_available
from aana.utils.general import pydantic_to_dict

EPSILON = 0.01


def compare_transcriptions(expected_transcription, transcription, words=True):
    """Compare two transcriptions.

    Texts and words are compared using Levenshtein distance.

    Args:
        expected_transcription (dict): the expected transcription
        transcription (dict): the actual transcription

    Raises:
        AssertionError: if transcriptions differ too much
    """

    if words:
        levenshtein_operator = [LevenshteinOperator([r"\['text'\]$", r"\['word'\]$"])]
    else:
        levenshtein_operator = [LevenshteinOperator([r"\['text'\]$"])]
    
    diff = DeepDiff(
        expected_transcription,
        transcription,
        math_epsilon=EPSILON,
        ignore_numeric_type_changes=True,
        custom_operators= levenshtein_operator,
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
@pytest.mark.parametrize("audio_file", ["physicsworks_16k.wav"])
async def test_whisper_deployment(audio_file):
    """Test whisper deployment."""
    for deployment in deployments.values():
        # skip if not a whisper deployment
        if deployment.name != "WhisperDeployment":
            continue

        handle = ray_setup(deployment)

        model_size = deployment.user_config["model_size"]

        expected_output_path = resources.path(
            f"aana.tests.files.expected.whisper.{model_size}", f"{audio_file}.json"
        )
        assert (
            expected_output_path.exists()
        ), f"Expected output not found: {expected_output_path}"
        with Path(expected_output_path) as path, path.open() as f:
            expected_output = json.load(f)

        # Test transcribe method
        path = resources.path("aana.tests.files.audios", audio_file)
        assert path.exists(), f"Audio not found: {path}"
        audio = Audio(path=path)

        output = await handle.transcribe.remote(
            media=audio, params=WhisperParams(word_timestamps=True)
        )
        output = pydantic_to_dict(output)

        compare_transcriptions(expected_output, output)

        # Test transcribe_stream method
        stream = handle.options(stream=True).transcribe_stream.remote(
            media=audio, params=WhisperParams(word_timestamps=True)
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
        audios = [audio, audio]

        batch_output = await handle.transcribe_batch.remote(
            media_batch=audios, params=WhisperParams(word_timestamps=True)
        )
        batch_output = pydantic_to_dict(batch_output)

        for i in range(len(audios)):
            output = {k: v[i] for k, v in batch_output.items()}
            compare_transcriptions(expected_output, output)

        # Test batched_inference method: expected asr output is different
        expected_batched_output_path = 
        # get expected vad_segments
        vad_path = resources.path(
            "aana.tests.files.expected.vad", f"{audio_file}_vad.json"
        )
        assert vad_path.exists(), f"vad expected predictions not found: {vad_path}"

        with Path(vad_path) as path, path.open() as f:
            expected_output_vad = json.load(f)

        # load expected_transcription_batched for batched version. check info

        stream = handle.options(stream=True).batched_inference.remote(
            media=audio,  # chnge alls
            vad_segments=expected_output_vad,
            batch_size=16,
            params=WhisperParams,
        )
        
        # Combine individual segments and compare with the final dict
        grouped_dict = defaultdict(list)
        async for chunk in stream:
            chunk = await chunk
            output = pydantic_to_dict(chunk)
            grouped_dict["segments"].append(output.get("segments")[0]["text"])

        grouped_dict["transcription_info"] = output.get("transcription_info")
        compare_transcriptions(expected_transcription_batched, dict(grouped_dict), words=False)
